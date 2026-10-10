"""Hidden Markov Model market regimes learned from returns, strictly causal.

Two features per bar: the smoothed log return and the log realised volatility
over `feature_window` bars. A Gaussian HMM is refitted walk-forward on past bars
only, and state probabilities come from the forward filter
P(state_t | x_1..x_t), never from Viterbi or forward-backward smoothing (both
look at bars after t).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from cappatrader.config import HMMConfig
from cappatrader.regime.detector import HIGH_VOL, RANGE, TREND_DOWN, TREND_UP, UNKNOWN

log = logging.getLogger(__name__)

FEATURES = ["hmm_ret", "hmm_logvol"]
_LOG_FLOOR = 1e-300


def hmm_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Smoothed log return and log realised volatility of 1-bar log returns over `window` bars."""
    r = np.log(df["close"]).diff()
    vol = r.rolling(window, min_periods=window).std()
    out = pd.DataFrame({
        "hmm_ret": r.rolling(window, min_periods=window).mean(),
        "hmm_logvol": np.log(vol.where(vol > 0)),
    }, index=df.index)
    return out.replace([np.inf, -np.inf], np.nan)


def refit_schedule(
    valid: np.ndarray, min_train: int, train_bars: int, retrain_every: int
) -> Iterator[tuple[np.ndarray, int, int]]:
    """Yield (train_rows, start, stop) for walk-forward refits.

    The first chunk starts right after the `min_train`-th valid row; each chunk is
    `retrain_every` rows long and is scored by a model fitted on the last
    `train_bars` valid rows strictly before its start.
    """
    valid_idx = np.flatnonzero(valid)
    n = len(valid)
    min_train = max(int(min_train), 1)
    if len(valid_idx) < min_train:
        return
    start = int(valid_idx[min_train - 1]) + 1
    step = max(int(retrain_every), 1)
    while start < n:
        stop = min(start + step, n)
        k = int(np.searchsorted(valid_idx, start))
        yield valid_idx[max(0, k - train_bars):k], start, stop
        start = stop


@contextmanager
def _quiet_hmmlearn() -> Iterator[None]:
    """Hide hmmlearn's convergence / degenerate-fit warnings; errors still pass."""
    logger = logging.getLogger("hmmlearn")
    level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(level)


@dataclass
class FittedHMM:
    model: GaussianHMM
    mean: np.ndarray  # training-window feature mean (standardisation)
    std: np.ndarray
    labels: list[str]  # semantic label of each state index
    state_ret: np.ndarray  # per-state mean smoothed log return (original units)
    state_vol: np.ndarray  # per-state volatility exp(mean log vol) (original units)

    @property
    def n_states(self) -> int:
        return len(self.labels)

    def standardise(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


def label_states(state_ret: np.ndarray, state_vol: np.ndarray, cfg: HMMConfig) -> list[str]:
    """Semantic state labels, stable across refits although hmmlearn permutes state indices.

    The highest-volatility state is HIGH_VOL when its volatility is at least
    `high_vol_ratio` times the calmest state's. Every other state is TREND_UP when
    its mean smoothed return >= `trend_threshold` * its volatility, TREND_DOWN when
    <= -`trend_threshold` * its volatility, otherwise RANGE.
    """
    labels: list[str] = []
    for ret, vol in zip(state_ret, state_vol):
        if ret >= cfg.trend_threshold * vol:
            labels.append(TREND_UP)
        elif ret <= -cfg.trend_threshold * vol:
            labels.append(TREND_DOWN)
        else:
            labels.append(RANGE)
    hi, lo = int(np.argmax(state_vol)), int(np.argmin(state_vol))
    if len(labels) > 1 and state_vol[hi] >= cfg.high_vol_ratio * state_vol[lo]:
        labels[hi] = HIGH_VOL
    return labels


def fit_hmm(features_train: pd.DataFrame | np.ndarray, cfg: HMMConfig) -> FittedHMM | None:
    """Fit a Gaussian HMM on the valid training rows (standardised with their own mean/std)."""
    x = np.asarray(features_train, dtype=float)
    x = x[np.isfinite(x).all(axis=1)]
    if len(x) <= cfg.n_states:
        return None
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    model = GaussianHMM(
        n_components=cfg.n_states,
        covariance_type=cfg.covariance_type,
        n_iter=cfg.n_iter,
        random_state=cfg.random_state,
    )
    try:
        with _quiet_hmmlearn():
            model.fit((x - mean) / std)
    except (ValueError, np.linalg.LinAlgError) as exc:
        log.warning("HMM eğitimi başarısız: %s", exc)
        return None
    if not (np.isfinite(model.means_).all() and np.isfinite(model.covars_).all()):
        return None
    orig = model.means_ * std + mean
    state_ret, state_vol = orig[:, 0], np.exp(orig[:, 1])
    return FittedHMM(model, mean, std, label_states(state_ret, state_vol, cfg), state_ret, state_vol)


def _full_covars(model: GaussianHMM) -> np.ndarray:
    """(n_states, d, d) covariances for every covariance_type (hmmlearn's covars_ mis-shapes "spherical")."""
    k, d = model.n_components, model.means_.shape[1]
    cv = np.asarray(getattr(model, "_covars_", model.covars_), dtype=float)
    if model.covariance_type == "full":
        return cv
    if model.covariance_type == "tied":
        return np.tile(cv, (k, 1, 1))
    cv = cv.reshape(k, -1)  # diag (k, d) or spherical (k,) / (k, d)
    if cv.shape[1] == 1:
        cv = np.repeat(cv, d, axis=1)
    return np.stack([np.diag(c) for c in cv])


def _emission_logpdf(fitted: FittedHMM, z: np.ndarray) -> np.ndarray:
    """Gaussian log-density of every (standardised) row under every state; NaN rows stay NaN."""
    means, covars = fitted.model.means_, _full_covars(fitted.model)
    n, d = z.shape
    out = np.empty((n, fitted.n_states))
    for k in range(fitted.n_states):
        chol = np.linalg.cholesky(covars[k])
        sol = np.linalg.solve(chol, (z - means[k]).T)
        maha = np.einsum("ij,ij->j", sol, sol)
        out[:, k] = -0.5 * (d * np.log(2.0 * np.pi) + maha) - np.log(np.diag(chol)).sum()
    return out


def forward_filter(fitted: FittedHMM, features: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Filtered posteriors P(state_t | x_1..x_t), shape (n, n_states), via the log-space forward pass.

    Rows with missing features get the prediction step only (alpha_t = alpha_{t-1} A).
    """
    x = np.asarray(features, dtype=float)
    n, k = len(x), fitted.n_states
    out = np.empty((n, k))
    if n == 0:
        return out
    log_b = _emission_logpdf(fitted, fitted.standardise(x))
    has_obs = np.isfinite(log_b).all(axis=1)
    log_a = np.log(np.maximum(fitted.model.transmat_, _LOG_FLOOR))
    la = np.log(np.maximum(fitted.model.startprob_, _LOG_FLOOR))
    for t in range(n):
        if t:
            x = la[:, None] + log_a  # log P(s_{t-1}=i | x_1..x_{t-1}) + log A_ij
            m = x.max(axis=0)
            la = m + np.log(np.exp(x - m).sum(axis=0))
        if has_obs[t]:
            la = la + log_b[t]
        m = la.max()
        la = la - (m + np.log(np.exp(la - m).sum()))
        out[t] = la
    return np.exp(out)


def _state_columns(post: np.ndarray, labels: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = post.argmax(axis=1)
    return state.astype(float), np.asarray(labels, dtype=object)[state], post[np.arange(len(post)), state]


def walk_forward_hmm(df: pd.DataFrame, cfg: HMMConfig) -> pd.DataFrame:
    """hmm_state, hmm_label, hmm_prob for every bar from walk-forward fits and forward filtering.

    Each chunk of `retrain_every` bars is scored by a model fitted on the last
    `train_bars` valid feature rows before the chunk; the filter runs from that
    training window's first row through the chunk, so row t only uses bars <= t.
    """
    feats = hmm_features(df, cfg.feature_window).to_numpy()
    n = len(df)
    state = np.full(n, np.nan)
    label = np.full(n, UNKNOWN, dtype=object)
    prob = np.full(n, np.nan)
    fitted: FittedHMM | None = None
    origin = 0
    valid = np.isfinite(feats).all(axis=1)
    for rows, start, stop in refit_schedule(valid, cfg.min_train, cfg.train_bars, cfg.retrain_every):
        new = fit_hmm(feats[rows], cfg) if len(rows) >= cfg.min_train else None
        if new is not None:
            fitted, origin = new, int(rows[0])
        if fitted is None:
            continue
        post = forward_filter(fitted, feats[origin:stop])[start - origin:]
        state[start:stop], label[start:stop], prob[start:stop] = _state_columns(post, fitted.labels)
    return pd.DataFrame({"hmm_state": state, "hmm_label": label, "hmm_prob": prob}, index=df.index)


class OnlineHMM:
    """Live counterpart of walk_forward_hmm: refit every `retrain_every` closed bars."""

    def __init__(self, cfg: HMMConfig):
        self.cfg = cfg
        self.fitted: FittedHMM | None = None
        self.train_start: Any = None  # index label of the first training row
        self.bars_since_fit = 0

    def needs_fit(self) -> bool:
        return self.fitted is None or self.bars_since_fit >= self.cfg.retrain_every

    def fit(self, df: pd.DataFrame) -> bool:
        feats = hmm_features(df, self.cfg.feature_window)
        rows = np.flatnonzero(feats.notna().all(axis=1).to_numpy())[-self.cfg.train_bars:]
        if len(rows) < self.cfg.min_train:
            return False
        fitted = fit_hmm(feats.to_numpy()[rows], self.cfg)
        if fitted is None:
            return False
        self.fitted = fitted
        self.train_start = df.index[rows[0]]
        self.bars_since_fit = 0
        return True

    def infer_last(self, df: pd.DataFrame) -> tuple[str, float, float]:
        """(label, probability, state) of the last bar, filtered from the training window start."""
        if self.fitted is None or df.empty:
            return UNKNOWN, float("nan"), float("nan")
        pos = int(df.index.get_indexer([self.train_start])[0]) if self.train_start is not None else -1
        if pos < 0:
            pos = max(0, len(df) - self.cfg.train_bars)
        feats = hmm_features(df, self.cfg.feature_window).to_numpy()[pos:]
        post = forward_filter(self.fitted, feats)[-1:]
        state, label, prob = _state_columns(post, self.fitted.labels)
        return str(label[0]), float(prob[0]), float(state[0])
