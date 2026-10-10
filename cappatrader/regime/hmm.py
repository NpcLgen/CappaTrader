"""Hidden Markov Model market regimes learned from returns, strictly causal.

Two features per bar: the smoothed log return and the log realised volatility
over `feature_window` bars. A Gaussian HMM is refitted walk-forward on past bars
only, and state probabilities come from the forward filter
P(state_t | x_1..x_t), never from Viterbi or forward-backward smoothing (both
look at bars after t).
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from numpy.lib.stride_tricks import sliding_window_view
from threadpoolctl import ThreadpoolController

try:  # hmmlearn's compiled forward pass; private, so keep a numpy fallback
    from hmmlearn._hmmc import forward_log as _forward_log_c
except ImportError:  # pragma: no cover
    _forward_log_c = None

from cappatrader.config import HMMConfig
from cappatrader.regime.detector import HIGH_VOL, RANGE, TREND_DOWN, TREND_UP, UNKNOWN

log = logging.getLogger(__name__)

_LOG_FLOOR = 1e-300
_MIN_COVAR = 1e-3  # hmmlearn's default floor added to the initial covariances
_COVARS_PRIOR = 1e-2  # hmmlearn's default covariance prior weight, applied as a ridge (see _start_model)
_threadpools: ThreadpoolController | None = None


def hmm_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Smoothed log return and log realised volatility of 1-bar log returns over `window` bars.

    Each window is reduced on its own (no running sums), so a bar's features are bit-identical
    whatever the history starts with: the live buffer and the backtest feed EM the same numbers
    (EM is chaotic enough that 1e-13 of rolling-sum drift can change the optimum it finds)."""
    r = np.log(df["close"].to_numpy(dtype=float))
    r = np.concatenate([[np.nan], np.diff(r)]) if len(r) else r
    mean = np.full(len(r), np.nan)
    vol = np.full(len(r), np.nan)
    if window >= 1 and len(r) >= window:
        win = sliding_window_view(r, window)
        mean[window - 1:] = win.mean(axis=1)
        if window > 1:
            vol[window - 1:] = win.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        logvol = np.log(np.where(vol > 0, vol, np.nan))
    out = pd.DataFrame({"hmm_ret": mean, "hmm_logvol": logvol}, index=df.index)
    return out.replace([np.inf, -np.inf], np.nan)


def refit_buckets(index: pd.Index, retrain_every: int) -> np.ndarray | None:
    """Epoch-anchored refit chunk id of every bar: timestamp // (bar length x retrain_every).

    Anchoring the refit schedule on time instead of on the first row makes it independent
    of where the data starts, so a live run refits on exactly the bars the backtest does.
    None for a non-datetime index (chunks then count rows from the first fit)."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return None
    ts = index.asi8  # integer timestamps in the index's own unit; the ratio below is unit-free
    d = np.diff(ts)
    d = d[d > 0]
    if not len(d):
        return None
    return ts // (int(np.median(d)) * max(int(retrain_every), 1))


def chunk_start(index: pd.Index, retrain_every: int) -> int:
    """Position of the first bar of the last bar's anchored refit chunk (len(index) when unanchored)."""
    b = refit_buckets(index, retrain_every)
    if b is None:
        return len(index)
    return int(np.searchsorted(b, b[-1], side="left"))


def starts_chunk(index: pd.Index, retrain_every: int) -> bool:
    """True when the last bar is the first bar of a new anchored refit chunk."""
    b = refit_buckets(index, retrain_every)
    return b is not None and b[-1] != b[-2]


def refit_schedule(
    valid: np.ndarray, min_train: int, train_bars: int, retrain_every: int, buckets: np.ndarray | None = None
) -> Iterator[tuple[np.ndarray, int, int]]:
    """Yield (train_rows, start, stop) for walk-forward refits.

    The first chunk starts right after the `min_train`-th valid row. Each chunk is scored
    by a model fitted on the last `train_bars` valid rows strictly before its start. With
    `buckets` (see refit_buckets) a chunk ends where the bucket changes, so refits happen on
    epoch-anchored bars; without, every chunk is `retrain_every` rows long.
    """
    valid_idx = np.flatnonzero(valid)
    n = len(valid)
    min_train = max(int(min_train), 1)
    if len(valid_idx) < min_train:
        return
    start = int(valid_idx[min_train - 1]) + 1
    step = max(int(retrain_every), 1)
    edges = np.flatnonzero(np.diff(buckets) != 0) + 1 if buckets is not None else None
    while start < n:
        if edges is None:
            stop = min(start + step, n)
        else:
            j = int(np.searchsorted(edges, start, side="right"))
            stop = int(edges[j]) if j < len(edges) else n
        k = int(np.searchsorted(valid_idx, start))
        yield valid_idx[max(0, k - train_bars):k], start, stop
        start = stop


@contextmanager
def _quiet_single_thread() -> Iterator[None]:
    """Hide hmmlearn's convergence / degenerate-fit log lines and its numpy deprecation noise
    (errors still pass), and run native code on one thread: the fits are tiny and
    oversubscribed OpenMP/BLAS pools slow them down ~100x on a busy machine."""
    global _threadpools
    if _threadpools is None:  # scanning the loaded native libraries costs ~5 ms, do it once
        _threadpools = ThreadpoolController()
    logger = logging.getLogger("hmmlearn")
    level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        with _threadpools.limit(limits=1), warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="hmmlearn")
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


def _gaussian_logpdf(z: np.ndarray, means: np.ndarray, covars: np.ndarray) -> np.ndarray:
    """log N(z_i; means_k, covars_k), shape (n, k); NaN rows stay NaN.

    Every state's Cholesky factor whitens all rows in one matmul: a third of the time
    of hmmlearn's per-state scipy solves, the biggest single cost of an EM iteration.
    """
    (n, d), k = z.shape, len(means)
    chol = np.linalg.cholesky(covars)
    inv = np.linalg.inv(chol)  # L_k^-1
    white = (z @ inv.transpose(2, 0, 1).reshape(d, k * d)).reshape(n, k, d) - np.einsum("kij,kj->ki", inv, means)
    logdet = np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1)
    return -0.5 * (d * np.log(2.0 * np.pi) + (white * white).sum(axis=2)) - logdet


class _GaussianHMM(GaussianHMM):
    """hmmlearn's GaussianHMM with the forward filter's emission densities (same values, faster)."""

    def _compute_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        return _gaussian_logpdf(X, self.means_, _full_covars(self))


def _kmeans_start(z: np.ndarray, k: int, n_iter: int = 30, max_rows: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic k-means (centres, row labels) used as the EM starting point.

    Lloyd iterations on an evenly strided subsample of at most `max_rows` rows start
    from k equal-count bins along every feature axis and every principal axis; the
    start with the lowest within-cluster sum of squares over all rows wins. A pure
    function of the training rows (no random restarts, no dependence on earlier
    fits) and ~5x cheaper than hmmlearn's 10 k-means++ restarts.
    """
    n, d = z.shape
    sub = z[:: -(-n // max_rows)]
    m = len(sub)
    _, axes = np.linalg.eigh(np.atleast_2d(np.cov(z, rowvar=False)))
    runs = []
    for axis in (*np.eye(d), *axes.T):
        lab = np.empty(m, dtype=np.intp)
        lab[np.argsort(sub @ axis, kind="stable")] = np.arange(m) * k // m
        centres = np.zeros((k, d))
        for _ in range(n_iter):
            counts = np.bincount(lab, minlength=k)[:, None]
            sums = np.stack([np.bincount(lab, weights=sub[:, j], minlength=k) for j in range(d)], axis=1)
            centres = np.where(counts > 0, sums / np.maximum(counts, 1), centres)  # empty cluster keeps its centre
            new = _nearest(sub, centres)
            if np.array_equal(new, lab):
                break
            lab = new
        full = _nearest(z, centres)
        runs.append((float(((z - centres[full]) ** 2).sum()), centres, full))
    _, centres, lab = min(runs, key=lambda run: run[0])  # ties: the first start wins
    return centres, lab


def _nearest(z: np.ndarray, centres: np.ndarray) -> np.ndarray:
    return ((centres * centres).sum(axis=1) - 2.0 * z @ centres.T).argmin(axis=1)


def _start_model(z: np.ndarray, cfg: HMMConfig, implementation: str) -> GaussianHMM:
    """GaussianHMM with every parameter preset from the data, so hmmlearn skips its own init.

    Means: deterministic k-means; transitions: smoothed k-means label transition
    counts; start: uniform; covariances: the pooled covariance (hmmlearn's choice).
    For full/tied covariances the prior is a ridge `_COVARS_PRIOR * I`: hmmlearn's
    scalar prior is added to every matrix entry, which leaves a state that collapses
    onto one outlier row with a singular covariance and the whole fit fails.
    """
    k, d = cfg.n_states, z.shape[1]
    centres, lab = _kmeans_start(z, k)
    trans = np.ones((k, k))
    np.add.at(trans, (lab[:-1], lab[1:]), 1.0)
    cov = np.atleast_2d(np.cov(z, rowvar=False)) + _MIN_COVAR * np.eye(d)
    ct = cfg.covariance_type
    model = _GaussianHMM(
        n_components=k,
        covariance_type=ct,
        min_covar=_MIN_COVAR,
        covars_prior=_COVARS_PRIOR * np.eye(d) if ct in ("full", "tied") else _COVARS_PRIOR,
        n_iter=cfg.n_iter,
        random_state=cfg.random_state,
        init_params="",
        implementation=implementation,
    )
    model.startprob_ = np.full(k, 1.0 / k)
    model.transmat_ = trans / trans.sum(axis=1, keepdims=True)
    model.means_ = centres
    model.covars_ = {"full": np.tile(cov, (k, 1, 1)), "tied": cov, "diag": np.tile(np.diag(cov), (k, 1)),
                     "spherical": np.full(k, np.diag(cov).mean())}[ct]
    return model


def fit_hmm(features_train: pd.DataFrame | np.ndarray, cfg: HMMConfig) -> FittedHMM | None:
    """Fit a Gaussian HMM on the valid training rows (standardised with their own mean/std).

    EM starts from a deterministic, data-driven initialisation (_start_model), so a
    fit depends only on its own training rows. It runs hmmlearn's scaled forward-
    backward (~3x faster than log space) and repeats in log space if that underflows.
    """
    x = np.asarray(features_train, dtype=float)
    x = x[np.isfinite(x).all(axis=1)]
    if len(x) <= cfg.n_states:
        return None
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    z = (x - mean) / std
    error: Exception | None = None
    for implementation in ("scaling", "log"):
        try:
            model = _start_model(z, cfg, implementation)
            with _quiet_single_thread():
                model.fit(z)
            np.linalg.cholesky(_full_covars(model))  # emission densities need positive-definite covariances
            break
        except (ValueError, np.linalg.LinAlgError) as exc:
            error = exc
    else:
        log.warning("HMM eğitimi başarısız: %s", error)
        return None
    params = (model.startprob_, model.transmat_, model.means_, model.covars_)
    if not all(np.isfinite(p).all() for p in params):
        return None
    orig = model.means_ * std + mean
    state_ret, state_vol = orig[:, 0], np.exp(orig[:, 1])
    return FittedHMM(model, mean, std, label_states(state_ret, state_vol, cfg), state_ret, state_vol)


def forward_filter(fitted: FittedHMM, features: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Filtered posteriors P(state_t | x_1..x_t), shape (n, n_states), via the log-space forward pass.

    Rows with missing features get the prediction step only (alpha_t = alpha_{t-1} A).
    """
    x = np.asarray(features, dtype=float)
    n, k = len(x), fitted.n_states
    if n == 0:
        return np.empty((0, k))
    log_b = _gaussian_logpdf(fitted.standardise(x), fitted.model.means_, _full_covars(fitted.model))
    log_b[~np.isfinite(log_b).all(axis=1)] = 0.0  # no emission update
    start = np.maximum(fitted.model.startprob_, _LOG_FLOOR)
    trans = np.maximum(fitted.model.transmat_, _LOG_FLOOR)
    if _forward_log_c is not None:
        la = _forward_log_c(start, trans, log_b)[1]
    else:
        la = _forward_log_py(np.log(start), np.log(trans), log_b)
    la = la - la.max(axis=1, keepdims=True)
    post = np.exp(la)
    return post / post.sum(axis=1, keepdims=True)


def _forward_log_py(log_start: np.ndarray, log_a: np.ndarray, log_b: np.ndarray) -> np.ndarray:
    """Unnormalised log forward variables log P(x_1..x_t, state_t) (numpy fallback of hmmlearn's)."""
    out = np.empty_like(log_b)
    la = log_start + log_b[0]
    out[0] = la
    for t in range(1, len(log_b)):
        joint = la[:, None] + log_a  # log P(x_1..x_{t-1}, s_{t-1}=i) + log A_ij
        m = joint.max(axis=0)
        la = m + np.log(np.exp(joint - m).sum(axis=0)) + log_b[t]
        out[t] = la
    return out


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
    buckets = refit_buckets(df.index, cfg.retrain_every)
    for rows, start, stop in refit_schedule(valid, cfg.min_train, cfg.train_bars, cfg.retrain_every, buckets):
        new = fit_hmm(feats[rows], cfg)
        if new is not None:
            fitted, origin = new, int(rows[0])
        if fitted is None:
            continue
        post = forward_filter(fitted, feats[origin:stop])[start - origin:]
        state[start:stop], label[start:stop], prob[start:stop] = _state_columns(post, fitted.labels)
    return pd.DataFrame({"hmm_state": state, "hmm_label": label, "hmm_prob": prob}, index=df.index)


class OnlineHMM:
    """Live counterpart of walk_forward_hmm.

    To reproduce the backtest exactly: before scoring each new closed bar, call
    `fit_chunk(bars)` when `due(bars)`; it fits on the bars before the epoch-anchored chunk of
    the last bar, exactly the walk-forward training window (start-up and data outages included).
    """

    def __init__(self, cfg: HMMConfig):
        self.cfg = cfg
        self.fitted: FittedHMM | None = None
        self.train_start: Any = None  # index label of the first training row
        self.bars_since_fit = 0
        self.chunk: int | None = None  # anchored refit chunk the current fit was made for

    def needs_fit(self) -> bool:
        return self.fitted is None or self.bars_since_fit >= self.cfg.retrain_every

    def due(self, df: pd.DataFrame) -> bool:
        """Refit before scoring the last bar of df? Whenever that bar lies in an epoch-anchored chunk
        the model was not fitted for, exactly where walk_forward_hmm refits, also when the chunk's
        first bar was missed (data outage). Falls back to needs_fit for a non-datetime index."""
        if self.fitted is None:
            return True
        b = refit_buckets(df.index, self.cfg.retrain_every)
        if b is None:
            return self.needs_fit()
        return self.chunk != int(b[-1])

    def fit_chunk(self, df: pd.DataFrame) -> bool:
        """Fit for the chunk of df's last bar on the bars before that chunk starts (the walk-forward
        training window), falling back to all of df when that leaves too little data."""
        b = refit_buckets(df.index, self.cfg.retrain_every)
        if b is None:
            return self.fit(df.iloc[:-1])
        ok = self.fit(df.iloc[:chunk_start(df.index, self.cfg.retrain_every)])
        if not ok and self.fitted is None:
            ok = self.fit(df)
        self.chunk = int(b[-1])  # like the walk-forward, a failed refit keeps the old model for this chunk
        return ok

    def fit(self, df: pd.DataFrame) -> bool:
        feats = hmm_features(df, self.cfg.feature_window)
        rows = np.flatnonzero(feats.notna().all(axis=1).to_numpy())
        if len(rows) < self.cfg.min_train:
            return False
        rows = rows[-self.cfg.train_bars:]
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
