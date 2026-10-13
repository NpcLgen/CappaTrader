"""GARCH(p, q) next-bar volatility forecast, refitted walk-forward on past bars only.

The model is fitted with `arch` on percent log returns (x100) with a constant
mean and normal errors; forecasts are then rolled forward with the explicit
GARCH recursion so the value at bar t uses returns <= t only:

    sigma2_{t+1} = omega + sum_i alpha_i * eps_{t+1-i}^2 + sum_j beta_j * sigma2_{t+1-j},  eps = r - mu
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from arch import arch_model

from cappatrader.config import GARCHConfig
from cappatrader.regime.hmm import chunk_start, refit_buckets, refit_schedule

log = logging.getLogger(__name__)

PCT = 100.0
# Near-integrated fits have an ill-defined long-run variance; use the sample variance instead.
MAX_PERSISTENCE = 0.999
# omega / (1 - persistence) divides by a small, noisy number: with omega at its lower bound it can
# land orders of magnitude away from the variance the window actually had (garch_ratio of ~250 for
# a whole chunk). Outside this factor of the sample variance it is not a credible normal level.
MAX_LONGRUN_FACTOR = 4.0
COLUMNS = ["garch_vol", "garch_longrun", "garch_ratio", "garch_size_mult"]


@dataclass
class GARCHParams:
    mu: float
    omega: float
    alphas: np.ndarray
    betas: np.ndarray
    backcast: float  # pre-sample variance of the training window (arch's convention)
    uncond_var: float  # long-run variance, percent^2

    @property
    def persistence(self) -> float:
        return float(self.alphas.sum() + self.betas.sum())


def pct_log_returns(close: pd.Series) -> np.ndarray:
    return PCT * np.log(close.astype(float)).diff().to_numpy()


def arch_backcast(resids: np.ndarray) -> float:
    """arch's pre-sample variance: exponentially weighted mean of the first 75 squared residuals."""
    tau = min(75, len(resids))
    w = 0.94 ** np.arange(tau)
    w = w / w.sum()
    return float(np.sum(resids[:tau] ** 2 * w))


def unconditional_variance(omega: float, alphas: np.ndarray, betas: np.ndarray, sample_var: float) -> float:
    """omega / (1 - sum(alpha) - sum(beta)); the sample variance for a (near-)integrated fit or when
    that level is more than MAX_LONGRUN_FACTOR away from the sample variance."""
    persistence = float(np.sum(alphas) + np.sum(betas))
    if persistence < MAX_PERSISTENCE:
        uv = omega / (1.0 - persistence)
        if np.isfinite(uv) and sample_var / MAX_LONGRUN_FACTOR <= uv <= sample_var * MAX_LONGRUN_FACTOR:
            return float(uv)
    return float(sample_var)


def fit_garch(returns_pct: np.ndarray, cfg: GARCHConfig) -> GARCHParams | None:
    """Fit GARCH(p, q) on the finite percent returns; None when the optimiser fails."""
    y = np.asarray(returns_pct, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < max(cfg.p + cfg.q + 2, 20) or not y.var() > 0:
        return None
    backcast = arch_backcast(y - y.mean())
    model = arch_model(y, mean="Constant", vol="GARCH", p=cfg.p, q=cfg.q, dist="normal", rescale=False)
    try:
        res = model.fit(disp="off", show_warning=False, backcast=backcast)
    except (ValueError, np.linalg.LinAlgError) as exc:
        log.warning("GARCH eğitimi başarısız: %s", exc)
        return None
    if res.convergence_flag != 0:
        return None
    prm = res.params
    alphas = np.array([prm[f"alpha[{i}]"] for i in range(1, cfg.p + 1)], dtype=float)
    betas = np.array([prm[f"beta[{j}]"] for j in range(1, cfg.q + 1)], dtype=float)
    mu, omega = float(prm["mu"]), float(prm["omega"])
    coefs = np.concatenate([[mu, omega], alphas, betas])
    if not np.isfinite(coefs).all() or omega <= 0 or (alphas < 0).any() or (betas < 0).any():
        return None
    return GARCHParams(mu, omega, alphas, betas, backcast,
                       unconditional_variance(omega, alphas, betas, float(y.var())))


def garch_filter(params: GARCHParams, returns_pct: np.ndarray, init_var: float) -> np.ndarray:
    """Next-step conditional variances: out[t] = Var(r_{t+1} | r_0..r_t), percent^2.

    Pre-sample squared residuals and variances are `init_var`; a missing or
    non-finite return is replaced by its expectation (eps^2 = sigma2), so the
    forecast keeps rolling.
    """
    a = [float(x) for x in params.alphas]
    b = [float(x) for x in params.betas]
    omega, mu = float(params.omega), float(params.mu)
    eps2 = [float(init_var)] * len(a)  # newest first
    sig2 = [float(init_var)] * len(b)
    s = omega + sum(a) * init_var + sum(b) * init_var  # Var(r_0)
    r = np.asarray(returns_pct, dtype=float).tolist()
    out = np.empty(len(r))
    for t, x in enumerate(r):
        e = (x - mu) ** 2 if math.isfinite(x) else s
        if a:
            eps2 = [e, *eps2[:-1]]
        if b:
            sig2 = [s, *sig2[:-1]]
        s = omega + sum(ai * ei for ai, ei in zip(a, eps2)) + sum(bj * sj for bj, sj in zip(b, sig2))
        out[t] = s
    return out


def garch_frame(var_next: np.ndarray, uncond_var: np.ndarray | float, cfg: GARCHConfig,
                index: pd.Index) -> pd.DataFrame:
    """Percent^2 variances -> garch_vol / garch_longrun (fractions), ratio and size multiplier."""
    vol = np.sqrt(np.asarray(var_next, dtype=float)) / PCT
    longrun = np.sqrt(np.broadcast_to(np.asarray(uncond_var, dtype=float), vol.shape)) / PCT
    ratio = vol / longrun
    mult = np.clip(1.0 / ratio, cfg.min_size_mult, 1.0)
    return pd.DataFrame({"garch_vol": vol, "garch_longrun": longrun, "garch_ratio": ratio,
                         "garch_size_mult": mult}, index=index)


def walk_forward_garch(df: pd.DataFrame, cfg: GARCHConfig) -> pd.DataFrame:
    """garch_* columns for every bar; the value at t forecasts bar t+1 from returns <= t.

    Each chunk of `retrain_every` bars uses a fit on the last `train_bars` returns
    before the chunk; the recursion runs from that training window's first return
    (arch's backcast as the pre-sample variance) through the chunk.
    """
    r = pct_log_returns(df["close"])
    n = len(df)
    var = np.full(n, np.nan)
    longrun = np.full(n, np.nan)
    params: GARCHParams | None = None
    origin = 0
    buckets = refit_buckets(df.index, cfg.retrain_every)
    for rows, start, stop in refit_schedule(np.isfinite(r), cfg.min_train, cfg.train_bars, cfg.retrain_every, buckets):
        new = fit_garch(r[rows], cfg)
        if new is not None:
            params, origin = new, int(rows[0])
        if params is None:
            continue
        var[start:stop] = garch_filter(params, r[origin:stop], params.backcast)[start - origin:]
        longrun[start:stop] = params.uncond_var
    return garch_frame(var, longrun, cfg, df.index)


class OnlineGARCH:
    """Live counterpart of walk_forward_garch.

    To reproduce the backtest exactly: before scoring each new closed bar, call
    `fit_chunk(bars)` when `due(bars)`; it fits on the bars before the epoch-anchored chunk of
    the last bar, exactly the walk-forward training window (start-up and data outages included).
    """

    def __init__(self, cfg: GARCHConfig):
        self.cfg = cfg
        self.params: GARCHParams | None = None
        self.train_start: Any = None  # index label of the first training bar
        self.bars_since_fit = 0
        self.chunk: int | None = None  # anchored refit chunk the current fit was made for

    def needs_fit(self) -> bool:
        return self.params is None or self.bars_since_fit >= self.cfg.retrain_every

    def due(self, df: pd.DataFrame) -> bool:
        """Refit before forecasting from the last bar of df? Whenever that bar lies in an anchored chunk
        the model was not fitted for (same bars as walk_forward_garch, outages included)."""
        if self.params is None:
            return True
        b = refit_buckets(df.index, self.cfg.retrain_every)
        if b is None:
            return self.needs_fit()
        return self.chunk != int(b[-1])

    def fit_chunk(self, df: pd.DataFrame) -> bool:
        """Fit for the chunk of df's last bar on the bars before that chunk starts (see OnlineHMM.fit_chunk)."""
        b = refit_buckets(df.index, self.cfg.retrain_every)
        if b is None:
            return self.fit(df.iloc[:-1])
        ok = self.fit(df.iloc[:chunk_start(df.index, self.cfg.retrain_every)])
        if not ok and self.params is None:
            ok = self.fit(df)
        self.chunk = int(b[-1])
        return ok

    def fit(self, df: pd.DataFrame) -> bool:
        r = pct_log_returns(df["close"])
        rows = np.flatnonzero(np.isfinite(r))
        if len(rows) < self.cfg.min_train:
            return False
        rows = rows[-self.cfg.train_bars:]
        params = fit_garch(r[rows], self.cfg)
        if params is None:
            return False
        self.params = params
        self.train_start = df.index[rows[0]]
        self.bars_since_fit = 0
        return True

    def forecast_last(self, df: pd.DataFrame) -> dict[str, float]:
        """garch_* values of the last bar (forecast for the next bar)."""
        if self.params is None or df.empty:
            return {c: float("nan") for c in COLUMNS}
        r = pct_log_returns(df["close"])
        pos = int(df.index.get_indexer([self.train_start])[0]) if self.train_start is not None else -1
        init = self.params.backcast
        if pos < 0:  # training window scrolled out of the buffer: the recursion forgets its start
            pos, init = max(0, len(df) - self.cfg.train_bars), self.params.uncond_var
        var = garch_filter(self.params, r[pos:], init)[-1:]
        row = garch_frame(var, self.params.uncond_var, self.cfg, df.index[-1:])
        return {c: float(row[c].iloc[0]) for c in COLUMNS}
