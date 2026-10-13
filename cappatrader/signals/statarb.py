"""Statistical arbitrage (pairs trading) on log prices.

Rolling OLS of log(A) on log(B) gives a hedge ratio; the residual z-score is the
spread. A wide negative z means A is cheap relative to B (long A / short B).
On a long-only spot account the signal simply favours the cheap leg.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cappatrader.config import StatArbConfig

# Engle-Granger (two variables, constant) MacKinnon critical values.
EG_CRITICAL = {"1%": -3.90, "5%": -3.34, "10%": -3.04}


def rolling_spread(close_a: pd.Series, close_b: pd.Series, window: int) -> pd.DataFrame:
    """Causal rolling hedge ratio, residual z-score and return correlation."""
    la, lb = np.log(close_a), np.log(close_b)
    mean_a = la.rolling(window, min_periods=window).mean()
    mean_b = lb.rolling(window, min_periods=window).mean()
    var_a = la.rolling(window, min_periods=window).var(ddof=0)
    var_b = lb.rolling(window, min_periods=window).var(ddof=0)
    cov = la.rolling(window, min_periods=window).cov(lb, ddof=0)
    beta = cov / var_b.replace(0.0, np.nan)
    rho2 = (cov**2 / (var_a * var_b)).clip(upper=0.999999)
    resid = (la - mean_a) - beta * (lb - mean_b)
    resid_std = np.sqrt(var_a * (1.0 - rho2))
    z = resid / resid_std.replace(0.0, np.nan)
    ret_corr = la.diff().rolling(window, min_periods=window).corr(lb.diff())
    return pd.DataFrame({"beta": beta, "z": z, "ret_corr": ret_corr})


def _z_to_score(z: pd.Series, entry_z: float) -> pd.Series:
    # Neutral inside |z| < 1, saturating at entry_z. Positive score = long leg A.
    mag = ((z.abs() - 1.0) / max(entry_z - 1.0, 1e-9)).clip(0.0, 1.0)
    return -np.sign(z) * mag


def statarb_scores(
    closes: dict[str, pd.Series], pairs: list[list[str]], cfg: StatArbConfig
) -> tuple[dict[str, pd.Series], dict[str, pd.DataFrame]]:
    """Average pair scores per symbol. Uncorrelated pairs contribute NaN (no opinion)."""
    per_symbol: dict[str, list[pd.Series]] = {}
    spreads: dict[str, pd.DataFrame] = {}
    for a, b in pairs:
        if a not in closes or b not in closes:
            continue
        ca = closes[a]
        cb = closes[b].reindex(ca.index).ffill(limit=3)
        sp = rolling_spread(ca, cb, cfg.window)
        spreads[f"{a}/{b}"] = sp
        score_a = _z_to_score(sp["z"], cfg.entry_z).where(sp["ret_corr"] >= cfg.min_corr)
        score_b = (-score_a).reindex(closes[b].index)
        per_symbol.setdefault(a, []).append(score_a)
        per_symbol.setdefault(b, []).append(score_b)
    out = {}
    for sym, scores in per_symbol.items():
        out[sym] = pd.concat(scores, axis=1).mean(axis=1, skipna=True)
    return out, spreads


@dataclass
class CointegrationReport:
    pair: str
    beta: float
    adf_t: float
    half_life_bars: float
    corr: float

    @property
    def cointegrated(self) -> bool:
        return self.adf_t < EG_CRITICAL["5%"]


def engle_granger(close_a: pd.Series, close_b: pd.Series, name: str = "") -> CointegrationReport:
    """Static Engle-Granger test (diagnostic for choosing pairs, not used per bar)."""
    df = pd.concat([np.log(close_a), np.log(close_b)], axis=1).dropna()
    y, x = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ coef
    de = np.diff(e)
    lag_e = e[:-1]
    # ADF with one augmentation lag, no constant (residuals are mean zero).
    Z = np.column_stack([lag_e[1:], de[:-1]])
    target = de[1:]
    g, *_ = np.linalg.lstsq(Z, target, rcond=None)
    resid = target - Z @ g
    sigma2 = resid @ resid / max(len(target) - Z.shape[1], 1)
    cov = sigma2 * np.linalg.inv(Z.T @ Z)
    adf_t = float(g[0] / np.sqrt(cov[0, 0]))
    gamma = g[0]
    half_life = float(-np.log(2) / np.log1p(gamma)) if -1 < gamma < 0 else float("inf")
    corr = float(np.corrcoef(np.diff(y), np.diff(x))[0, 1])
    return CointegrationReport(name, float(coef[1]), adf_t, half_life, corr)
