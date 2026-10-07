"""Regime-weighted ensemble of signal scores -> composite score and 0-100 confidence."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cappatrader.config import COMPONENTS
from cappatrader.regime.detector import regime_kind


def combine(scores: pd.DataFrame, regime: pd.Series, weights: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Weighted mean of available scores. NaN scores are excluded (no opinion);
    a zero score is a neutral vote and does dilute the composite."""
    n = len(scores)
    kinds = regime.map(regime_kind).to_numpy()
    W = np.zeros((n, len(COMPONENTS)))
    for kind, wmap in weights.items():
        rows = kinds == kind
        if rows.any():
            W[rows] = [float(wmap.get(c, 0.0)) for c in COMPONENTS]
    S = scores.reindex(columns=list(COMPONENTS)).to_numpy(dtype=float)
    avail = ~np.isnan(S)
    WS = np.where(avail, W * np.nan_to_num(S), 0.0)
    den = (W * avail).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        composite = np.where(den > 0, WS.sum(axis=1) / den, np.nan)
        contrib = np.where(den[:, None] > 0, WS / den[:, None], np.nan)
    out = pd.DataFrame(contrib, index=scores.index, columns=[f"contrib_{c}" for c in COMPONENTS])
    out["composite"] = composite
    out["conf_long"] = 50.0 * (1.0 + composite)
    out["conf_short"] = 50.0 * (1.0 - composite)
    return out
