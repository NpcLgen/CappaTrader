"""Regime layer: HMM or ADX regime plus a GARCH or ATR-percentile volatility state.

regime.method picks the regime label (HMM state label or ADX classification);
regime.vol_method picks the volatility state (GARCH forecast / long-run ratio or
ATR% percentile). An HMM HIGH_VOL regime always implies at least a HIGH
volatility state. ATR-based vol_pct / atr_pct are always kept for explanations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cappatrader.config import RegimeConfig
from cappatrader.regime.detector import HIGH_VOL, UNKNOWN, VOL_EXTREME, VOL_HIGH, VOL_NORMAL, classify
from cappatrader.regime.garch import COLUMNS as GARCH_COLUMNS
from cappatrader.regime.garch import walk_forward_garch
from cappatrader.regime.hmm import walk_forward_hmm

HMM_COLUMNS = ["hmm_state", "hmm_label", "hmm_prob"]
REGIME_COLUMNS = ["regime", "vol_state", "vol_pct", "atr_pct", *HMM_COLUMNS, *GARCH_COLUMNS]


def _passthrough(src: pd.DataFrame | None, columns: list[str], index: pd.Index) -> pd.DataFrame:
    if src is None:
        return pd.DataFrame(np.nan, index=index, columns=columns)
    return src.reindex(index=index, columns=columns)


def classify_regime(
    frame: pd.DataFrame,
    cfg: RegimeConfig,
    hmm: pd.DataFrame | None = None,
    garch: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """regime, vol_state, vol_pct, atr_pct and the hmm_* / garch_* columns (NaN when not computed)."""
    base = classify(frame, cfg)
    h = _passthrough(hmm, HMM_COLUMNS, frame.index)
    h["hmm_label"] = h["hmm_label"].astype(object).where(h["hmm_label"].notna(), UNKNOWN)
    g = _passthrough(garch, GARCH_COLUMNS, frame.index)

    if cfg.method == "hmm":
        regime = h["hmm_label"].to_numpy(dtype=object)
    else:
        regime = base["regime"].to_numpy(dtype=object)

    vol_state = base["vol_state"].to_numpy(dtype=object).copy()
    if cfg.vol_method == "garch":
        ratio = g["garch_ratio"].to_numpy(dtype=float)
        known = np.isfinite(ratio)
        r = np.where(known, ratio, 0.0)
        garch_state = np.where(r >= cfg.garch.extreme_ratio, VOL_EXTREME,
                               np.where(r >= cfg.garch.high_ratio, VOL_HIGH, VOL_NORMAL))
        vol_state[known] = garch_state[known]
    vol_state[(regime == HIGH_VOL) & (vol_state == VOL_NORMAL)] = VOL_HIGH

    out = pd.DataFrame({"regime": regime, "vol_state": vol_state, "vol_pct": base["vol_pct"],
                        "atr_pct": base["atr_pct"]}, index=frame.index)
    return pd.concat([out, h, g], axis=1)


def compute_regime_layer(df: pd.DataFrame, cfg: RegimeConfig) -> pd.DataFrame:
    """Run the walk-forward HMM / GARCH the config needs, then classify_regime.

    `df` must already hold close, atr, adx, plus_di and minus_di.
    """
    hmm = walk_forward_hmm(df, cfg.hmm) if cfg.method == "hmm" else None
    garch = walk_forward_garch(df, cfg.garch) if cfg.vol_method == "garch" else None
    return classify_regime(df, cfg, hmm, garch)
