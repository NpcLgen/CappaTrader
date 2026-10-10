"""Market regime detection: trend / range / transition plus a volatility state.

Trend strength comes from ADX (+DI/-DI for direction); volatility from the rolling
percentile of ATR% within the symbol's own history, so gold and altcoins are each
judged against their own normal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cappatrader.config import RegimeConfig
from cappatrader.features.indicators import rolling_pct_rank

TREND_UP = "TREND_UP"
TREND_DOWN = "TREND_DOWN"
RANGE = "RANGE"
TRANSITION = "TRANSITION"
HIGH_VOL = "HIGH_VOL"  # HMM state with clearly elevated volatility
UNKNOWN = "UNKNOWN"

VOL_NORMAL = "NORMAL"
VOL_HIGH = "HIGH"
VOL_EXTREME = "EXTREME"

REGIME_TR = {
    TREND_UP: "Trend ↑ (yükseliş)",
    TREND_DOWN: "Trend ↓ (düşüş)",
    RANGE: "Yatay bant (ortalamaya dönüş)",
    TRANSITION: "Geçiş (zayıf trend)",
    HIGH_VOL: "Yüksek volatilite (HMM)",
    UNKNOWN: "Belirsiz (yetersiz veri)",
}

VOL_TR = {
    VOL_NORMAL: "normal",
    VOL_HIGH: "yüksek (pozisyon küçültülür)",
    VOL_EXTREME: "aşırı (yeni işlem açılmaz)",
}


def regime_kind(regime: str) -> str:
    """Collapse TREND_UP/TREND_DOWN to TREND and HIGH_VOL to TRANSITION for weight lookup."""
    if regime in (TREND_UP, TREND_DOWN):
        return "TREND"
    return TRANSITION if regime == HIGH_VOL else regime


def classify(frame: pd.DataFrame, cfg: RegimeConfig) -> pd.DataFrame:
    """Return regime, vol_state, vol_pct and atr_pct columns for an indicator frame."""
    adx = frame["adx"].to_numpy()
    plus_di = frame["plus_di"].to_numpy()
    minus_di = frame["minus_di"].to_numpy()
    atr_pct = frame["atr"] / frame["close"]
    vol_pct = rolling_pct_rank(atr_pct, cfg.vol_lookback)

    regime = np.where(
        adx >= cfg.adx_trend,
        np.where(plus_di >= minus_di, TREND_UP, TREND_DOWN),
        np.where(adx < cfg.adx_range, RANGE, TRANSITION),
    ).astype(object)
    regime[np.isnan(adx) | vol_pct.isna().to_numpy()] = UNKNOWN

    vp = vol_pct.to_numpy()
    vol_state = np.where(
        vp >= cfg.extreme_vol_pct, VOL_EXTREME, np.where(vp >= cfg.high_vol_pct, VOL_HIGH, VOL_NORMAL)
    ).astype(object)

    return pd.DataFrame(
        {"regime": regime, "vol_state": vol_state, "vol_pct": vol_pct, "atr_pct": atr_pct},
        index=frame.index,
    )
