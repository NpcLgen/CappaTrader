"""Technical signal generators. Each returns a score in [-1, 1] (+ bullish)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cappatrader.config import SignalConfig


def ema_trend_score(f: pd.DataFrame) -> pd.Series:
    """EMA fast/slow spread measured in ATRs (1 ATR apart ~ 0.76)."""
    return np.tanh((f["ema_fast"] - f["ema_slow"]) / f["atr"])


def macd_score(f: pd.DataFrame) -> pd.Series:
    """MACD line relative to the zero line, normalised by ATR."""
    return np.tanh(2.0 * f["macd"] / f["atr"])


def breakout_score(f: pd.DataFrame) -> pd.Series:
    """Position inside the previous Donchian channel; a new high/low saturates to +/-1."""
    width = (f["dc_high"] - f["dc_low"]).replace(0.0, np.nan)
    return (2.0 * (f["close"] - f["dc_low"]) / width - 1.0).clip(-1.0, 1.0)


def rsi_mr_score(f: pd.DataFrame, cfg: SignalConfig) -> pd.Series:
    """Mean reversion: oversold -> +1, overbought -> -1, linear around 50."""
    r = f["rsi"]
    below = (50.0 - r) / (50.0 - cfg.rsi_oversold)
    above = (50.0 - r) / (cfg.rsi_overbought - 50.0)
    return pd.Series(np.where(r < 50.0, below, above), index=f.index).where(r.notna()).clip(-1.0, 1.0)


def bb_mr_score(f: pd.DataFrame) -> pd.Series:
    """Bollinger %b: lower band -> +1, upper band -> -1."""
    return (1.0 - 2.0 * f["bb_pctb"]).clip(-1.0, 1.0)
