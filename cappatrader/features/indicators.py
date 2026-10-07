"""Vectorised, causal technical indicators (value at t uses data <= t only)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = wilder(delta.clip(lower=0.0), n)
    loss = wilder((-delta).clip(lower=0.0), n)
    rs = gain / loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # No losses in the window -> RSI 100; flat window -> 50.
    out = out.where(loss != 0.0, np.where(gain > 0, 100.0, 50.0))
    return out.where(gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    return pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return wilder(true_range(high, low, close), n)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.DataFrame:
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    tr = wilder(true_range(high, low, close), n)
    plus_di = 100.0 * wilder(plus_dm, n) / tr
    minus_di = 100.0 * wilder(minus_dm, n) / tr
    denom = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denom
    return pd.DataFrame({"adx": wilder(dx, n), "plus_di": plus_di, "minus_di": minus_di})


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, n)
    std = close.rolling(n, min_periods=n).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    width = upper - lower
    pctb = (close - lower) / width.replace(0.0, np.nan)
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower,
                         "bb_pctb": pctb, "bb_width": width / mid})


def donchian_prev(high: pd.Series, low: pd.Series, n: int) -> pd.DataFrame:
    """Channel of the previous n bars (excludes the current bar) for breakout tests."""
    return pd.DataFrame({
        "dc_high": high.rolling(n, min_periods=n).max().shift(1),
        "dc_low": low.rolling(n, min_periods=n).min().shift(1),
    })


def zscore(s: pd.Series, n: int) -> pd.Series:
    mean = s.rolling(n, min_periods=n).mean()
    std = s.rolling(n, min_periods=n).std(ddof=0)
    return (s - mean) / std.replace(0.0, np.nan)


def rolling_pct_rank(s: pd.Series, n: int, min_periods: int | None = None) -> pd.Series:
    return s.rolling(n, min_periods=min_periods or max(20, n // 5)).rank(pct=True)


def ewma_vol(close: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics EWMA volatility of log returns: a light GARCH(1,1) stand-in."""
    r = np.log(close).diff()
    return np.sqrt((r**2).ewm(alpha=1 - lam, adjust=False).mean())


def bars_since_sign_change(s: pd.Series) -> pd.Series:
    sign = np.sign(s).fillna(0.0)
    group = (sign != sign.shift()).cumsum()
    return sign.groupby(group).cumcount()
