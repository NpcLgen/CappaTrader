"""Squeeze Momentum (port of the user's "SQZ_Breakout_Pro_NEW" Pine Script v5) and anchored VWAP.

Every column is causal: the value at bar t uses bars <= t only. Pine semantics are kept:
ta.stdev is the population standard deviation, ta.highest/lowest/sma include the current
bar and need a full window, ta.tr is na on the first bar, nz() maps na to 0 and any
comparison with na is false.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from cappatrader.config import Config, SqueezeConfig

SQUEEZE_COLUMNS = (
    "sqz_val", "sqz_mom_up", "sqz_mom_down", "sqz_upper", "sqz_lower",
    "sqz_bull_pivot", "sqz_bear_pivot", "sqz_on", "sqz_released",
    "sqz_up_trend", "sqz_down_trend", "sqz_up3", "sqz_down3",
)


def _sq(cfg: SqueezeConfig | Config) -> SqueezeConfig:
    return cfg.strategy.squeeze if isinstance(cfg, Config) else cfg


def _rolling(a: np.ndarray, n: int, fn) -> np.ndarray:
    """fn applied to every full window of n bars ending at t (NaN before, or if the window holds NaN)."""
    out = np.full(len(a), np.nan)
    if n >= 1 and len(a) >= n:
        out[n - 1:] = fn(sliding_window_view(a, n))
    return out


def _sma(a: np.ndarray, n: int) -> np.ndarray:
    return _rolling(a, n, lambda w: w.mean(axis=1))


def _stdev(a: np.ndarray, n: int) -> np.ndarray:
    return _rolling(a, n, lambda w: w.std(axis=1))


def _highest(a: np.ndarray, n: int) -> np.ndarray:
    return _rolling(a, n, lambda w: w.max(axis=1))


def _lowest(a: np.ndarray, n: int) -> np.ndarray:
    return _rolling(a, n, lambda w: w.min(axis=1))


def _linreg(a: np.ndarray, n: int, offset: int = 0) -> np.ndarray:
    x = np.arange(n, dtype=float)
    xc = x - x.mean()
    denom = float(xc @ xc)

    def fit(w: np.ndarray) -> np.ndarray:
        slope = w @ xc / denom if denom > 0 else np.zeros(len(w))
        return w.mean(axis=1) + slope * (n - 1 - offset - x.mean())

    return _rolling(a, n, fit)


def _lag(a: np.ndarray, k: int, fill) -> np.ndarray:
    out = np.empty_like(a)
    out[:k] = fill
    out[k:] = a[:max(len(a) - k, 0)]
    return out


def _run_length(flag: np.ndarray) -> np.ndarray:
    """Number of consecutive True bars ending at each bar (0 where False)."""
    idx = np.arange(len(flag))
    return idx - np.maximum.accumulate(np.where(flag, -1, idx))


def pine_linreg(series: pd.Series | np.ndarray, length: int, offset: int = 0) -> pd.Series:
    """Pine ta.linreg: least-squares line over the last `length` bars (x = 0..length-1, oldest first),
    evaluated at x = length - 1 - offset."""
    s = series if isinstance(series, pd.Series) else pd.Series(series)
    return pd.Series(_linreg(s.to_numpy(dtype=float), length, offset), index=s.index, name=s.name)


def squeeze_momentum(df: pd.DataFrame, cfg: SqueezeConfig | Config) -> pd.DataFrame:
    """The sqz_* columns of the Pine script (source = close), plus sqz_mom_streak:
    +k / -k after k consecutive rising / falling momentum bars, 0 when unchanged."""
    sq = _sq(cfg)
    n = int(sq.length)
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))

    mid = ((_highest(h, n) + _lowest(l, n)) / 2.0 + _sma(c, n)) / 2.0
    val = _linreg(c - mid, n)
    prev = _lag(val, 1, np.nan)
    nz_prev = np.nan_to_num(prev, nan=0.0)
    nz_prev2 = np.nan_to_num(_lag(val, 2, np.nan), nan=0.0)
    mom_up = val > nz_prev
    mom_down = val < nz_prev

    if sq.calc_mode == "fixed":
        upper = np.full(len(val), float(sq.fixed_val))
    elif sq.calc_mode == "dynamic":
        upper = _stdev(val, n) * sq.dyn_mult
    elif sq.calc_mode == "percent":
        upper = _highest(np.abs(val), 2 * n) * (sq.perc_val / 100.0)
    else:
        raise ValueError(f"Bilinmeyen squeeze calc_mode: {sq.calc_mode}")
    lower = -upper

    # ta.crossover(val, nz(val[1])) = val > nz(val[1]) and val[1] <= nz(val[2]); crossunder mirrored.
    cross_up = mom_up & (prev <= nz_prev2)
    cross_down = mom_down & (prev >= nz_prev2)
    bull_pivot = (val < 0) & (val < lower) & cross_up
    bear_pivot = (val > 0) & (val > upper) & cross_down

    basis = _sma(c, n)
    dev = sq.bb_mult * _stdev(c, n)
    upper_bb, lower_bb = basis + dev, basis - dev
    prev_close = _lag(c, 1, np.nan)
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_close), np.abs(l - prev_close)))
    range_ma = _sma(tr, n)
    upper_kc, lower_kc = basis + range_ma * sq.kc_mult, basis - range_ma * sq.kc_mult  # Pine ma == basis
    sqz_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)

    released = ~_lag(sqz_on, 1, False) & _lag(sqz_on, 2, False)
    green_prev = _lag(c > o, 1, False)
    red_prev = _lag(c < o, 1, False)

    def three(m: np.ndarray) -> np.ndarray:
        return m & _lag(m, 1, False) & _lag(m, 2, False) & ~_lag(m, 3, False)

    return pd.DataFrame({
        "sqz_val": val,
        "sqz_mom_up": mom_up,
        "sqz_mom_down": mom_down,
        "sqz_upper": upper,
        "sqz_lower": lower,
        "sqz_bull_pivot": bull_pivot,
        "sqz_bear_pivot": bear_pivot,
        "sqz_on": sqz_on,
        "sqz_released": released,
        "sqz_up_trend": released & green_prev,
        "sqz_down_trend": released & red_prev,
        "sqz_up3": three(mom_up),
        "sqz_down3": three(mom_down),
        "sqz_mom_streak": _run_length(mom_up) - _run_length(mom_down),
    }, index=df.index)


def _utc(index: pd.Index) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("VWAP gün/hafta çapası için DatetimeIndex gerekli")
    return index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")


def vwap(df: pd.DataFrame, anchor: str = "D", rolling_bars: int = 24) -> pd.Series:
    """Volume-weighted average of hlc3, reset at each UTC day ("D") or week starting Monday ("W"),
    or over the last `rolling_bars` bars ("rolling").

    Bars with zero volume carry no weight; while the period (or window) has no volume at all the
    equal-weighted mean of hlc3 is used. The first anchor period is NaN unless the data starts at
    its boundary (a partial period would depend on where the history happens to begin), and a
    rolling window needs `rolling_bars` bars.
    """
    tp = ((df["high"] + df["low"] + df["close"]) / 3.0).to_numpy(dtype=float)
    vol = np.clip(np.nan_to_num(df["volume"].to_numpy(dtype=float), nan=0.0), 0.0, None)
    pv = tp * vol
    if anchor == "rolling":
        n = int(rolling_bars)
        num = _rolling(pv, n, lambda w: w.sum(axis=1))
        den = _rolling(vol, n, lambda w: w.sum(axis=1))
        flat = _sma(tp, n)
    elif anchor in ("D", "W"):
        idx = _utc(df.index)
        start = idx.floor("D")
        if anchor == "W":
            start = start - pd.to_timedelta(start.dayofweek, unit="D")
        key = start.asi8
        frame = pd.DataFrame({"pv": pv, "v": vol, "tp": tp})
        g = frame.groupby(key, sort=False)
        csum = g.cumsum()
        num, den = csum["pv"].to_numpy(), csum["v"].to_numpy()
        flat = csum["tp"].to_numpy() / (g.cumcount().to_numpy() + 1.0)
        if len(idx) and idx[0] != start[0]:
            num = np.where(key == key[0], np.nan, num)
    else:
        raise ValueError(f"Bilinmeyen VWAP çapası: {anchor}")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den > 0, num / den, flat)
    out = np.where(np.isnan(num), np.nan, out)
    return pd.Series(out, index=df.index, name="vwap")


def squeeze_layer(df: pd.DataFrame, cfg: SqueezeConfig | Config) -> pd.DataFrame:
    """squeeze_momentum columns plus "vwap": the squeeze block of the prepared frame."""
    sq = _sq(cfg)
    out = squeeze_momentum(df, sq)
    out["vwap"] = vwap(df, sq.vwap_anchor, sq.vwap_rolling_bars)
    return out
