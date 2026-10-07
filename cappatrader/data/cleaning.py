"""Data cleansing (PRD F-1.3): missing candles, inconsistent OHLC, absurd spikes."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cappatrader.config import TIMEFRAME_MS

KLINE_COLUMNS = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume"]


def klines_to_frame(rows: list[list]) -> pd.DataFrame:
    """Binance kline arrays -> DataFrame indexed by UTC open time."""
    if not rows:
        return pd.DataFrame(columns=KLINE_COLUMNS, index=pd.DatetimeIndex([], tz="UTC", name="open_time"))
    arr = pd.DataFrame(rows)
    df = pd.DataFrame({
        "open": arr[1].astype(float), "high": arr[2].astype(float), "low": arr[3].astype(float),
        "close": arr[4].astype(float), "volume": arr[5].astype(float), "quote_volume": arr[7].astype(float),
        "taker_buy_volume": arr[9].astype(float),
    })
    df.index = pd.to_datetime(arr[0].astype("int64"), unit="ms", utc=True)
    df.index.name = "open_time"
    return df


@dataclass
class CleaningReport:
    duplicates: int = 0
    invalid_rows: int = 0
    ohlc_fixed: int = 0
    spikes_fixed: int = 0
    gaps_filled: int = 0

    def __str__(self) -> str:
        return (f"tekrar {self.duplicates}, geçersiz {self.invalid_rows}, OHLC düzeltme {self.ohlc_fixed}, "
                f"iğne düzeltme {self.spikes_fixed}, eksik mum doldurma {self.gaps_filled}")


def clean_klines(df: pd.DataFrame, timeframe: str, mad_k: float = 8.0) -> tuple[pd.DataFrame, CleaningReport]:
    """Offline cleaning for historical candles.

    A spike is a bar whose close jumps > mad_k robust deviations and immediately
    reverts on the next bar; its close/wick are replaced by the neighbours' level.
    """
    rep = CleaningReport()
    df = df.sort_index()
    dup = df.index.duplicated(keep="last")
    rep.duplicates = int(dup.sum())
    df = df[~dup].copy()

    px = df[["open", "high", "low", "close"]]
    bad = (px <= 0).any(axis=1) | px.isna().any(axis=1) | (df["volume"] < 0)
    rep.invalid_rows = int(bad.sum())
    df = df[~bad].copy()

    hi = df[["open", "high", "low", "close"]].max(axis=1)
    lo = df[["open", "high", "low", "close"]].min(axis=1)
    fix = (df["high"] < hi) | (df["low"] > lo)
    rep.ohlc_fixed = int(fix.sum())
    df["high"], df["low"] = hi, lo

    if len(df) > 60:
        r = np.log(df["close"]).diff()
        med = r.rolling(50, min_periods=20).median()
        mad = (r - med).abs().rolling(50, min_periods=20).median() * 1.4826
        jump = (r - med).abs() > mad_k * mad.replace(0.0, np.nan)
        revert = (np.sign(r) != np.sign(r.shift(-1))) & ((r + r.shift(-1)).abs() < 0.5 * r.abs())
        spikes = (jump & revert).fillna(False).to_numpy()
        idx = np.where(spikes)[0]
        idx = idx[(idx > 0) & (idx < len(df) - 1)]
        if len(idx):
            close = df["close"].to_numpy().copy()
            high = df["high"].to_numpy().copy()
            low = df["low"].to_numpy().copy()
            opn = df["open"].to_numpy()
            for i in idx:
                close[i] = 0.5 * (close[i - 1] + close[i + 1])
                high[i] = max(opn[i], close[i], min(high[i], max(high[i - 1], high[i + 1])))
                low[i] = min(opn[i], close[i], max(low[i], min(low[i - 1], low[i + 1])))
            df["close"], df["high"], df["low"] = close, high, low
            rep.spikes_fixed = len(idx)

    if len(df) > 1:
        full = pd.date_range(df.index[0], df.index[-1], freq=pd.Timedelta(milliseconds=TIMEFRAME_MS[timeframe]),
                             tz="UTC", name=df.index.name)
        missing = full.difference(df.index)
        rep.gaps_filled = len(missing)
        if len(missing):
            df = df.reindex(full)
            prev_close = df["close"].ffill()
            for col in ("open", "high", "low", "close"):
                df[col] = df[col].fillna(prev_close)
            for col in ("volume", "quote_volume", "taker_buy_volume"):
                if col in df:
                    df[col] = df[col].fillna(0.0)
    return df, rep


@dataclass
class TickFilter:
    """Rejects absurd live prints (> max_dev from the recent median) unless the
    move is confirmed by `confirm` consecutive prints on the same side."""

    max_dev: float = 0.03
    window: int = 50
    confirm: int = 3
    history: deque = field(default_factory=deque)
    pending: list = field(default_factory=list)
    rejected: int = 0

    def __post_init__(self) -> None:
        self.history = deque(self.history, maxlen=self.window)

    def accept(self, price: float) -> bool:
        if not price > 0:
            self.rejected += 1
            return False
        if len(self.history) < 5:
            self.history.append(price)
            return True
        med = float(np.median(self.history))
        dev = price / med - 1.0
        if abs(dev) <= self.max_dev:
            self.pending.clear()
            self.history.append(price)
            return True
        if self.pending and np.sign(self.pending[-1] / med - 1.0) != np.sign(dev):
            self.pending.clear()
        self.pending.append(price)
        if len(self.pending) >= self.confirm:
            self.history.clear()
            self.history.extend(self.pending)
            self.pending.clear()
            return True
        self.rejected += 1
        return False
