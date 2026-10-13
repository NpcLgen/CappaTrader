"""Historical candles for research: download from Binance once, cache as CSV."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import pandas as pd

from cappatrader.config import Config
from cappatrader.data.binance_rest import BinanceREST
from cappatrader.data.cleaning import clean_klines

log = logging.getLogger(__name__)


def _cache_path(cfg: Config, symbol: str) -> Path:
    # research uses the market the bot trades: USD-M perpetual klines in live mode (prices, volume and
    # therefore VWAP differ from spot), so the two never share a cache file
    suffix = "_futures" if cfg.data_market == "futures" else ""
    return Path(cfg.data.data_dir) / f"{symbol}_{cfg.timeframe}{suffix}.csv"


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "open_time"
    return df


async def _download(cfg: Config, symbols: list[str], start_ms: int, end_ms: int) -> dict[str, pd.DataFrame]:
    rest = BinanceREST(cfg.exchange, cfg.secrets, market=cfg.data_market)
    try:
        out = {}
        for s in symbols:
            market = "USD-M futures" if cfg.data_market == "futures" else "spot"
            log.info("%s indiriliyor (%s, %s)...", s, cfg.timeframe, market)
            out[s] = await rest.klines_range(s, cfg.timeframe, start_ms, end_ms)
        return out
    finally:
        await rest.close()


def load_history(cfg: Config, days: int, symbols: list[str] | None = None, refresh: bool = False) -> dict[str, pd.DataFrame]:
    symbols = symbols or cfg.symbols
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86_400_000
    start_ts = pd.Timestamp(start_ms, unit="ms", tz="UTC")
    data, missing = {}, []
    for s in symbols:
        path = _cache_path(cfg, s)
        if path.exists() and not refresh:
            df = _read(path)
            fresh = len(df) and df.index[0] <= start_ts + pd.Timedelta(milliseconds=cfg.bar_ms) and \
                (pd.Timestamp.now(tz="UTC") - df.index[-1]) < pd.Timedelta(days=1)
            if fresh:
                data[s] = df[df.index >= start_ts]
                continue
        missing.append(s)
    if missing:
        downloaded = asyncio.run(_download(cfg, missing, start_ms, end_ms))
        Path(cfg.data.data_dir).mkdir(parents=True, exist_ok=True)
        for s, df in downloaded.items():
            df.to_csv(_cache_path(cfg, s))
            data[s] = df
    cleaned = {}
    for s in symbols:
        df, rep = clean_klines(data[s], cfg.timeframe, cfg.data.outlier_mad_k)
        log.info("%s: %d bar (%s)", s, len(df), rep)
        cleaned[s] = df
    return cleaned
