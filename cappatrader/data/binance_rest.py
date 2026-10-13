"""Async Binance REST client (spot or USD-M futures market data, funding, key permissions).

Market data endpoints are public. The API key is only used for the read-only
/sapi/v1/account/apiRestrictions check (PRD: withdrawals must be disabled).
Request weight is tracked from X-MBX-USED-WEIGHT-1M so we back off before the
exchange's limit instead of after a 429.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from urllib.parse import urlencode

import aiohttp
import pandas as pd

from cappatrader.config import TIMEFRAME_MS, ExchangeConfig, Secrets
from cappatrader.data.cleaning import klines_to_frame

log = logging.getLogger(__name__)


def _min_notional(flt: dict) -> float:
    # spot: NOTIONAL.minNotional / MIN_NOTIONAL.minNotional; futures: MIN_NOTIONAL.notional
    return float(flt.get("minNotional") or flt.get("notional") or 0)


class BinanceAPIError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"Binance HTTP {status}: {message}")
        self.status = status


class WeightLimiter:
    def __init__(self, max_per_minute: int, safety: float):
        self.limit = max_per_minute * safety
        self.used = 0
        self.minute = int(time.time() // 60)

    async def acquire(self, weight: int) -> None:
        now_min = int(time.time() // 60)
        if now_min != self.minute:
            self.minute, self.used = now_min, 0
        if self.used + weight > self.limit:
            wait = 60 - time.time() % 60 + 0.5
            log.warning("REST ağırlık sınırına yaklaşıldı (%s/%s), %.1f sn bekleniyor", self.used, self.limit, wait)
            await asyncio.sleep(wait)
            self.minute, self.used = int(time.time() // 60), 0
        self.used += weight

    def observe(self, header_value: str | None) -> None:
        if header_value and header_value.isdigit():
            self.used = max(self.used, int(header_value))


class BinanceREST:
    def __init__(self, cfg: ExchangeConfig, secrets: Secrets | None = None, session: aiohttp.ClientSession | None = None,
                 market: str = "spot"):
        if market not in ("spot", "futures"):
            raise ValueError("market: 'spot' veya 'futures'")
        self.cfg = cfg
        self.market = market
        fut = market == "futures"
        self._data_base = cfg.futures_rest_base if fut else cfg.rest_base
        self._paths = {
            "ping": "/fapi/v1/ping" if fut else "/api/v3/ping",
            "klines": "/fapi/v1/klines" if fut else "/api/v3/klines",
            "exchangeInfo": "/fapi/v1/exchangeInfo" if fut else "/api/v3/exchangeInfo",
            "ticker": "/fapi/v1/ticker/price" if fut else "/api/v3/ticker/price",
        }
        self.secrets = secrets or Secrets()
        self._session = session
        self._own_session = session is None
        self.limiter = WeightLimiter(cfg.max_weight_per_minute, cfg.weight_safety)

    async def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.cfg.request_timeout_s), trust_env=True)
            self._own_session = True
        return self._session

    async def close(self) -> None:
        if self._own_session and self._session is not None and not self._session.closed:
            await self._session.close()

    async def _get(self, base: str, path: str, params: dict | None = None, weight: int = 1,
                   signed: bool = False, retries: int = 3):
        params = dict(params or {})
        headers = {}
        for attempt in range(retries + 1):
            await self.limiter.acquire(weight)
            query_params = dict(params)
            if signed:
                if not self.secrets.has_binance:
                    raise BinanceAPIError(0, "API anahtarı tanımlı değil")
                query_params["timestamp"] = int(time.time() * 1000)
                query_params["recvWindow"] = self.cfg.recv_window_ms
                query = urlencode(query_params)
                sig = hmac.new(self.secrets.binance_secret_key.encode(), query.encode(), hashlib.sha256).hexdigest()
                query = f"{query}&signature={sig}"
                headers["X-MBX-APIKEY"] = self.secrets.binance_api_key
            else:
                query = urlencode(query_params)
            url = f"{base}{path}" + (f"?{query}" if query else "")
            try:
                s = await self.session()
                async with s.get(url, headers=headers) as resp:
                    self.limiter.observe(resp.headers.get("X-MBX-USED-WEIGHT-1M"))
                    if resp.status in (418, 429):
                        retry_after = float(resp.headers.get("Retry-After", "60"))
                        if attempt >= retries or retry_after > 300:  # 418 = IP ban, can last hours
                            raise BinanceAPIError(resp.status, f"rate limit / IP yasağı, Retry-After {retry_after:.0f} sn")
                        log.error("Binance rate limit (%s), %.0f sn bekleniyor", resp.status, retry_after)
                        await asyncio.sleep(retry_after)
                        continue
                    if resp.status >= 500 and attempt < retries:
                        await asyncio.sleep(2**attempt)
                        continue
                    if resp.status >= 400:
                        raise BinanceAPIError(resp.status, (await resp.text())[:300])
                    return await resp.json()
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt >= retries:
                    raise
                log.warning("Binance isteği başarısız (%s), tekrar deneniyor: %s", path, type(exc).__name__)
                await asyncio.sleep(2**attempt)
        raise BinanceAPIError(429, "rate limit nedeniyle istek tamamlanamadı")

    # -- spot market data ------------------------------------------------
    async def ping(self) -> bool:
        await self._get(self._data_base, self._paths["ping"])
        return True

    async def klines(self, symbol: str, interval: str, start_ms: int | None = None, end_ms: int | None = None,
                     limit: int = 1000) -> list[list]:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_ms is not None:
            params["startTime"] = int(start_ms)
        if end_ms is not None:
            params["endTime"] = int(end_ms)
        weight = 5 if self.market == "futures" else 2  # futures: weight 5 for limit 500-1000
        return await self._get(self._data_base, self._paths["klines"], params, weight=weight)

    async def recent_klines(self, symbol: str, interval: str, n_bars: int) -> pd.DataFrame:
        """The last n closed bars (the still-forming bar is dropped)."""
        step = TIMEFRAME_MS[interval]
        now_ms = int(time.time() * 1000)
        start = now_ms - (n_bars + 1) * step
        return await self.klines_range(symbol, interval, start, now_ms, drop_open_bar=True)

    async def klines_range(self, symbol: str, interval: str, start_ms: int, end_ms: int,
                           drop_open_bar: bool = True) -> pd.DataFrame:
        step = TIMEFRAME_MS[interval]
        rows: list[list] = []
        cursor = int(start_ms)
        while cursor < end_ms:
            batch = await self.klines(symbol, interval, start_ms=cursor, end_ms=end_ms, limit=1000)
            if not batch:
                break
            rows.extend(batch)
            nxt = int(batch[-1][0]) + step
            if nxt <= cursor or len(batch) < 1000:
                break
            cursor = nxt
        df = klines_to_frame(rows)
        df = df[~df.index.duplicated(keep="last")]
        if drop_open_bar and rows:
            now_ms = int(time.time() * 1000)
            close_times = {int(r[0]): int(r[6]) for r in rows}
            keep = [close_times.get(int(ts.value // 1_000_000), 0) < now_ms for ts in df.index]
            df = df[keep]
        return df

    async def exchange_filters(self, symbols: list[str]) -> dict[str, dict[str, float]]:
        if self.market == "futures":
            # The futures endpoint has no symbols filter: fetch all, keep perpetual contracts we trade.
            data = await self._get(self._data_base, self._paths["exchangeInfo"], weight=1)
            wanted = set(symbols)
            rows = [s for s in data.get("symbols", [])
                    if s.get("symbol") in wanted and s.get("contractType", "PERPETUAL") == "PERPETUAL"]
        else:
            data = await self._get(self._data_base, self._paths["exchangeInfo"],
                                   {"symbols": json.dumps(symbols, separators=(",", ":"))}, weight=20)
            rows = data.get("symbols", [])
        out = {}
        for s in rows:
            f = {x["filterType"]: x for x in s.get("filters", [])}
            out[s["symbol"]] = {
                "step_size": float(f.get("LOT_SIZE", {}).get("stepSize", 0) or 0),
                "min_qty": float(f.get("LOT_SIZE", {}).get("minQty", 0) or 0),
                "tick_size": float(f.get("PRICE_FILTER", {}).get("tickSize", 0) or 0),
                "min_notional": float(_min_notional(f.get("NOTIONAL") or f.get("MIN_NOTIONAL") or {})),
                "status": s.get("status", ""),
            }
        return out

    async def ticker_price(self, symbol: str) -> float:
        data = await self._get(self._data_base, self._paths["ticker"], {"symbol": symbol}, weight=2)
        return float(data["price"])

    # -- futures (alternative data) ----------------------------------------
    async def funding_rate(self, symbol: str) -> float | None:
        try:
            data = await self._get(self.cfg.futures_rest_base, "/fapi/v1/premiumIndex", {"symbol": symbol})
            return float(data["lastFundingRate"])
        except (BinanceAPIError, aiohttp.ClientError, asyncio.TimeoutError, KeyError, ValueError):
            return None

    # -- account (read-only) ------------------------------------------------
    async def api_restrictions(self) -> dict:
        return await self._get(self.cfg.rest_base, "/sapi/v1/account/apiRestrictions", weight=1, signed=True)
