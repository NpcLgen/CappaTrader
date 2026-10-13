"""Live paper broker: simulated market orders against the real order book.

Latency is simulated for real (we wait `latency_ms` before reading the book),
the order walks the live L2 levels, and any configured stress slippage is
added on top. Falls back to the cost model when the book is stale.
"""

from __future__ import annotations

import asyncio
import time

import pandas as pd

from cappatrader.data.orderbook import OrderBook
from cappatrader.execution.costs import CostModel
from cappatrader.models import Fill


class PaperBroker:
    def __init__(self, costs: CostModel, books: dict[str, OrderBook], last_price: dict[str, float]):
        self.costs = costs
        self.books = books
        self.last_price = last_price

    def reference_price(self, symbol: str) -> float:
        book = self.books.get(symbol)
        if book is not None and book.is_fresh():
            return book.mid
        return self.last_price[symbol]

    async def market_order(self, symbol: str, side: int, qty: float, ref_price: float, atr: float,
                           bar_quote_volume: float | None = None) -> Fill:
        t0 = time.perf_counter()
        latency_s = self.costs.cfg.latency_ms / 1000.0
        if latency_s > 0:
            await asyncio.sleep(latency_s)
        now = pd.Timestamp.now(tz="UTC")
        book = self.books.get(symbol)
        if book is not None and book.is_fresh():
            avg, _, _ = book.vwap_fill(side, qty)
            price = avg * (1 + side * self.costs.cfg.extra_slippage_bps / 1e4)
            fill = Fill(symbol=symbol, side=side, qty=qty, price=price, ref_price=ref_price,
                        fee=qty * price * self.costs.cfg.taker_fee,
                        slippage_bps=side * (price / ref_price - 1) * 1e4, latency_ms=0.0, timestamp=now)
        else:
            base = self.last_price.get(symbol, ref_price)
            fill = self.costs.fill(symbol, side, qty, base, atr, bar_quote_volume, now)
            fill.ref_price = ref_price
            fill.slippage_bps = side * (fill.price / ref_price - 1) * 1e4
        fill.latency_ms = (time.perf_counter() - t0) * 1000
        return fill
