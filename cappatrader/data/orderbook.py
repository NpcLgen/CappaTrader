"""Level-2 order book analytics: imbalance, liquidity walls, spoofing, trade flow.

Fed by Binance partial-depth snapshots (<symbol>@depth20@100ms) and aggTrades.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from cappatrader import fmt


@dataclass
class Wall:
    side: str  # "bid" | "ask"
    price: float
    qty: float


class OrderBook:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids: list[tuple[float, float]] = []
        self.asks: list[tuple[float, float]] = []
        self.updated = 0.0
        self.update_id = 0

    def update(self, bids, asks, update_id: int = 0, now: float | None = None) -> None:
        self.bids = [(float(p), float(q)) for p, q in bids if float(q) > 0]
        self.asks = [(float(p), float(q)) for p, q in asks if float(q) > 0]
        self.bids.sort(key=lambda x: -x[0])
        self.asks.sort(key=lambda x: x[0])
        self.update_id = update_id
        self.updated = time.monotonic() if now is None else now

    def is_fresh(self, max_age_s: float = 5.0, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        return bool(self.bids and self.asks) and now - self.updated <= max_age_s

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else float("nan")

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else float("nan")

    @property
    def mid(self) -> float:
        return 0.5 * (self.best_bid + self.best_ask)

    @property
    def spread_bps(self) -> float:
        return (self.best_ask - self.best_bid) / self.mid * 1e4 if self.bids and self.asks else float("nan")

    def imbalance(self, levels: int = 10) -> float:
        """(bid notional - ask notional) / total over the top levels, in [-1, 1]."""
        b = sum(p * q for p, q in self.bids[:levels])
        a = sum(p * q for p, q in self.asks[:levels])
        return (b - a) / (b + a) if b + a > 0 else 0.0

    def walls(self, k: float = 5.0) -> list[Wall]:
        out = []
        for side, levels in (("bid", self.bids), ("ask", self.asks)):
            if len(levels) < 5:
                continue
            med = float(np.median([q for _, q in levels]))
            out.extend(Wall(side, p, q) for p, q in levels if med > 0 and q >= k * med)
        return out

    def vwap_fill(self, side: int, qty: float) -> tuple[float, float, int]:
        """Average price for a market order walking the book. Depth beyond the
        snapshot is assumed 10 bps worse than the last visible level."""
        levels = self.asks if side > 0 else self.bids
        remaining, cost, used = qty, 0.0, 0
        for p, q in levels:
            take = min(remaining, q)
            cost += take * p
            remaining -= take
            used += 1
            if remaining <= 1e-12:
                break
        if remaining > 1e-12:
            worst = levels[-1][0] if levels else float("nan")
            cost += remaining * worst * (1 + side * 0.001)
        return cost / qty, qty, used


class SpoofDetector:
    """A wall that appears and vanishes within `max_life_s` without price ever
    trading at its level is counted as a likely spoof."""

    def __init__(self, k: float = 5.0, max_life_s: float = 10.0, window_s: float = 600.0):
        self.k = k
        self.max_life_s = max_life_s
        self.window_s = window_s
        self.active: dict[tuple[str, float], dict] = {}
        self.events: deque = deque()

    def on_book(self, book: OrderBook, now: float) -> None:
        current = {(w.side, w.price): w for w in book.walls(self.k)}
        for key, w in current.items():
            if key not in self.active:
                self.active[key] = {"first": now, "qty": w.qty, "touched": False}
        for key in list(self.active):
            if key in current:
                continue
            info = self.active.pop(key)
            side, price = key
            away = (side == "bid" and book.best_bid > price) or (side == "ask" and book.best_ask < price)
            if now - info["first"] <= self.max_life_s and not info["touched"] and away:
                self.events.append((now, side, price, info["qty"]))
        self._expire(now)

    def on_trade(self, price: float) -> None:
        for (side, level), info in self.active.items():
            if (side == "bid" and price <= level) or (side == "ask" and price >= level):
                info["touched"] = True

    def _expire(self, now: float) -> None:
        while self.events and now - self.events[0][0] > self.window_s:
            self.events.popleft()

    def recent(self, side: str, now: float) -> int:
        self._expire(now)
        return sum(1 for e in self.events if e[1] == side)


class TradeFlow:
    """Taker buy vs sell notional over a rolling window (aggTrade 'm' flag)."""

    def __init__(self, window_s: float = 60.0):
        self.window_s = window_s
        self.trades: deque = deque()

    def add(self, now: float, price: float, qty: float, buyer_is_maker: bool) -> None:
        self.trades.append((now, price * qty, not buyer_is_maker))
        while self.trades and now - self.trades[0][0] > self.window_s:
            self.trades.popleft()

    def imbalance(self) -> tuple[float, float]:
        buy = sum(n for _, n, is_buy in self.trades if is_buy)
        sell = sum(n for _, n, is_buy in self.trades if not is_buy)
        total = buy + sell
        return ((buy - sell) / total if total > 0 else 0.0), total


def orderflow_signal(book: OrderBook, flow: TradeFlow, spoof: SpoofDetector, now: float,
                     base_asset: str = "") -> tuple[float | None, str]:
    """Combine book imbalance (discounted when that side shows spoofing) with taker flow."""
    if not book.is_fresh(now=now):
        return None, ""
    imb = book.imbalance(10)
    spoof_bid, spoof_ask = spoof.recent("bid", now), spoof.recent("ask", now)
    if imb > 0 and spoof_bid >= 2:
        imb *= 0.5
    elif imb < 0 and spoof_ask >= 2:
        imb *= 0.5
    flow_imb, flow_notional = flow.imbalance()
    score = float(np.clip(0.6 * imb + 0.4 * flow_imb, -1.0, 1.0))
    parts = [f"Emir defteri dengesizliği {imb:+.2f}",
             f"son {int(flow.window_s)} sn alıcı payı %{(flow_imb + 1) * 50:.0f}"]
    walls = sorted(book.walls(), key=lambda w: -w.qty)[:2]
    for w in walls:
        parts.append(f"{'alış' if w.side == 'bid' else 'satış'} duvarı {fmt.price(w.price)} ({fmt.qty(w.qty)} {base_asset})")
    if spoof_bid or spoof_ask:
        parts.append(f"spoofing şüphesi: {spoof_bid} alış / {spoof_ask} satış duvarı kayboldu (10 dk)")
    return score, " · ".join(parts)
