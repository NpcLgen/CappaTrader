"""Transaction-cost model: taker fee + slippage + market impact + latency drift.

slippage_bps = base
             + vol_coef * ATR(bps)                          (wider bars, wider spreads)
             + impact_coef * sqrt(order notional / bar quote volume)
             + latency_z * ATR(bps) * sqrt(latency / bar length)   (price drift while the order travels)
             + extra (stress test knob: "what if every fill is X bps worse?")

Every fill is moved against us by this amount; the fee is charged on the fill value.
"""

from __future__ import annotations

import math

from cappatrader.config import CostConfig
from cappatrader.models import Fill


class CostModel:
    def __init__(self, cfg: CostConfig, bar_ms: int):
        self.cfg = cfg
        self.bar_ms = bar_ms

    def slippage_bps(self, price: float, atr: float, notional: float, bar_quote_volume: float | None) -> float:
        c = self.cfg
        atr_bps = (atr / price * 1e4) if (price > 0 and atr and math.isfinite(atr)) else 0.0
        bps = c.base_slippage_bps + c.vol_slippage_coef * atr_bps
        if bar_quote_volume and bar_quote_volume > 0 and math.isfinite(bar_quote_volume):
            bps += c.impact_coef_bps * math.sqrt(notional / bar_quote_volume)
        if c.latency_ms > 0:
            bps += c.latency_z * atr_bps * math.sqrt(c.latency_ms / self.bar_ms)
        return bps + c.extra_slippage_bps

    def round_trip_pct(self, price: float, atr: float, notional: float, bar_quote_volume: float | None) -> float:
        """Expected entry+exit cost as a fraction of price (fees + slippage both ways)."""
        return 2.0 * (self.cfg.taker_fee + self.slippage_bps(price, atr, notional, bar_quote_volume) / 1e4)

    def fill(self, symbol: str, side: int, qty: float, ref_price: float, atr: float,
             bar_quote_volume: float | None, timestamp) -> Fill:
        bps = self.slippage_bps(ref_price, atr, qty * ref_price, bar_quote_volume)
        price = ref_price * (1.0 + side * bps / 1e4)
        return Fill(symbol=symbol, side=side, qty=qty, price=price, ref_price=ref_price,
                    fee=qty * price * self.cfg.taker_fee, slippage_bps=bps,
                    latency_ms=self.cfg.latency_ms, timestamp=timestamp)
