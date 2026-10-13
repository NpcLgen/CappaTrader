"""Turn an approved decision + fill into a tracked position."""

from __future__ import annotations

from cappatrader.config import RiskConfig
from cappatrader.models import Decision, Fill, Position
from cappatrader.risk.sizing import SizingResult


def build_position(d: Decision, fill: Fill, sizing: SizingResult, cfg: RiskConfig,
                   reasons: list[str], when: str) -> Position:
    side = d.side
    stop = fill.price - side * sizing.stop_distance
    tp = fill.price + side * cfg.take_profit_r * sizing.stop_distance if cfg.take_profit_r else None
    return Position(
        symbol=d.symbol, side=side, qty=fill.qty, entry_price=fill.price, entry_time=when,
        stop_price=stop, initial_stop=stop, risk_per_unit=sizing.stop_distance, best_price=fill.price,
        entry_regime=d.regime, entry_confidence=d.confidence, entry_fee=fill.fee,
        entry_slippage_cost=fill.slippage_cost, reasons=list(reasons), take_profit=tp, style=d.style,
    )
