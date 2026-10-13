"""Stop-loss, trailing stop (chandelier) and take-profit mechanics."""

from __future__ import annotations

from cappatrader import fmt
from cappatrader.config import RiskConfig
from cappatrader.models import LONG, Position


def update_trailing(pos: Position, high: float, low: float, atr: float, cfg: RiskConfig) -> bool:
    """Ratchet the stop behind price once the trade is `trail_activate_r` in profit.
    The stop never loosens, and when trailing starts it is at least breakeven."""
    if pos.side == LONG:
        pos.best_price = max(pos.best_price, high)
    else:
        pos.best_price = min(pos.best_price, low)
    gain_r = pos.side * (pos.best_price - pos.entry_price) / pos.risk_per_unit if pos.risk_per_unit > 0 else 0.0
    if not pos.trailing_active and gain_r >= cfg.trail_activate_r:
        pos.trailing_active = True
    if not pos.trailing_active or not atr > 0:
        return False
    candidate = pos.best_price - pos.side * cfg.trail_atr_mult * atr
    if pos.side == LONG:
        candidate = max(candidate, pos.entry_price)
        if candidate > pos.stop_price:
            pos.stop_price = candidate
            return True
    else:
        candidate = min(candidate, pos.entry_price)
        if candidate < pos.stop_price:
            pos.stop_price = candidate
            return True
    return False


def stop_hit(pos: Position, open_: float, high: float, low: float) -> float | None:
    """Reference exit price if the bar touched the stop (gap -> the open)."""
    if pos.side == LONG:
        if open_ <= pos.stop_price:
            return open_
        if low <= pos.stop_price:
            return pos.stop_price
    else:
        if open_ >= pos.stop_price:
            return open_
        if high >= pos.stop_price:
            return pos.stop_price
    return None


def take_profit_hit(pos: Position, open_: float, high: float, low: float) -> float | None:
    tp = pos.take_profit
    if tp is None:
        return None
    if pos.side == LONG:
        if open_ >= tp:
            return open_
        if high >= tp:
            return tp
    else:
        if open_ <= tp:
            return open_
        if low <= tp:
            return tp
    return None


def stop_reason(pos: Position) -> str:
    if pos.trailing_active:
        locked = pos.side * (pos.stop_price - pos.entry_price) / pos.entry_price
        return f"İzleyen stop (trailing) tetiklendi — stop {fmt.price(pos.stop_price)}, kilitlenen hareket {locked * 100:+.2f}%"
    return f"Zarar-kes (stop-loss) tetiklendi — stop {fmt.price(pos.stop_price)}"
