"""Core domain objects shared by backtest and live paper trading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

LONG = 1
SHORT = -1


def side_name(side: int) -> str:
    return "LONG" if side == LONG else "SHORT"


@dataclass
class Decision:
    symbol: str
    timestamp: Any
    action: str  # ENTER_LONG | ENTER_SHORT | EXIT | HOLD
    composite: float
    confidence: float  # 0..100 in the direction of the action (or of the composite for HOLD)
    regime: str
    vol_state: str
    close: float
    atr: float
    reasons: list[str] = field(default_factory=list)
    components: dict[str, float] = field(default_factory=dict)
    exit_reason: str = ""
    style: str = "TREND"  # TREND (momentum), MR (mean reversion) or SQZ (squeeze-VWAP): drives the exit rule
    size_mult: float = float("nan")  # GARCH volatility-forecast size multiplier (NaN = not used)
    # EXIT only: the opposite entry signalled on the same bar (stop-and-reverse), executed after the close fill
    reverse: Decision | None = None

    @property
    def side(self) -> int:
        return SHORT if self.action == "ENTER_SHORT" else LONG


@dataclass
class Fill:
    symbol: str
    side: int  # +1 buy, -1 sell
    qty: float
    price: float
    ref_price: float
    fee: float
    slippage_bps: float
    latency_ms: float
    timestamp: Any

    @property
    def notional(self) -> float:
        return self.qty * self.price

    @property
    def slippage_cost(self) -> float:
        return abs(self.price - self.ref_price) * self.qty


@dataclass
class Position:
    symbol: str
    side: int
    qty: float
    entry_price: float
    entry_time: str
    stop_price: float
    initial_stop: float
    risk_per_unit: float
    best_price: float
    entry_regime: str
    entry_confidence: float
    entry_fee: float
    entry_slippage_cost: float
    reasons: list[str] = field(default_factory=list)
    trailing_active: bool = False
    take_profit: float | None = None
    bars_held: int = 0
    style: str = "TREND"

    @property
    def collateral(self) -> float:
        return self.qty * self.entry_price

    def unrealized(self, price: float) -> float:
        return self.side * (price - self.entry_price) * self.qty

    def value(self, price: float) -> float:
        """Contribution to equity: collateral plus open P&L."""
        return self.collateral + self.unrealized(price)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Position:
        return cls(**data)


@dataclass
class Trade:
    symbol: str
    side: int
    qty: float
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    gross_pnl: float
    fees: float
    slippage_cost: float
    pnl: float  # net of fees (slippage is already inside the fill prices)
    r_multiple: float
    exit_reason: str
    entry_regime: str
    entry_reasons: list[str] = field(default_factory=list)

    @property
    def return_pct(self) -> float:
        base = self.entry_price * self.qty
        return self.pnl / base if base else 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trade:
        return cls(**data)
