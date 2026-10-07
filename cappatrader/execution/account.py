"""Cash/position ledger shared by the backtester and the live paper broker.

Longs and (optional, simulated 1x perpetual) shorts both lock qty*entry_price
as collateral, so equity = cash + sum(collateral + open P&L).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from cappatrader.models import Fill, Position, Trade


class Account:
    def __init__(self, starting_balance: float):
        self.starting_balance = float(starting_balance)
        self.cash = float(starting_balance)
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.peak_equity = float(starting_balance)
        self.halted = False
        self.halt_reason = ""
        self.paused_until: str | None = None
        self.day: str | None = None
        self.day_start_equity = float(starting_balance)
        self.symbol_halts: dict[str, str] = {}
        self.last_exit: dict[str, str] = {}

    # -- valuation -------------------------------------------------------
    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(p.value(prices.get(s, p.entry_price)) for s, p in self.positions.items())

    def exposure(self, prices: dict[str, float]) -> float:
        return sum(p.qty * prices.get(s, p.entry_price) for s, p in self.positions.items())

    @property
    def realized_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def total_fees(self) -> float:
        return sum(t.fees for t in self.trades) + sum(p.entry_fee for p in self.positions.values())

    # -- ledger ----------------------------------------------------------
    def open_position(self, pos: Position, fill: Fill) -> None:
        if pos.symbol in self.positions:
            raise ValueError(f"{pos.symbol} için zaten açık pozisyon var")
        cost = fill.qty * fill.price + fill.fee
        if cost > self.cash + 1e-9:
            raise ValueError(f"Yetersiz bakiye: gereken {cost:.2f}, mevcut {self.cash:.2f}")
        self.cash -= cost
        self.positions[pos.symbol] = pos

    def close_position(self, symbol: str, fill: Fill, reason: str, when: str) -> Trade:
        pos = self.positions.pop(symbol)
        gross = pos.unrealized(fill.price)
        self.cash += pos.value(fill.price) - fill.fee
        fees = pos.entry_fee + fill.fee
        pnl = gross - fees
        risk = pos.risk_per_unit * pos.qty
        trade = Trade(
            symbol=symbol, side=pos.side, qty=pos.qty, entry_price=pos.entry_price, exit_price=fill.price,
            entry_time=pos.entry_time, exit_time=when, gross_pnl=gross, fees=fees,
            slippage_cost=pos.entry_slippage_cost + fill.slippage_cost, pnl=pnl,
            r_multiple=pnl / risk if risk > 0 else 0.0, exit_reason=reason,
            entry_regime=pos.entry_regime, entry_reasons=list(pos.reasons),
        )
        self.trades.append(trade)
        self.last_exit[symbol] = when
        return trade

    # -- persistence -----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "starting_balance": self.starting_balance,
            "cash": self.cash,
            "positions": {s: p.to_dict() for s, p in self.positions.items()},
            "trades": [asdict(t) for t in self.trades],
            "peak_equity": self.peak_equity,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "paused_until": self.paused_until,
            "day": self.day,
            "day_start_equity": self.day_start_equity,
            "symbol_halts": dict(self.symbol_halts),
            "last_exit": dict(self.last_exit),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Account:
        acct = cls(data["starting_balance"])
        acct.cash = data["cash"]
        acct.positions = {s: Position.from_dict(p) for s, p in data.get("positions", {}).items()}
        acct.trades = [Trade.from_dict(t) for t in data.get("trades", [])]
        for key in ("peak_equity", "halted", "halt_reason", "paused_until", "day", "day_start_equity"):
            if key in data:
                setattr(acct, key, data[key])
        acct.symbol_halts = dict(data.get("symbol_halts", {}))
        acct.last_exit = dict(data.get("last_exit", {}))
        return acct
