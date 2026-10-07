"""Portfolio-level guards: drawdown kill-switch, daily loss pause, exposure limits."""

from __future__ import annotations

import pandas as pd

from cappatrader.config import RiskConfig
from cappatrader.execution.account import Account


def _ts(value: str | None) -> pd.Timestamp | None:
    return pd.Timestamp(value) if value else None


class RiskManager:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg

    def update_guards(self, acct: Account, equity: float, now: pd.Timestamp) -> list[tuple[str, str]]:
        """Call once per bar close (or tick in live). Returns (event, message) pairs."""
        events: list[tuple[str, str]] = []
        day = now.strftime("%Y-%m-%d")
        if acct.day != day:
            acct.day = day
            acct.day_start_equity = equity
        acct.peak_equity = max(acct.peak_equity, equity)
        dd = 1.0 - equity / acct.peak_equity if acct.peak_equity > 0 else 0.0
        if not acct.halted and dd >= self.cfg.max_drawdown:
            acct.halted = True
            acct.halt_reason = (
                f"Maksimum düşüş %{dd * 100:.1f} ≥ limit %{self.cfg.max_drawdown * 100:.0f} "
                f"(zirve {acct.peak_equity:,.2f} → {equity:,.2f} USDT)"
            )
            events.append(("KILL_SWITCH", acct.halt_reason))
        day_ret = equity / acct.day_start_equity - 1.0 if acct.day_start_equity > 0 else 0.0
        paused = _ts(acct.paused_until)
        if day_ret <= -self.cfg.daily_loss_limit and (paused is None or now >= paused):
            until = now.normalize() + pd.Timedelta(days=1)
            acct.paused_until = until.isoformat()
            events.append((
                "DAILY_LIMIT",
                f"Günlük zarar %{day_ret * 100:.2f} ≤ -%{self.cfg.daily_loss_limit * 100:.1f}; "
                f"yeni işlemler {until:%Y-%m-%d %H:%M} UTC'ye kadar durduruldu",
            ))
        return events

    def entry_allowed(self, acct: Account, symbol: str, now: pd.Timestamp, pending_entries: int = 0) -> tuple[bool, str]:
        if acct.halted:
            return False, f"Kill-switch aktif: {acct.halt_reason}"
        paused = _ts(acct.paused_until)
        if paused is not None and now < paused:
            return False, "Günlük zarar limiti nedeniyle yeni işlemler duraklatıldı"
        halt = _ts(acct.symbol_halts.get(symbol))
        if halt is not None and now < halt:
            return False, "Ani fiyat hareketi (flash) koruması aktif"
        if symbol in acct.positions:
            return False, "Bu paritede zaten açık pozisyon var"
        if len(acct.positions) + pending_entries >= self.cfg.max_open_positions:
            return False, f"Maksimum açık pozisyon sayısı ({self.cfg.max_open_positions})"
        return True, ""
