"""Position sizing: fractional Kelly risk budget, ATR stop distance, hard caps.

risk_amount = equity * risk_fraction (Kelly-adjusted) * volatility multiplier
qty         = risk_amount / (atr_stop_mult * ATR)

so a wider (more volatile) stop automatically means a smaller position.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from cappatrader.config import RiskConfig
from cappatrader.models import Trade
from cappatrader.regime.detector import VOL_HIGH


@dataclass
class KellyStats:
    n: int
    win_rate: float
    payoff: float  # average win R / average loss R
    kelly: float


def kelly_stats(trades: list[Trade], lookback: int) -> KellyStats:
    recent = trades[-lookback:]
    n = len(recent)
    if n == 0:
        return KellyStats(0, float("nan"), float("nan"), float("nan"))
    wins = [t.r_multiple for t in recent if t.pnl > 0]
    losses = [-t.r_multiple for t in recent if t.pnl <= 0]
    w = len(wins) / n
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    if avg_loss <= 0:
        return KellyStats(n, w, float("inf"), w if wins else 0.0)
    payoff = avg_win / avg_loss
    kelly = w - (1 - w) / payoff if payoff > 0 else -1.0
    return KellyStats(n, w, payoff, kelly)


def risk_fraction(cfg: RiskConfig, trades: list[Trade]) -> tuple[float, str]:
    ks = kelly_stats(trades, cfg.kelly_lookback)
    if ks.n < cfg.kelly_min_trades:
        return cfg.base_risk_per_trade, (
            f"Kelly için yetersiz geçmiş ({ks.n}/{cfg.kelly_min_trades} işlem) → sabit %{cfg.base_risk_per_trade * 100:.2f} risk"
        )
    if ks.kelly <= 0:
        return cfg.min_risk_per_trade, (
            f"Kelly negatif (kazanma %{ks.win_rate * 100:.0f}, K/Z {ks.payoff:.2f}) → minimum %{cfg.min_risk_per_trade * 100:.2f} risk"
        )
    frac = min(max(cfg.kelly_fraction * ks.kelly, cfg.min_risk_per_trade), cfg.max_risk_per_trade)
    return frac, (
        f"Kelly: kazanma %{ks.win_rate * 100:.0f}, K/Z {ks.payoff:.2f} → f*={ks.kelly:.3f}; "
        f"{cfg.kelly_fraction:g}×Kelly, tavan %{cfg.max_risk_per_trade * 100:.1f} → %{frac * 100:.2f} risk"
    )


@dataclass
class SizingResult:
    qty: float = 0.0
    stop_price: float = 0.0
    stop_distance: float = 0.0
    risk_amount: float = 0.0
    risk_fraction: float = 0.0
    notes: list[str] = field(default_factory=list)
    rejected: str = ""

    @property
    def ok(self) -> bool:
        return not self.rejected and self.qty > 0


def round_step(qty: float, step: float | None) -> float:
    if not step or step <= 0:
        return qty
    return math.floor(qty / step + 1e-9) * step


def size_position(
    cfg: RiskConfig,
    equity: float,
    available_cash: float,
    price: float,
    atr: float,
    side: int,
    vol_state: str,
    trades: list[Trade],
    step_size: float | None = None,
    min_notional: float | None = None,
    round_trip_cost_pct: float = 0.0,
    size_mult: float | None = None,
) -> SizingResult:
    res = SizingResult()
    if not (price > 0 and atr > 0 and equity > 0):
        res.rejected = "Geçersiz fiyat/ATR/özsermaye"
        return res
    frac, kelly_note = risk_fraction(cfg, trades)
    res.notes.append(kelly_note)
    mult = 1.0
    if size_mult is not None and math.isfinite(size_mult):
        mult = min(max(size_mult, 0.0), 1.0)
        if mult < 1.0:
            res.notes.append(f"GARCH volatilite tahmini uzun dönem ortalamasının üstünde → boyut ×{mult:.2f}")
    if vol_state == VOL_HIGH and cfg.high_vol_size_mult < mult:
        mult = cfg.high_vol_size_mult
        res.notes.append(f"Yüksek volatilite rejimi → boyut ×{mult:g}")
    res.risk_fraction = frac * mult
    res.risk_amount = equity * res.risk_fraction
    res.stop_distance = cfg.atr_stop_mult * atr
    res.notes.append(
        f"Stop mesafesi {cfg.atr_stop_mult:g}×ATR = {res.stop_distance / price * 100:.2f}% "
        f"(ATR yüksekse pozisyon otomatik küçülür)"
    )
    if round_trip_cost_pct > 0 and cfg.max_cost_r > 0:
        floor_dist = price * round_trip_cost_pct / cfg.max_cost_r
        if floor_dist > res.stop_distance:
            res.stop_distance = floor_dist
            res.notes.append(
                f"Maliyet tabanı: gidiş-dönüş maliyet %{round_trip_cost_pct * 100:.2f} riskin en fazla "
                f"%{cfg.max_cost_r * 100:.0f}'i olabilir → stop {floor_dist / price * 100:.2f}%'e genişletildi"
            )
    qty = res.risk_amount / res.stop_distance
    cap_notional = min(cfg.max_position_pct * equity, available_cash * 0.995)
    if qty * price > cap_notional:
        qty = max(cap_notional, 0.0) / price
        res.notes.append(
            f"Pozisyon tavanı (%{cfg.max_position_pct * 100:.0f} kasa / kullanılabilir nakit) → gerçek risk "
            f"{qty * res.stop_distance:.2f} USDT"
        )
    qty = round_step(qty, step_size)
    floor = max(cfg.min_notional, min_notional or 0.0)
    if qty * price < floor:
        res.rejected = f"Emir büyüklüğü {qty * price:.2f} USDT < minimum {floor:.2f} USDT"
        return res
    res.qty = qty
    res.stop_price = price - side * res.stop_distance
    res.risk_amount = qty * res.stop_distance
    return res
