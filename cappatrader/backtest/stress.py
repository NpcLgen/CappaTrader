"""Slippage / latency stress test.

Answers: "If every fill is X bps worse, or the order arrives late, is the
strategy still profitable?" — instead of assuming fills at the signal price.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from cappatrader.backtest.engine import Backtester
from cappatrader.config import Config
from cappatrader.strategy.pipeline import prepare_universe


@dataclass
class StressRow:
    scenario: str
    value: float
    total_return: float
    sharpe: float
    max_drawdown: float
    profit_factor: float
    trades: int
    fees: float
    slippage_cost: float


@dataclass
class StressResult:
    ideal: StressRow
    realistic: StressRow
    slippage: list[StressRow] = field(default_factory=list)
    latency: list[StressRow] = field(default_factory=list)
    delay: list[StressRow] = field(default_factory=list)
    breakeven_extra_bps: float = float("nan")

    def summary(self) -> str:
        def row(r: StressRow, label: str) -> str:
            return (f"{label:<34} getiri {r.total_return * 100:+8.2f}%  Sharpe {r.sharpe:+6.2f}  MaxDD {r.max_drawdown * 100:5.1f}%  "
                    f"PF {r.profit_factor:5.2f}  işlem {r.trades:4d}  komisyon {r.fees:8.2f}  slip {r.slippage_cost:8.2f}")

        lines = ["══ Slippage & Gecikme Stres Testi (kill-switch kapalı, yalnız maliyet etkisi) ══",
                 row(self.ideal, "İdeal (sıfır maliyet, sinyal fiyatı)"),
                 row(self.realistic, "Gerçekçi (varsayılan maliyet modeli)"),
                 "— Her dolumda ek kayma —"]
        lines += [row(r, f"+{r.value:g} bps") for r in self.slippage]
        lines.append("— Emir gecikmesi —")
        lines += [row(r, f"{r.value:g} ms") for r in self.latency]
        lines.append("— Sinyalden sonra bar gecikmesi —")
        lines += [row(r, f"{int(r.value)} bar geç") for r in self.delay]
        be = self.breakeven_extra_bps
        if self.realistic.total_return <= 0:
            verdict = "Strateji gerçekçi maliyetlerle ZATEN kârsız: kâğıt üstündeki kâr maliyetlere gidiyor."
        elif math.isinf(be):
            verdict = f"Test edilen en kötü senaryoda (+{self.slippage[-1].value:g} bps) bile kârlı."
        else:
            verdict = f"Başabaş noktası: dolum başına ~+{be:.1f} bps ek kayma. Bunun üzerinde strateji para kaybeder."
        lines.append("SONUÇ: " + verdict)
        return "\n".join(lines)


def _row(cfg: Config, frames, scenario: str, value: float) -> StressRow:
    m = Backtester(cfg).run(frames, collect_reasons=False).metrics
    return StressRow(scenario, value, m.get("total_return", 0.0), m.get("sharpe", 0.0), m.get("max_drawdown", 0.0),
                     m.get("profit_factor", 0.0), m.get("trades", 0), m.get("fees", 0.0), m.get("slippage_cost", 0.0))


def _breakeven(rows: list[StressRow]) -> float:
    if not rows or rows[0].total_return <= 0:
        return 0.0
    for a, b in zip(rows, rows[1:]):
        if a.total_return > 0 >= b.total_return:
            return a.value + (b.value - a.value) * a.total_return / (a.total_return - b.total_return)
    return math.inf


def stress_test(cfg: Config, data: dict[str, pd.DataFrame], ml_probas: dict[str, pd.Series] | None = None) -> StressResult:
    # Portfolio guards are switched off so a kill-switch halt cannot mask the cost effect.
    cfg = cfg.with_overrides({"risk.max_drawdown": 0.99, "risk.daily_loss_limit": 1.0})
    frames = prepare_universe(data, cfg, ml_probas)  # costs do not change the signals
    ideal_cfg = cfg.with_overrides({
        "costs.taker_fee": 0.0, "costs.base_slippage_bps": 0.0, "costs.vol_slippage_coef": 0.0,
        "costs.impact_coef_bps": 0.0, "costs.latency_ms": 0.0, "costs.extra_slippage_bps": 0.0,
        "risk.max_cost_r": 0.0,
    })
    res = StressResult(ideal=_row(ideal_cfg, frames, "ideal", 0), realistic=_row(cfg, frames, "realistic", 0))
    for bps in sorted(cfg.stress.extra_slippage_bps):
        res.slippage.append(_row(cfg.with_overrides({"costs.extra_slippage_bps": bps}), frames, "slippage", bps))
    for ms in cfg.stress.latency_ms:
        res.latency.append(_row(cfg.with_overrides({"costs.latency_ms": ms}), frames, "latency", ms))
    for bars in cfg.stress.delay_bars:
        res.delay.append(_row(cfg.with_overrides({"costs.execution_delay_bars": bars}), frames, "delay", bars))
    res.breakeven_extra_bps = _breakeven(res.slippage)
    return res
