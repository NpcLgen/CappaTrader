"""Performance metrics and PRD KPI checks."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from cappatrader.models import Trade

KPI_SHARPE = 1.5
KPI_MAX_DD = 0.15


def compute_metrics(
    equity: pd.Series,
    trades: list[Trade],
    bars_per_year: float,
    benchmark: pd.Series | None = None,
    exposure: float | None = None,
) -> dict:
    m: dict = {}
    eq = equity.dropna()
    if len(eq) < 2:
        return {"trades": len(trades), "total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    ret = eq.pct_change().dropna()
    m["start_equity"] = float(eq.iloc[0])
    m["end_equity"] = float(eq.iloc[-1])
    m["total_return"] = float(eq.iloc[-1] / eq.iloc[0] - 1)
    years = len(eq) / bars_per_year
    m["years"] = years
    m["cagr"] = float((1 + m["total_return"]) ** (1 / years) - 1) if years > 0 and m["total_return"] > -1 else -1.0
    std = ret.std()
    m["volatility"] = float(std * math.sqrt(bars_per_year)) if std > 0 else 0.0
    m["sharpe"] = float(ret.mean() / std * math.sqrt(bars_per_year)) if std > 0 else 0.0
    downside = ret[ret < 0].std()
    m["sortino"] = float(ret.mean() / downside * math.sqrt(bars_per_year)) if downside and downside > 0 else 0.0
    dd = eq / eq.cummax() - 1
    m["max_drawdown"] = float(-dd.min())
    m["calmar"] = float(m["cagr"] / m["max_drawdown"]) if m["max_drawdown"] > 0 else 0.0
    if exposure is not None:
        m["exposure"] = exposure

    n = len(trades)
    m["trades"] = n
    pnl = np.array([t.pnl for t in trades]) if n else np.array([])
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    m["win_rate"] = float(len(wins) / n) if n else 0.0
    m["avg_win"] = float(wins.mean()) if len(wins) else 0.0
    m["avg_loss"] = float(losses.mean()) if len(losses) else 0.0
    m["profit_factor"] = float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else (float("inf") if len(wins) else 0.0)
    m["expectancy_r"] = float(np.mean([t.r_multiple for t in trades])) if n else 0.0
    m["fees"] = float(sum(t.fees for t in trades))
    m["slippage_cost"] = float(sum(t.slippage_cost for t in trades))
    m["gross_pnl"] = float(sum(t.gross_pnl for t in trades))
    m["net_pnl"] = float(pnl.sum()) if n else 0.0
    if n:
        holds = [(pd.Timestamp(t.exit_time) - pd.Timestamp(t.entry_time)).total_seconds() / 3600 for t in trades]
        m["avg_hold_hours"] = float(np.mean(holds))
    if benchmark is not None and len(benchmark.dropna()) > 1:
        b = benchmark.dropna()
        m["benchmark_return"] = float(b.iloc[-1] / b.iloc[0] - 1)
        bdd = b / b.cummax() - 1
        m["benchmark_max_drawdown"] = float(-bdd.min())
        m["alpha_vs_benchmark"] = m["total_return"] - m["benchmark_return"]
    m["kpi"] = {
        "sharpe_gt_1_5": m["sharpe"] > KPI_SHARPE,
        "max_dd_lt_15pct": m["max_drawdown"] < KPI_MAX_DD,
        "beats_benchmark": m.get("alpha_vs_benchmark", 0.0) > 0,
    }
    return m


def _pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def _yes(flag: bool | None) -> str:
    return "✔" if flag else "✘"


def format_metrics(m: dict, title: str = "Sonuç") -> str:
    if not m or "end_equity" not in m:
        return f"{title}: yetersiz veri"
    kpi = m.get("kpi", {})
    lines = [
        f"── {title} ──",
        f"Bakiye: {m['start_equity']:,.2f} → {m['end_equity']:,.2f} USDT ({_pct(m['total_return'])}, yıllık {_pct(m['cagr'])})",
        f"Sharpe {m['sharpe']:.2f} · Sortino {m['sortino']:.2f} · Max DD {m['max_drawdown'] * 100:.2f}% · Calmar {m['calmar']:.2f}",
        f"İşlem {m['trades']} · kazanma %{m['win_rate'] * 100:.1f} · kâr faktörü {m['profit_factor']:.2f} · "
        f"beklenti {m['expectancy_r']:+.2f}R",
        f"Brüt P&L {m['gross_pnl']:+,.2f} · komisyon -{m['fees']:,.2f} · slippage maliyeti -{m['slippage_cost']:,.2f} · "
        f"net {m['net_pnl']:+,.2f} USDT",
    ]
    if "exposure" in m:
        lines.append(f"Piyasada kalma oranı %{m['exposure'] * 100:.1f}" +
                     (f" · ort. pozisyon süresi {m['avg_hold_hours']:.1f} saat" if "avg_hold_hours" in m else ""))
    if "benchmark_return" in m:
        lines.append(f"Benchmark (eşit ağırlıklı al-tut) {_pct(m['benchmark_return'])}, max DD "
                     f"{m['benchmark_max_drawdown'] * 100:.1f}% → alfa {_pct(m['alpha_vs_benchmark'])}")
    lines.append(
        f"PRD KPI: Sharpe>1.5 {_yes(kpi.get('sharpe_gt_1_5'))} · MaxDD<%15 {_yes(kpi.get('max_dd_lt_15pct'))} · "
        f"Benchmark üstü {_yes(kpi.get('beats_benchmark'))}"
    )
    return "\n".join(lines)
