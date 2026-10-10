"""Walk-forward optimisation: optimise in-sample, judge only on unseen out-of-sample.

    |---- IS (optimise grid) ----|-- OOS (trade best params) --|
                     |---- IS ----|-- OOS --|   (rolled forward by OOS length)

The OOS windows are traded as ONE continuous account (balance, open-trade history and
a fired kill-switch carry over), so the OOS equity is the honest estimate of live
performance. If it is not profitable, the in-sample optimum was curve fitting.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from cappatrader.backtest.engine import Backtester, benchmark_curve
from cappatrader.backtest.metrics import compute_metrics, format_metrics
from cappatrader.config import Config
from cappatrader.models import Trade
from cappatrader.strategy.pipeline import Layers, prepare_universe

# Params that change the prepared frame; everything else (risk, ensemble
# thresholds, costs) only changes how the backtester acts on it.
_FRAME_PREFIXES = ("signals.", "regime.", "ensemble.weights", "pairs", "strategy.squeeze.")


@dataclass
class WindowResult:
    index: int
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    params: dict[str, Any]
    is_metrics: dict
    oos_metrics: dict


@dataclass
class WalkForwardResult:
    windows: list[WindowResult]
    oos_equity: pd.Series
    oos_trades: list[Trade]
    oos_metrics: dict
    wfe: float
    verdict: bool
    verdict_reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["══ Walk-Forward Analizi ══"]
        for w in self.windows:
            lines.append(
                f"#{w.index + 1} IS {w.is_start:%Y-%m-%d}→{w.is_end:%Y-%m-%d} Sharpe {w.is_metrics.get('sharpe', 0):+.2f} "
                f"getiri {w.is_metrics.get('total_return', 0) * 100:+.1f}% | OOS {w.oos_start:%Y-%m-%d}→{w.oos_end:%Y-%m-%d} "
                f"Sharpe {w.oos_metrics.get('sharpe', 0):+.2f} getiri {w.oos_metrics.get('total_return', 0) * 100:+.1f}% "
                f"({w.oos_metrics.get('trades', 0)} işlem) | {_fmt_params(w.params)}"
            )
        lines.append(format_metrics(self.oos_metrics, "Birleştirilmiş OOS (hiç görülmemiş veri)"))
        lines.append(f"Walk-forward verimliliği (OOS/IS yıllık getiri): {self.wfe:.2f}")
        lines.append(("SONUÇ: GEÇTİ ✔ — " if self.verdict else "SONUÇ: BAŞARISIZ ✘ — ") + "; ".join(self.verdict_reasons))
        return "\n".join(lines)


def _fmt_params(p: dict[str, Any]) -> str:
    return ", ".join(f"{k.split('.')[-1]}={v}" for k, v in p.items()) or "varsayılan"


def _frame_key(combo: dict[str, Any]) -> tuple:
    return tuple(sorted((k, v) for k, v in combo.items() if k.startswith(_FRAME_PREFIXES)))


def _objective(m: dict, name: str, min_trades: int) -> float:
    if m.get("trades", 0) < min_trades:
        return -math.inf
    if name == "return":
        return m.get("total_return", -math.inf)
    if name == "calmar":
        return m.get("calmar", -math.inf)
    return m.get("sharpe", -math.inf)


def walk_forward(
    cfg: Config,
    data: dict[str, pd.DataFrame],
    ml_probas: dict[str, pd.Series] | None = None,
    progress: Callable[[str], None] | None = None,
    layers: dict[str, Layers] | None = None,
) -> WalkForwardResult:
    wf = cfg.walkforward
    master = None
    for df in data.values():
        master = df.index if master is None else master.union(df.index)
    n = len(master)
    if n < wf.is_bars + wf.oos_bars:
        raise ValueError(f"Walk-forward için en az {wf.is_bars + wf.oos_bars} bar gerekli, mevcut {n}")

    keys = list(wf.grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(wf.grid[k] for k in keys))] or [{}]
    cfgs, frame_cache = [], {}
    for combo in combos:
        try:
            c = cfg.with_overrides(combo)
        except ValueError:
            continue  # e.g. ema_fast >= ema_slow
        key = _frame_key(combo)
        if key not in frame_cache:
            # A grid over regime.* changes the HMM/GARCH layers themselves: recompute them.
            reuse = layers if not any(k.startswith("regime.") for k in combo) else None
            frame_cache[key] = prepare_universe(data, c, ml_probas, layers=reuse)
        cfgs.append((combo, c, frame_cache[key]))

    windows: list[WindowResult] = []
    oos_curves, oos_trades, oos_events = [], [], []
    acct = None
    start = 0
    while start + wf.is_bars + wf.oos_bars <= n:
        is_s, is_e = master[start], master[start + wf.is_bars]
        oos_s = is_e
        oos_end_idx = start + wf.is_bars + wf.oos_bars
        oos_e = master[oos_end_idx] if oos_end_idx < n else master[-1] + pd.Timedelta(milliseconds=cfg.bar_ms)
        best, best_score = None, -math.inf
        for combo, c, frames in cfgs:
            r = Backtester(c).run(frames, is_s, is_e, collect_reasons=False)
            score = _objective(r.metrics, wf.objective, wf.min_trades)
            if score > best_score:
                best, best_score = (combo, c, frames, r.metrics), score
        if best is None:
            combo, c, frames = cfgs[0]
            best = (combo, c, frames, {})
        combo, c, frames, is_m = best
        oos = Backtester(c).run(frames, oos_s, oos_e, account=acct)
        acct = oos.account
        oos_events.extend(oos.events)
        windows.append(WindowResult(len(windows), is_s, is_e, oos_s, oos_e, combo, is_m, oos.metrics))
        oos_curves.append(oos.equity)
        oos_trades.extend(oos.trades)
        if progress:
            progress(f"Pencere {len(windows)}: OOS getiri {oos.metrics.get('total_return', 0) * 100:+.2f}% ({_fmt_params(combo)})")
        start += wf.oos_bars

    curves = [c.dropna() for c in oos_curves if len(c.dropna())]
    equity = pd.concat(curves) if curves else pd.Series(dtype=float)
    first_frames = cfgs[0][2]
    bench = benchmark_curve(first_frames, equity.index, cfg.account.starting_balance) if len(equity) else None
    oos_m = compute_metrics(equity, oos_trades, cfg.bars_per_year, bench)

    is_ann = [w.is_metrics.get("cagr", 0.0) for w in windows if w.is_metrics]
    oos_ann = [w.oos_metrics.get("cagr", 0.0) for w in windows]
    mean_is = float(np.mean(is_ann)) if is_ann else 0.0
    wfe = float(np.mean(oos_ann) / mean_is) if mean_is > 0 else float("nan")

    reasons, ok = [], True
    if oos_m.get("total_return", 0.0) <= 0:
        ok = False
        reasons.append(f"OOS toplam getiri {oos_m.get('total_return', 0) * 100:+.2f}% (kârsız: strateji geçmişi ezberlemiş)")
    else:
        reasons.append(f"OOS toplam getiri {oos_m['total_return'] * 100:+.2f}%")
    profitable = sum(1 for w in windows if w.oos_metrics.get("total_return", 0) > 0)
    share = profitable / len(windows) if windows else 0.0
    if share < 0.5:
        ok = False
        reasons.append(f"kârlı OOS pencere oranı {profitable}/{len(windows)} (< %50, sonuç tutarsız)")
    else:
        reasons.append(f"kârlı OOS pencere oranı {profitable}/{len(windows)}")
    if not (wfe >= 0.5):
        ok = False
        reasons.append("walk-forward verimliliği < 0.5 (IS performansı OOS'a taşınmıyor)")
    if oos_m.get("sharpe", 0.0) < 0.5:
        ok = False
        reasons.append(f"OOS Sharpe {oos_m.get('sharpe', 0):.2f} < 0.5")
    for ts, ev, text in oos_events:
        if ev == "KILL_SWITCH":
            reasons.append(f"OOS'ta {ts:%Y-%m-%d} tarihinde kill-switch tetiklendi, sonrasında işlem yok "
                           "(canlı hesapta da böyle olurdu)")
            break
    return WalkForwardResult(windows, equity, oos_trades, oos_m, wfe, ok, reasons)
