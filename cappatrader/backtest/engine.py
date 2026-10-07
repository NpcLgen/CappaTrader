"""Event-driven multi-symbol backtester.

Bar loop, per timestamp:
  1. at the open: execute orders decided on the previous close (+ optional delay)
     with slippage/fee/latency costs, then check stops/targets against the bar's
     range and ratchet trailing stops (effective from the next bar);
  2. portfolio guards (drawdown kill-switch, daily loss pause) at the close;
  3. decisions at the close using only data up to that close.
The same DecisionEngine, sizing, stops and cost model run in live paper trading.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cappatrader.backtest.metrics import compute_metrics, format_metrics
from cappatrader.config import Config
from cappatrader.execution.account import Account
from cappatrader.execution.costs import CostModel
from cappatrader.execution.orders import build_position
from cappatrader.models import Decision, Trade
from cappatrader.risk.manager import RiskManager
from cappatrader.risk.sizing import size_position
from cappatrader.risk.stops import stop_hit, stop_reason, take_profit_hit, update_trailing
from cappatrader.strategy.decision import DecisionEngine, FrameView
from cappatrader.strategy.explain import explain
from cappatrader.strategy.pipeline import prepare_universe


@dataclass
class BacktestResult:
    equity: pd.Series
    benchmark: pd.Series
    trades: list[Trade]
    metrics: dict
    events: list[tuple] = field(default_factory=list)

    def summary(self, title: str = "Backtest") -> str:
        return format_metrics(self.metrics, title)


def benchmark_curve(frames: dict[str, pd.DataFrame], index: pd.DatetimeIndex, start_balance: float) -> pd.Series:
    """Equal-weight buy & hold of every traded symbol over the same bars."""
    curves = []
    for f in frames.values():
        c = f["close"].reindex(index).ffill().bfill()
        if c.notna().any() and c.iloc[0] > 0:
            curves.append(c / c.iloc[0])
    if not curves:
        return pd.Series(start_balance, index=index)
    return pd.concat(curves, axis=1).mean(axis=1) * start_balance


class Backtester:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.costs = CostModel(cfg.costs, cfg.bar_ms)
        self.risk = RiskManager(cfg.risk)
        self.decider = DecisionEngine(cfg)

    # -- order helpers -----------------------------------------------------
    @staticmethod
    def _prev(v: FrameView, col: str, i: int) -> float | None:
        if col not in v.cols:
            return None
        x = v[col][max(i - 1, 0)]
        return float(x) if math.isfinite(x) else None

    def _open(self, acct: Account, s: str, ref: float, v: FrameView, i: int, ts, d: Decision,
              reasons: list[str], last: dict[str, float]) -> None:
        eq = acct.equity({**last, s: ref})
        qv = self._prev(v, "quote_volume", i)
        rt_cost = self.costs.round_trip_pct(ref, d.atr, 0.1 * eq, qv)
        sz = size_position(self.cfg.risk, eq, acct.cash, ref, d.atr, d.side, d.vol_state, acct.trades,
                           round_trip_cost_pct=rt_cost)
        if not sz.ok:
            return
        fill = self.costs.fill(s, d.side, sz.qty, ref, d.atr, qv, ts)
        cost = fill.notional + fill.fee
        if cost > acct.cash:
            qty = sz.qty * acct.cash / cost * 0.999
            if qty * ref < self.cfg.risk.min_notional:
                return
            fill = self.costs.fill(s, d.side, qty, ref, d.atr, qv, ts)
        acct.open_position(build_position(d, fill, sz, self.cfg.risk, reasons, str(ts)), fill)

    def _close(self, acct: Account, s: str, ref: float, v: FrameView, i: int, ts, reason: str) -> None:
        pos = acct.positions[s]
        atr = self._prev(v, "atr", i) or 0.0
        fill = self.costs.fill(s, -pos.side, pos.qty, ref, atr, self._prev(v, "quote_volume", i), ts)
        acct.close_position(s, fill, reason, str(ts))

    # -- main loop -----------------------------------------------------------
    def run(self, frames: dict[str, pd.DataFrame], start=None, end=None, collect_reasons: bool = True) -> BacktestResult:
        cfg = self.cfg
        symbols = list(frames)
        views = {s: FrameView(f) for s, f in frames.items()}
        master = frames[symbols[0]].index
        for s in symbols[1:]:
            master = master.union(frames[s].index)
        if start is not None:
            master = master[master >= pd.Timestamp(start)]
        if end is not None:
            master = master[master < pd.Timestamp(end)]
        lookup = {s: dict(zip(f.index, range(len(f)))) for s, f in frames.items()}

        acct = Account(cfg.account.starting_balance)
        pending: dict[str, dict] = {}
        last: dict[str, float] = {}
        last_exit_k: dict[str, int] = {}
        events: list[tuple] = []
        eq_vals = np.full(len(master), np.nan)
        exposed = np.zeros(len(master), dtype=bool)
        delay = max(int(cfg.costs.execution_delay_bars), 0)

        for k, ts in enumerate(master):
            # 1. open-of-bar execution and intrabar risk
            for s in symbols:
                i = lookup[s].get(ts)
                if i is None:
                    continue
                v = views[s]
                o, h, l, c = v["open"][i], v["high"][i], v["low"][i], v["close"][i]
                p = pending.get(s)
                if p is not None and k >= p["exec_k"]:
                    del pending[s]
                    if p["kind"] == "exit" and s in acct.positions:
                        self._close(acct, s, o, v, i, ts, p["reason"])
                        last_exit_k[s] = k
                    elif p["kind"] == "enter" and s not in acct.positions and not acct.halted:
                        self._open(acct, s, o, v, i, ts, p["decision"], p["reasons"], last)
                pos = acct.positions.get(s)
                if pos is not None:
                    ref = stop_hit(pos, o, h, l)
                    if ref is not None:
                        self._close(acct, s, ref, v, i, ts, stop_reason(pos))
                        last_exit_k[s] = k
                    else:
                        tp = take_profit_hit(pos, o, h, l)
                        if tp is not None:
                            self._close(acct, s, tp, v, i, ts, "Hedef kâr (take-profit) seviyesine ulaşıldı")
                            last_exit_k[s] = k
                        else:
                            update_trailing(pos, h, l, v["atr"][i], cfg.risk)
                last[s] = c

            # 2. portfolio guards at the close
            eq = acct.equity(last)
            for ev, msg in self.risk.update_guards(acct, eq, ts):
                events.append((ts, ev, msg))
                if ev == "KILL_SWITCH":
                    pending = {s: {"kind": "exit", "exec_k": k + 1, "reason": f"Kill-switch: {msg}"}
                               for s in acct.positions}

            # 3. decisions at the close
            n_pending_entries = sum(1 for p in pending.values() if p["kind"] == "enter")
            for s in symbols:
                i = lookup[s].get(ts)
                if i is None or s in pending:
                    continue
                pos = acct.positions.get(s)
                if pos is not None:
                    pos.bars_held += 1
                bse = k - last_exit_k[s] if s in last_exit_k else None
                d = self.decider.evaluate(s, views[s], i, pos, bse)
                if d.action == "EXIT":
                    pending[s] = {"kind": "exit", "exec_k": k + 1 + delay, "reason": d.exit_reason}
                elif d.action in ("ENTER_LONG", "ENTER_SHORT"):
                    ok, _ = self.risk.entry_allowed(acct, s, ts, n_pending_entries)
                    if ok:
                        reasons = explain(views[s].row(i), d.side, cfg) if collect_reasons else []
                        pending[s] = {"kind": "enter", "exec_k": k + 1 + delay, "decision": d, "reasons": reasons}
                        n_pending_entries += 1
            eq_vals[k] = eq
            exposed[k] = bool(acct.positions)

        # close whatever is still open at the final close
        if len(master):
            ts = master[-1]
            for s in list(acct.positions):
                i = lookup[s].get(ts, len(views[s]) - 1)
                self._close(acct, s, last.get(s, views[s]["close"][i]), views[s], i, ts, "Backtest sonu (pozisyon kapatıldı)")
            eq_vals[-1] = acct.equity(last)

        equity = pd.Series(eq_vals, index=master, name="equity")
        bench = benchmark_curve(frames, master, cfg.account.starting_balance)
        metrics = compute_metrics(equity, acct.trades, cfg.bars_per_year, bench,
                                  exposure=float(exposed.mean()) if len(exposed) else 0.0)
        return BacktestResult(equity, bench, acct.trades, metrics, events)


def run_backtest(cfg: Config, data: dict[str, pd.DataFrame], ml_probas: dict[str, pd.Series] | None = None,
                 start=None, end=None) -> BacktestResult:
    frames = prepare_universe(data, cfg, ml_probas)
    return Backtester(cfg).run(frames, start, end)
