import pandas as pd
import pytest

from cappatrader.backtest.engine import Backtester, run_backtest
from cappatrader.backtest.stress import _breakeven, StressRow, stress_test
from cappatrader.backtest.walkforward import walk_forward
from cappatrader.config import CostConfig
from cappatrader.execution.account import Account
from cappatrader.execution.costs import CostModel
from cappatrader.models import LONG, SHORT, Fill, Position
from cappatrader.strategy.pipeline import prepare_universe


def test_cost_model_moves_fills_against_us():
    cm = CostModel(CostConfig(taker_fee=0.001, base_slippage_bps=2, vol_slippage_coef=0, impact_coef_bps=0,
                              latency_ms=0), 3_600_000)
    buy = cm.fill("X", LONG, 1.0, 100.0, 1.0, None, None)
    sell = cm.fill("X", SHORT, 1.0, 100.0, 1.0, None, None)
    assert buy.price == pytest.approx(100.02) and sell.price == pytest.approx(99.98)
    assert buy.fee == pytest.approx(0.10002)
    more = CostModel(CostConfig(extra_slippage_bps=10, vol_slippage_coef=0, impact_coef_bps=0, latency_ms=0), 3_600_000)
    assert more.fill("X", LONG, 1.0, 100.0, 1.0, None, None).price > buy.price


def test_latency_adds_cost():
    base = dict(vol_slippage_coef=0, impact_coef_bps=0, base_slippage_bps=0)
    fast = CostModel(CostConfig(latency_ms=0, **base), 3_600_000).slippage_bps(100, 1, 100, None)
    slow = CostModel(CostConfig(latency_ms=5000, **base), 3_600_000).slippage_bps(100, 1, 100, None)
    assert slow > fast == 0


def test_account_long_and_short_ledger():
    acct = Account(1000)
    pos = Position("X", LONG, 2.0, 100.0, "t0", 90, 90, 10, 100, "TREND_UP", 70, 0.2, 0.0)
    acct.open_position(pos, Fill("X", 1, 2.0, 100.0, 100.0, 0.2, 0, 0, "t0"))
    assert acct.cash == pytest.approx(799.8)
    assert acct.equity({"X": 110}) == pytest.approx(1019.8)
    t = acct.close_position("X", Fill("X", -1, 2.0, 110.0, 110.0, 0.22, 0, 0, "t1"), "test", "t1")
    assert t.pnl == pytest.approx(20 - 0.42) and t.r_multiple == pytest.approx((20 - 0.42) / 20)
    assert acct.cash == pytest.approx(1000 + 20 - 0.42)
    short = Position("X", SHORT, 1.0, 100.0, "t2", 110, 110, 10, 100, "TREND_DOWN", 70, 0.1, 0.0)
    acct.open_position(short, Fill("X", -1, 1.0, 100.0, 100.0, 0.1, 0, 0, "t2"))
    assert acct.equity({"X": 90}) == pytest.approx(acct.cash + 110)
    t2 = acct.close_position("X", Fill("X", 1, 1.0, 90.0, 90.0, 0.09, 0, 0, "t3"), "test", "t3")
    assert t2.gross_pnl == pytest.approx(10)


def test_account_roundtrip_dict():
    acct = Account(1000)
    acct.positions["X"] = Position("X", LONG, 1.0, 100.0, "t0", 90, 90, 10, 100, "RANGE", 70, 0.1, 0.0, ["r"])
    acct.halted = True
    clone = Account.from_dict(acct.to_dict())
    assert clone.positions["X"].reasons == ["r"] and clone.halted


def test_account_rejects_overspend():
    acct = Account(100)
    pos = Position("X", LONG, 2.0, 100.0, "t0", 90, 90, 10, 100, "TREND_UP", 70, 0.2, 0.0)
    with pytest.raises(ValueError):
        acct.open_position(pos, Fill("X", 1, 2.0, 100.0, 100.0, 0.2, 0, 0, "t0"))


@pytest.fixture
def bt_result(cfg, universe):
    return run_backtest(cfg.with_overrides({"signals.ml.enabled": False}), universe)


def test_backtest_invariants(cfg, bt_result):
    r = bt_result
    assert r.metrics["trades"] > 10
    assert r.equity.notna().all()
    # Final equity equals start + net P&L of all (closed) trades.
    assert r.equity.iloc[-1] == pytest.approx(cfg.account.starting_balance + sum(t.pnl for t in r.trades), rel=1e-9)
    assert all(t.fees > 0 for t in r.trades)
    assert all(pd.Timestamp(t.exit_time) >= pd.Timestamp(t.entry_time) for t in r.trades)
    assert all(t.side == LONG for t in r.trades)  # spot: long only by default
    assert all(t.entry_reasons and t.entry_reasons[0].startswith("Rejim") for t in r.trades)


def test_entries_fill_on_next_bar_open(cfg, universe):
    c = cfg.with_overrides({"signals.ml.enabled": False, "costs.base_slippage_bps": 0, "costs.vol_slippage_coef": 0,
                            "costs.impact_coef_bps": 0, "costs.latency_ms": 0})
    r = run_backtest(c, universe)
    t = r.trades[0]
    df = universe[t.symbol]
    assert t.entry_price == pytest.approx(df.loc[pd.Timestamp(t.entry_time), "open"])


def test_shorts_when_allowed(cfg, universe):
    r = run_backtest(cfg.with_overrides({"signals.ml.enabled": False, "account.allow_short": True}), universe)
    assert any(t.side == SHORT for t in r.trades)


def test_execution_delay_changes_fills(cfg, universe):
    base = cfg.with_overrides({"signals.ml.enabled": False})
    frames = prepare_universe(universe, base)
    a = Backtester(base).run(frames, collect_reasons=False)
    b = Backtester(base.with_overrides({"costs.execution_delay_bars": 2})).run(frames, collect_reasons=False)
    assert a.trades[0].entry_time != b.trades[0].entry_time


def test_extra_slippage_hurts(cfg, universe):
    # max_cost_r 0: the cost-aware stop floor would otherwise change which trades are taken
    base = cfg.with_overrides({"signals.ml.enabled": False, "risk.max_drawdown": 0.99, "risk.max_cost_r": 0.0})
    frames = prepare_universe(universe, base)
    a = Backtester(base).run(frames, collect_reasons=False).metrics
    b = Backtester(base.with_overrides({"costs.extra_slippage_bps": 50})).run(frames, collect_reasons=False).metrics
    assert b["slippage_cost"] > a["slippage_cost"]
    assert b["total_return"] < a["total_return"]


def test_breakeven_interpolation():
    rows = [StressRow("s", v, r, 0, 0, 0, 0, 0, 0) for v, r in [(0, 0.10), (10, 0.05), (20, -0.05)]]
    assert _breakeven(rows) == pytest.approx(15.0)
    assert _breakeven(rows[:2]) == float("inf")


def test_stress_report(cfg, universe):
    c = cfg.with_overrides({"signals.ml.enabled": False})
    c.stress.extra_slippage_bps = [0, 20]
    c.stress.latency_ms = [0]
    c.stress.delay_bars = [0]
    res = stress_test(c, universe)
    assert res.ideal.total_return > res.realistic.total_return
    assert "SONUÇ" in res.summary()


def test_walk_forward_windows_do_not_overlap(cfg, universe):
    c = cfg.with_overrides({"signals.ml.enabled": False})
    c.walkforward.is_bars, c.walkforward.oos_bars = 1000, 500
    c.walkforward.grid = {"ensemble.entry_confidence": [60, 70], "risk.atr_stop_mult": [2.0]}
    res = walk_forward(c, universe)
    assert len(res.windows) == 3
    for w, nxt in zip(res.windows, res.windows[1:]):
        assert w.oos_end <= nxt.oos_start
        assert w.is_end == w.oos_start
    assert res.oos_equity.index.is_monotonic_increasing
    assert all(pd.Timestamp(t.entry_time) >= res.windows[0].oos_start for t in res.oos_trades)
    assert "Walk-Forward" in res.summary()


def test_walk_forward_oos_is_one_continuous_account(cfg, universe):
    c = cfg.with_overrides({"account.allow_short": True, "risk.max_drawdown": 0.99, "risk.daily_loss_limit": 1.0})
    c.walkforward.is_bars, c.walkforward.oos_bars = 1000, 500
    c.walkforward.grid = {"strategy.squeeze.atr_stop_mult": [1.5, 2.0]}
    res = walk_forward(c, universe)
    eq = res.oos_equity
    # trade P&L reconciles with the reported OOS balance change
    assert sum(t.pnl for t in res.oos_trades) == pytest.approx(eq.iloc[-1] - c.account.starting_balance, rel=1e-9)
    assert res.oos_metrics["net_pnl"] == pytest.approx(res.oos_metrics["end_equity"] - c.account.starting_balance)


def test_walk_forward_kill_switch_carries_over(cfg, universe):
    c = cfg.with_overrides({"account.allow_short": True, "risk.max_drawdown": 0.01})
    c.walkforward.is_bars, c.walkforward.oos_bars = 1000, 500
    c.walkforward.grid = {"strategy.squeeze.atr_stop_mult": [1.5]}
    res = walk_forward(c, universe)
    halted_in = next(i for i, w in enumerate(res.windows) if w.oos_metrics.get("trades", 0) > 0)
    assert all(w.oos_metrics.get("trades", 0) == 0 for w in res.windows[halted_in + 1:])
    assert any("kill-switch" in r for r in res.verdict_reasons)


def test_reports_are_strict_json(tmp_path):
    import json

    from cappatrader.cli import _save

    _save(tmp_path, "x", {"a": float("inf"), "b": [float("nan"), 1.5], "c": {"d": float("-inf")}})
    data = json.loads((tmp_path / "metrics.json").read_text(), parse_constant=lambda c: pytest.fail(c))
    assert data == {"a": None, "b": [None, 1.5], "c": {"d": None}}


def test_stress_breakeven_uses_realistic_baseline(cfg, universe, monkeypatch):
    import cappatrader.backtest.stress as stress

    returns = {("ideal", 0): 0.3, ("realistic", 0): 0.10, ("slippage", 20): -0.10, ("slippage", 30): -0.20}

    def fake_row(cfg_, frames, scenario, value):
        return stress.StressRow(scenario, value, returns.get((scenario, value), 0.0), 0.0, 0.0, 1.0, 10, 0.0, 0.0)

    monkeypatch.setattr(stress, "_row", fake_row)
    c = cfg.with_overrides({"signals.ml.enabled": False})
    c.stress.extra_slippage_bps = [20, 30]  # no 0 entry in the user's grid
    c.stress.latency_ms = [0]
    c.stress.delay_bars = [0]
    res = stress_test(c, universe)
    # interpolated from the realistic 0-bps run (+10%) to the 20-bps row (-10%); the user's grid alone gives 0
    assert res.breakeven_extra_bps == pytest.approx(10.0)
