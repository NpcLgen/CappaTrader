"""End-to-end wiring of HMM/GARCH regime + squeeze-VWAP strategy through pipeline and backtester."""

import math

import numpy as np
import pandas as pd
import pytest

from cappatrader.backtest.engine import Backtester, run_backtest
from cappatrader.backtest.walkforward import walk_forward
from cappatrader.config import RiskConfig
from cappatrader.models import LONG, SHORT
from cappatrader.risk.sizing import size_position
from cappatrader.strategy.factory import explain_decision, make_decider, strategy_risk, uses_trailing
from cappatrader.strategy.pipeline import prepare_universe, regime_layers
from cappatrader.strategy.squeeze_decision import SqueezeVwapEngine
from cappatrader.strategy.decision import DecisionEngine, FrameView


def test_default_strategy_is_squeeze_with_hmm_garch(cfg):
    assert cfg.strategy.name == "squeeze_vwap"
    assert cfg.regime.method == "hmm" and cfg.regime.vol_method == "garch"
    assert isinstance(make_decider(cfg), SqueezeVwapEngine)
    assert isinstance(make_decider(cfg.with_overrides({"strategy.name": "ensemble"})), DecisionEngine)
    r = strategy_risk(cfg)
    assert r.atr_stop_mult == cfg.strategy.squeeze.atr_stop_mult and r.take_profit_r == 1.1
    assert not uses_trailing(cfg) and math.isinf(r.trail_activate_r)
    assert r.max_cost_r == 0.0 and cfg.risk.max_cost_r > 0  # pure ATR stop for squeeze; floor kept for ensemble
    assert cfg.risk.take_profit_r is None  # the shared risk config itself is untouched


def test_prepared_frame_has_regime_and_squeeze_layers(cfg, universe):
    f = prepare_universe(universe, cfg)["BTCUSDT"]
    for col in ("regime", "vol_state", "hmm_label", "hmm_prob", "garch_vol", "garch_ratio", "garch_size_mult",
                "sqz_val", "sqz_mom_up", "sqz_mom_down", "sqz_on", "vwap", "atr", "composite"):
        assert col in f.columns, col
    tail = f.iloc[-500:]
    assert set(tail["regime"]) <= {"TREND_UP", "TREND_DOWN", "RANGE", "HIGH_VOL", "UNKNOWN"}
    assert (tail["regime"] != "UNKNOWN").mean() > 0.9
    assert tail["garch_vol"].notna().all() and (tail["garch_size_mult"] <= 1.0).all()


def test_full_pipeline_with_hmm_garch_is_causal(cfg, universe):
    cut = 1800
    head = {s: df.iloc[:cut] for s, df in universe.items()}
    full = prepare_universe(universe, cfg)["SOLUSDT"]
    part = prepare_universe(head, cfg)["SOLUSDT"]
    cols = ["regime", "vol_state", "hmm_prob", "garch_vol", "sqz_val", "sqz_mom_up", "vwap", "composite"]
    a = part[cols].iloc[-300:]
    pd.testing.assert_frame_equal(a, full[cols].loc[a.index])


def test_precomputed_layers_match_inline(cfg, universe):
    layers = regime_layers(universe, cfg)
    a = prepare_universe(universe, cfg, layers=layers)["ETHUSDT"]
    b = prepare_universe(universe, cfg)["ETHUSDT"]
    pd.testing.assert_frame_equal(a, b)


@pytest.fixture
def sq_result(cfg, universe):
    c = cfg.with_overrides({"account.allow_short": True, "risk.max_drawdown": 0.99, "risk.daily_loss_limit": 1.0})
    return c, run_backtest(c, universe)


def test_squeeze_backtest_rules(sq_result, universe):
    c, r = sq_result
    assert r.metrics["trades"] > 20
    assert any(t.side == SHORT for t in r.trades) and any(t.side == LONG for t in r.trades)
    reasons = {t.exit_reason.split(" ")[0] for t in r.trades}
    assert not any(t.exit_reason.startswith("İzleyen stop") for t in r.trades)  # no trailing in this strategy
    assert reasons <= {"Hedef", "Zarar-kes", "Momentum", "Backtest"}, reasons
    for t in r.trades:
        if t.exit_reason.startswith("Hedef"):
            # target = 1.1 R before costs: net R below 1.1 (costs are not folded into the stop)
            assert 0.0 < t.r_multiple < 1.1 + 1e-9, t
        if t.exit_reason.startswith("Zarar-kes"):
            assert t.r_multiple < 0
        assert t.entry_reasons and t.entry_reasons[0].startswith("Rejim")
    assert r.equity.iloc[-1] == pytest.approx(c.account.starting_balance + sum(t.pnl for t in r.trades), rel=1e-9)


def test_entries_respect_vwap_and_momentum(sq_result, universe):
    c, r = sq_result
    frames = prepare_universe(universe, c)
    for t in r.trades[:60]:
        f = frames[t.symbol]
        i = f.index.get_loc(pd.Timestamp(t.entry_time)) - 1  # signal bar = bar before the fill
        row = f.iloc[i]
        if t.side == LONG:
            assert row["close"] > row["vwap"] and bool(row["sqz_mom_up"])
        else:
            assert row["close"] < row["vwap"] and bool(row["sqz_mom_down"])
        assert row["regime"] in c.strategy.squeeze.tradeable_regimes
        assert row["vol_state"] != "EXTREME"


def test_legacy_adx_atr_path_still_works(cfg, universe):
    c = cfg.with_overrides({"regime.method": "adx", "regime.vol_method": "atr", "strategy.name": "ensemble"})
    f = prepare_universe(universe, c)["BTCUSDT"]
    assert set(f["regime"].dropna()) <= {"TREND_UP", "TREND_DOWN", "RANGE", "TRANSITION", "UNKNOWN"}
    assert f["hmm_prob"].isna().all() and f["garch_vol"].isna().all()
    assert run_backtest(c.with_overrides({"signals.ml.enabled": False}), universe).metrics["trades"] > 0


def test_ensemble_on_hmm_regime(ens_cfg, universe):
    r = run_backtest(ens_cfg.with_overrides({"signals.ml.enabled": False}), universe)
    assert r.metrics["trades"] > 0
    assert all("HMM" in t.entry_reasons[0] for t in r.trades if t.entry_reasons)


def test_garch_multiplier_shrinks_position():
    cfg = RiskConfig(max_position_pct=1.0)
    full = size_position(cfg, 1000, 1000, 100.0, 1.0, LONG, "NORMAL", [])
    half = size_position(cfg, 1000, 1000, 100.0, 1.0, LONG, "NORMAL", [], size_mult=0.5)
    assert half.qty == pytest.approx(full.qty * 0.5)
    assert any("GARCH" in n for n in half.notes)
    nan = size_position(cfg, 1000, 1000, 100.0, 1.0, LONG, "NORMAL", [], size_mult=float("nan"))
    assert nan.qty == pytest.approx(full.qty)


def test_squeeze_stop_is_pure_atr_and_target_1_1r(cfg, universe, monkeypatch):
    import cappatrader.backtest.engine as eng

    c = cfg.with_overrides({"account.allow_short": True})
    built = []
    orig = eng.build_position

    def spy(d, fill, sz, rcfg, reasons, when):
        pos = orig(d, fill, sz, rcfg, reasons, when)
        built.append((d.atr, pos))
        return pos

    monkeypatch.setattr(eng, "build_position", spy)
    run_backtest(c, universe)
    assert len(built) > 20
    k = c.strategy.squeeze.atr_stop_mult
    for atr, pos in built:
        assert pos.risk_per_unit == pytest.approx(k * atr, rel=1e-12)  # never widened by the cost floor
        assert pos.stop_price == pytest.approx(pos.entry_price - pos.side * k * atr, rel=1e-12)
        assert pos.take_profit == pytest.approx(pos.entry_price + pos.side * 1.1 * k * atr, rel=1e-12)


def test_size_multiplier_flows_from_frame(cfg, universe):
    c = cfg.with_overrides({"account.allow_short": True})
    frames = prepare_universe(universe, c)
    bt = Backtester(c)
    seen = []
    orig = bt._open

    def spy(acct, s, ref, v, i, ts, d, reasons, last):
        seen.append(d.size_mult)
        return orig(acct, s, ref, v, i, ts, d, reasons, last)

    bt._open = spy
    bt.run(frames, collect_reasons=False)
    assert seen and all(np.isfinite(x) and 0 < x <= 1 for x in seen)


def test_walk_forward_squeeze_grid(cfg, universe):
    c = cfg.with_overrides({"account.allow_short": True})
    c.walkforward.is_bars, c.walkforward.oos_bars = 1000, 500
    c.walkforward.grid = {"strategy.squeeze.atr_stop_mult": [1.0, 2.0], "strategy.squeeze.length": [30, 60]}
    res = walk_forward(c, universe, layers=regime_layers(universe, c))
    assert len(res.windows) == 3
    assert all(set(w.params) == {"strategy.squeeze.atr_stop_mult", "strategy.squeeze.length"} for w in res.windows)


def test_explain_decision_dispatch(cfg, universe):
    f = prepare_universe(universe, cfg)["BTCUSDT"]
    row = f.iloc[-1]
    lines = explain_decision(cfg, row, LONG, {"orderflow": "Emir defteri notu"})
    assert lines[0].startswith("Rejim") and any("VWAP" in x for x in lines)
    assert "Emir defteri notu" not in lines  # squeeze ignores order flow: never presented as a reason
    assert "ortalamaya dönüş" not in lines[0]  # the RANGE hint belongs to the ensemble strategy
    ens = cfg.with_overrides({"strategy.name": "ensemble"})
    assert any("Ensemble skoru" in x for x in explain_decision(ens, row, LONG))


def test_momentum_turn_reverses_long_into_short(cfg, universe):
    # LONG closes when momentum turns to sellers; below VWAP that same turn is the SHORT rule, so the short
    # is opened right after the close fill (stop-and-reverse) instead of being lost.
    c = cfg.with_overrides({"account.allow_short": True, "risk.max_drawdown": 0.99, "risk.daily_loss_limit": 1.0,
                            "risk.max_open_positions": 10})
    frames = prepare_universe(universe, c)
    r = Backtester(c).run(frames, collect_reasons=False)
    eng = make_decider(c)
    by_sym = {}
    for t in r.trades:
        by_sym.setdefault(t.symbol, []).append(t)
    expected = reversed_ = 0
    for s, trades in by_sym.items():
        f = frames[s]
        v = FrameView(f)
        entries = {(t.entry_time, t.side) for t in trades}
        for t in trades:
            if t.side != LONG or not t.exit_reason.startswith("Momentum satıcıya"):
                continue
            i = f.index.get_loc(pd.Timestamp(t.exit_time)) - 1  # decision bar = bar before the exit fill
            row = f.iloc[i]
            gate = (row["regime"] in c.strategy.squeeze.tradeable_regimes and row["vol_state"] != "EXTREME"
                    and row["close"] < row["vwap"] and eng.short_signal(v, i))
            if gate:
                expected += 1
                reversed_ += (t.exit_time, SHORT) in entries
    assert expected > 5 and reversed_ == expected


def test_volatility_multiplier_shrinks_even_when_the_position_cap_binds():
    cfg = RiskConfig(max_position_pct=0.30)
    # 1% risk / 0.75% stop = 133% notional, far above the 30% cap
    full = size_position(cfg, 1000, 1000, 100.0, 0.5, LONG, "NORMAL", [])
    garch = size_position(cfg, 1000, 1000, 100.0, 0.5, LONG, "NORMAL", [], size_mult=0.6)
    high = size_position(cfg, 1000, 1000, 100.0, 0.5, LONG, "HIGH", [])
    assert full.qty * 100.0 == pytest.approx(300.0, rel=1e-6)
    assert garch.qty == pytest.approx(full.qty * 0.6) and high.qty == pytest.approx(full.qty * cfg.high_vol_size_mult)
    assert any("GARCH" in n for n in garch.notes) and garch.risk_amount == pytest.approx(garch.qty * garch.stop_distance)


def test_walk_forward_keeps_positions_across_oos_windows(cfg, universe):
    c = cfg.with_overrides({"account.allow_short": True})
    c.walkforward.is_bars, c.walkforward.oos_bars = 1000, 500
    c.walkforward.grid = {"strategy.squeeze.atr_stop_mult": [1.0, 2.0]}
    res = walk_forward(c, universe, layers=regime_layers(universe, c))
    forced = [t for t in res.oos_trades if t.exit_reason.startswith("Backtest sonu")]
    last_oos_end = res.windows[-1].oos_end
    assert all(pd.Timestamp(t.exit_time) >= pd.Timestamp(last_oos_end) - pd.Timedelta(hours=1) for t in forced)
    assert len(forced) <= len(universe)


def test_open_orders_fill_before_any_intrabar_exit_of_the_same_bar(cfg, universe, monkeypatch):
    # Orders decided at the previous close must be sized with the account as it was then (as live does),
    # never with cash or Kelly history released by another symbol's stop later inside the same bar.
    c = cfg.with_overrides({"account.allow_short": True, "risk.max_drawdown": 0.99, "risk.daily_loss_limit": 1.0})
    frames = prepare_universe(universe, c)
    bt = Backtester(c)
    log = []
    o_open, o_close = bt._open, bt._close

    def spy_open(acct, s, ref, v, i, ts, d, reasons, last):
        log.append((ts, "open", acct.cash, len(acct.trades)))
        return o_open(acct, s, ref, v, i, ts, d, reasons, last)

    def spy_close(acct, s, ref, v, i, ts, reason):
        intrabar = reason.startswith(("Zarar-kes", "Hedef kâr", "İzleyen stop"))
        log.append((ts, "intrabar" if intrabar else "open-exit", acct.cash, len(acct.trades)))
        return o_close(acct, s, ref, v, i, ts, reason)

    bt._open, bt._close = spy_open, spy_close
    bt.run(frames, collect_reasons=False)
    seen_intrabar, mixed = set(), 0
    for ts, kind, _, _ in log:
        if kind == "intrabar":
            seen_intrabar.add(ts)
        elif ts in seen_intrabar:
            mixed += 1
    assert len({ts for ts, *_ in log}) > 50 and mixed == 0
