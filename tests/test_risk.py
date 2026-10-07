import pandas as pd
import pytest

from cappatrader.config import RiskConfig
from cappatrader.execution.account import Account
from cappatrader.models import LONG, SHORT, Position, Trade
from cappatrader.risk.manager import RiskManager
from cappatrader.risk.sizing import kelly_stats, risk_fraction, round_step, size_position
from cappatrader.risk.stops import stop_hit, update_trailing


def _trade(pnl: float, r: float) -> Trade:
    return Trade("BTCUSDT", 1, 1, 100, 100 + pnl, "a", "b", pnl, 0, 0, pnl, r, "x", "TREND_UP")


def test_kelly_formula():
    trades = [_trade(2, 2.0)] * 6 + [_trade(-1, -1.0)] * 4
    ks = kelly_stats(trades, 100)
    assert ks.win_rate == pytest.approx(0.6)
    assert ks.payoff == pytest.approx(2.0)
    assert ks.kelly == pytest.approx(0.6 - 0.4 / 2.0)


def test_risk_fraction_uses_base_until_enough_history():
    cfg = RiskConfig(kelly_min_trades=30)
    frac, note = risk_fraction(cfg, [_trade(1, 1.0)] * 5)
    assert frac == cfg.base_risk_per_trade and "yetersiz" in note


def test_risk_fraction_capped_and_floored():
    cfg = RiskConfig(kelly_min_trades=10, max_risk_per_trade=0.02, min_risk_per_trade=0.0025)
    good = [_trade(3, 3.0)] * 9 + [_trade(-1, -1.0)]
    assert risk_fraction(cfg, good)[0] == 0.02
    bad = [_trade(1, 1.0)] * 2 + [_trade(-1, -1.0)] * 8
    assert risk_fraction(cfg, bad)[0] == 0.0025


def test_higher_volatility_means_smaller_position():
    cfg = RiskConfig(max_position_pct=1.0)
    calm = size_position(cfg, 1000, 1000, 100.0, 1.0, LONG, "NORMAL", [])
    wild = size_position(cfg, 1000, 1000, 100.0, 3.0, LONG, "NORMAL", [])
    assert calm.ok and wild.ok
    assert wild.qty == pytest.approx(calm.qty / 3)
    assert calm.risk_amount == pytest.approx(wild.risk_amount)


def test_high_vol_regime_halves_size():
    cfg = RiskConfig(max_position_pct=1.0)
    normal = size_position(cfg, 1000, 1000, 100.0, 1.0, LONG, "NORMAL", [])
    high = size_position(cfg, 1000, 1000, 100.0, 1.0, LONG, "HIGH", [])
    assert high.qty == pytest.approx(normal.qty * cfg.high_vol_size_mult)


def test_position_cap_min_notional_and_step():
    cfg = RiskConfig(max_position_pct=0.3, min_notional=10)
    r = size_position(cfg, 1000, 1000, 100.0, 0.01, LONG, "NORMAL", [], step_size=0.01)
    assert r.qty * 100 <= 300 + 1e-9
    assert r.qty == round_step(r.qty, 0.01)
    tiny = size_position(cfg, 20, 20, 100.0, 50.0, LONG, "NORMAL", [])
    assert not tiny.ok and "minimum" in tiny.rejected


def test_cost_floor_widens_stop():
    cfg = RiskConfig(max_cost_r=0.2, max_position_pct=1.0)
    r = size_position(cfg, 1000, 1000, 100.0, 0.05, LONG, "NORMAL", [], round_trip_cost_pct=0.003)
    assert r.stop_distance == pytest.approx(100 * 0.003 / 0.2)
    assert any("Maliyet tabanı" in n for n in r.notes)


def _pos(side=LONG, entry=100.0, stop=96.0) -> Position:
    return Position("BTCUSDT", side, 1.0, entry, "t", stop, stop, abs(entry - stop), entry, "TREND_UP", 70, 0.1, 0.0)


def test_trailing_stop_ratchets_and_locks_breakeven():
    cfg = RiskConfig(trail_atr_mult=3.0, trail_activate_r=1.0)
    p = _pos()
    assert not update_trailing(p, 102, 99, 1.0, cfg)  # +0.5R: not active
    assert p.stop_price == 96.0
    update_trailing(p, 104, 101, 5.0, cfg)  # +1R: active; 104 - 3*5 < entry, so stop = breakeven
    assert p.trailing_active and p.stop_price == 100.0
    update_trailing(p, 110, 106, 1.0, cfg)
    assert p.stop_price == 107.0
    update_trailing(p, 108, 105, 1.0, cfg)  # price falls back: stop never loosens
    assert p.stop_price == 107.0


def test_trailing_stop_short():
    cfg = RiskConfig(trail_atr_mult=2.0, trail_activate_r=1.0)
    p = _pos(SHORT, 100.0, 104.0)
    update_trailing(p, 99, 90, 1.0, cfg)
    assert p.stop_price == 92.0


def test_stop_hit_gap_and_touch():
    p = _pos()
    assert stop_hit(p, 95.0, 97.0, 94.0) == 95.0  # gapped below the stop: fill at the open
    assert stop_hit(p, 99.0, 100.0, 95.5) == 96.0
    assert stop_hit(p, 99.0, 100.0, 97.0) is None


def test_kill_switch_and_daily_limit():
    rm = RiskManager(RiskConfig(max_drawdown=0.15, daily_loss_limit=0.04))
    acct = Account(1000)
    t0 = pd.Timestamp("2025-01-01 00:00", tz="UTC")
    assert rm.update_guards(acct, 1000, t0) == []
    events = rm.update_guards(acct, 955, t0 + pd.Timedelta(hours=1))
    assert events[0][0] == "DAILY_LIMIT"
    ok, why = rm.entry_allowed(acct, "BTCUSDT", t0 + pd.Timedelta(hours=2))
    assert not ok and "Günlük" in why
    assert rm.entry_allowed(acct, "BTCUSDT", t0 + pd.Timedelta(days=1, hours=1))[0]
    events = rm.update_guards(acct, 840, t0 + pd.Timedelta(days=1, hours=2))
    assert any(e[0] == "KILL_SWITCH" for e in events)
    assert not rm.entry_allowed(acct, "BTCUSDT", t0 + pd.Timedelta(days=3))[0]


def test_max_open_positions():
    rm = RiskManager(RiskConfig(max_open_positions=1))
    acct = Account(1000)
    acct.positions["ETHUSDT"] = _pos()
    ok, why = rm.entry_allowed(acct, "BTCUSDT", pd.Timestamp("2025-01-01", tz="UTC"))
    assert not ok and "Maksimum" in why
