"""Live engine wired to a fake exchange: no network needed."""

import asyncio

import pandas as pd
import pytest

from cappatrader.config import Secrets
from cappatrader.data.cleaning import KLINE_COLUMNS
from cappatrader.live.runner import LiveEngine
from cappatrader.live.state import StateStore
from cappatrader.models import Decision
from cappatrader.notify.telegram import TelegramNotifier
from cappatrader.strategy.pipeline import prepare_universe


class FakeREST:
    def __init__(self, universe, restrictions=None):
        self.u = universe
        self.restrictions = restrictions or {"enableWithdrawals": False, "ipRestrict": True}

    async def exchange_filters(self, symbols):
        return {s: {"step_size": 1e-5, "min_qty": 0, "tick_size": 0.01, "min_notional": 5, "status": "TRADING"}
                for s in symbols}

    async def recent_klines(self, symbol, tf, n):
        return self.u[symbol].iloc[-n - 1:-1]  # keep the final bar for the stream

    async def klines_range(self, *a, **k):
        return pd.DataFrame(columns=KLINE_COLUMNS)

    async def api_restrictions(self):
        return self.restrictions

    async def ticker_price(self, symbol):
        return float(self.u[symbol]["close"].iloc[-1])

    async def funding_rate(self, symbol):
        return None

    async def close(self):
        pass


class CaptureNotifier(TelegramNotifier):
    def __init__(self):
        super().__init__("", "", enabled=False)
        self.messages: list[str] = []

    def notify(self, text: str) -> None:
        self.messages.append(text)


@pytest.fixture
def live_cfg(cfg):
    c = cfg.with_overrides({"data.tick_max_deviation": 0.5})
    c.symbols = ["BTCUSDT", "ETHUSDT"]
    c.pairs = [["ETHUSDT", "BTCUSDT"]]
    return c


def _engine(cfg, universe, **rest_kw):
    return LiveEngine(cfg, rest=FakeREST(universe, **rest_kw), notifier=CaptureNotifier(), store=StateStore(":memory:"))


async def _book(eng, sym, px):
    await eng.on_stream(f"{sym.lower()}@depth20@100ms",
                        {"lastUpdateId": 1, "bids": [[str(px * 0.9999), "50"]], "asks": [[str(px * 1.0001), "50"]]})


def test_entry_then_tick_stop_exit_sends_telegram(live_cfg, universe):
    async def scenario():
        eng = _engine(live_cfg, universe)
        await eng.bootstrap()
        sym = "BTCUSDT"
        px = eng.last_price[sym]
        await _book(eng, sym, px)
        row = prepare_universe({sym: eng.bars[sym]}, live_cfg)[sym].iloc[-1]
        d = Decision(sym, row.name, "ENTER_LONG", 0.5, 75.0, row["regime"], "NORMAL", px, float(row["atr"]))
        await eng.open_position(sym, d, row, {"orderflow": "Emir defteri test notu"})
        pos = eng.account.positions[sym]
        assert pos.entry_price == pytest.approx(px * 1.0001)  # filled at the ask of the live book
        assert eng.account.cash < live_cfg.account.starting_balance
        stop = pos.stop_price
        await _book(eng, sym, stop * 0.999)
        await eng.on_stream(f"{sym.lower()}@aggTrade", {"p": str(stop * 0.999), "q": "0.1", "m": True})
        await asyncio.gather(*eng.tasks)
        return eng

    eng = asyncio.run(scenario())
    assert not eng.account.positions and len(eng.account.trades) == 1
    t = eng.account.trades[0]
    assert t.pnl < 0 and "Zarar-kes" in t.exit_reason
    entry, exit_ = eng.notifier.messages
    assert "ALIŞ" in entry and "BTC/USDT" in entry and "Neden alındı" in entry and "Güncel bakiye" in entry
    assert "SATIŞ" in exit_ and "P&amp;L" in exit_ and "Zarar-kes" in exit_ and "Güncel bakiye" in exit_
    assert eng.store.trades()[0].pnl == pytest.approx(t.pnl)


def test_closed_kline_runs_full_decision_cycle(live_cfg, universe):
    async def scenario():
        eng = _engine(live_cfg, universe)
        await eng.bootstrap()
        before = len(eng.bars["BTCUSDT"])
        last = universe["BTCUSDT"].iloc[-1]
        k = {"t": int(universe["BTCUSDT"].index[-1].value // 1_000_000), "o": last.open, "h": last.high,
             "l": last.low, "c": last.close, "v": last.volume, "q": last.quote_volume, "V": 0.0, "x": True}
        await eng.on_stream("btcusdt@kline_1h", {"k": k})
        await asyncio.gather(*eng.tasks)
        return eng, before

    eng, before = asyncio.run(scenario())
    assert eng.bars["BTCUSDT"].index[-1] == universe["BTCUSDT"].index[-1]
    assert len(eng.bars["BTCUSDT"]) >= before
    assert eng.hmm["BTCUSDT"].fitted is not None and eng.garch["BTCUSDT"].params is not None
    assert not eng.use_ml  # the ML direction model only votes in the ensemble strategy


def test_ensemble_strategy_trains_ml_live(live_cfg, universe):
    cfg = live_cfg.with_overrides({"strategy.name": "ensemble"})

    async def scenario():
        eng = _engine(cfg, universe)
        await eng.bootstrap()
        return eng

    eng = asyncio.run(scenario())
    assert eng.use_ml and eng.ml["BTCUSDT"].model is not None


def test_kill_switch_closes_everything(live_cfg, universe):
    async def scenario():
        eng = _engine(live_cfg, universe)
        await eng.bootstrap()
        sym = "ETHUSDT"
        px = eng.last_price[sym]
        await _book(eng, sym, px)
        row = prepare_universe({sym: eng.bars[sym]}, live_cfg)[sym].iloc[-1]
        d = Decision(sym, row.name, "ENTER_LONG", 0.5, 75.0, row["regime"], "NORMAL", px, float(row["atr"]))
        await eng.open_position(sym, d, row, {})
        eng.account.peak_equity = 2000.0  # pretend we were far higher
        await eng.check_guards()
        await asyncio.gather(*eng.tasks)
        return eng

    eng = asyncio.run(scenario())
    assert eng.account.halted and not eng.account.positions
    assert any("KILL-SWITCH" in m for m in eng.notifier.messages)


def test_api_key_with_withdrawals_is_flagged(live_cfg, universe):
    live_cfg.secrets = Secrets("key", "secret", "", "")

    async def scenario():
        eng = _engine(live_cfg, universe, restrictions={"enableWithdrawals": True, "ipRestrict": False})
        await eng.bootstrap()
        return eng

    eng = asyncio.run(scenario())
    assert any("PARA ÇEKME" in w for w in eng.warnings)
    assert any("IP" in w for w in eng.warnings)


def test_state_survives_restart(tmp_path, live_cfg, universe):
    db = str(tmp_path / "state.sqlite")

    async def open_one():
        eng = LiveEngine(live_cfg, rest=FakeREST(universe), notifier=CaptureNotifier(), store=StateStore(db))
        await eng.bootstrap()
        sym = "BTCUSDT"
        px = eng.last_price[sym]
        await _book(eng, sym, px)
        row = prepare_universe({sym: eng.bars[sym]}, live_cfg)[sym].iloc[-1]
        d = Decision(sym, row.name, "ENTER_LONG", 0.5, 75.0, row["regime"], "NORMAL", px, float(row["atr"]))
        await eng.open_position(sym, d, row, {})
        eng.store.close()

    asyncio.run(open_one())
    acct = StateStore(db).load_account(1000)
    assert "BTCUSDT" in acct.positions and acct.cash < 1000


def test_protection_stays_on_while_ws_down_and_rest_prices_do_not_revive(live_cfg, universe):
    async def scenario():
        eng = _engine(live_cfg, universe)
        await eng.bootstrap()
        await eng.on_ws_state(True)
        await eng.on_ws_state(False)  # stream dropped, data still "fresh"
        assert eng.protection
        before = dict(eng.last_seen)
        await eng.on_price("BTCUSDT", eng.last_price["BTCUSDT"], from_stream=False)
        assert eng.last_seen == before  # REST fallback prices never refresh liveness
        task = asyncio.create_task(eng.watchdog())
        await asyncio.sleep(5.3)  # one watchdog cycle
        task.cancel()
        return eng

    eng = asyncio.run(scenario())
    assert eng.protection  # not cleared while the WebSocket is still down
    assert not any("normale döndü" in m for m in eng.notifier.messages)


def test_stop_during_bootstrap_exits_cleanly(live_cfg, universe, tmp_path):
    live_cfg.runtime.state_db = str(tmp_path / "s.sqlite")

    class SlowREST(FakeREST):
        async def recent_klines(self, symbol, tf, n):
            await asyncio.sleep(30)
            return await super().recent_klines(symbol, tf, n)

    async def scenario():
        eng = LiveEngine(live_cfg, rest=SlowREST(universe), notifier=CaptureNotifier(),
                         store=StateStore(live_cfg.runtime.state_db))
        task = asyncio.create_task(eng.run())
        await asyncio.sleep(0.5)
        eng.stop_event.set()
        await asyncio.wait_for(task, timeout=5)
        return eng

    eng = asyncio.run(scenario())
    assert not any("başlatıldı" in m for m in eng.notifier.messages)
    from cappatrader.live.lock import running_pid

    assert running_pid(live_cfg.state_db) is None  # the lock was released


def test_removed_symbol_with_open_position_is_still_managed(live_cfg, universe, tmp_path):
    db = str(tmp_path / "s.sqlite")
    store = StateStore(db)
    acct = store.load_account(1000)
    from cappatrader.models import Position

    acct.positions["SOLUSDT"] = Position("SOLUSDT", 1, 1.0, 100.0, "2025-01-01T00:00:00+00:00", 90, 90, 10, 100,
                                         "TREND_UP", 70, 0.1, 0.0)
    store.save_account(acct)
    store.close()

    async def scenario():
        eng = LiveEngine(live_cfg, rest=FakeREST(universe), notifier=CaptureNotifier(), store=StateStore(db))
        await eng.bootstrap()
        return eng

    eng = asyncio.run(scenario())
    assert "SOLUSDT" in eng.symbols and "SOLUSDT" in eng.manage_only and "SOLUSDT" in eng.bars
    assert any("yalnızca stop/hedef takibi" in w for w in eng.warnings)


def test_streamed_bars_drive_the_same_decisions_as_the_backtest(cfg, universe, monkeypatch):
    """The live decision loop (on_kline -> on_bar_close: online HMM/GARCH refits, prepare_universe on the
    rolling buffer, decide, entries) must reproduce the backtest's view of every bar (CLAUDE.md parity)."""
    import math

    import cappatrader.live.runner as runner
    from cappatrader.strategy.decision import FrameView
    from cappatrader.strategy.factory import decide, make_decider
    from cappatrader.strategy.pipeline import prepare_universe

    c = cfg.with_overrides({"account.allow_short": True, "data.tick_max_deviation": 0.5})
    c.symbols, c.pairs = ["BTCUSDT"], []
    df, n_feed = universe["BTCUSDT"], 300

    class StreamREST(FakeREST):
        async def recent_klines(self, symbol, tf, n):
            return self.u[symbol].iloc[-n_feed - n:-n_feed]

    seen, opened = {}, []
    orig_decide = runner.decide

    def spy(decider, cfg_, sym, v, i, pos, bse):
        d = orig_decide(decider, cfg_, sym, v, i, pos, bse)
        seen[v.index[i]] = (d.regime, d.vol_state, d.size_mult, d.atr, decider._entry_action(v, i, d))
        return d

    monkeypatch.setattr(runner, "decide", spy)

    def kline(ts, row):
        return {"t": int(ts.value // 1_000_000), "o": row.open, "h": row.high, "l": row.low, "c": row.close,
                "v": row.volume, "q": row.quote_volume, "V": 0.0, "x": True}

    async def scenario():
        eng = LiveEngine(c, rest=StreamREST(universe), notifier=CaptureNotifier(), store=StateStore(":memory:"))
        await eng.bootstrap()
        orig_open = eng.open_position

        async def spy_open(sym, d, row, notes):
            opened.append(d)
            return await orig_open(sym, d, row, notes)

        eng.open_position = spy_open
        for j in range(n_feed, 0, -1):
            await eng.on_stream("btcusdt@kline_1h", {"k": kline(df.index[-j], df.iloc[-j])})
            if eng.tasks:
                await asyncio.gather(*list(eng.tasks))
        return eng

    asyncio.run(scenario())
    frame = prepare_universe({"BTCUSDT": df}, c)["BTCUSDT"]
    v, dec = FrameView(frame), make_decider(c)
    assert len(seen) == n_feed
    for ts, (regime, vol, mult, atr, entry) in seen.items():
        i = frame.index.get_loc(ts)
        d = decide(dec, c, "BTCUSDT", v, i, None, None)
        assert (d.regime, d.vol_state, dec._entry_action(v, i, d)) == (regime, vol, entry), ts
        assert math.isclose(d.atr, atr, rel_tol=1e-9), ts
        assert (math.isnan(d.size_mult) and math.isnan(mult)) or math.isclose(d.size_mult, mult, rel_tol=1e-9), ts
    assert sum(e is not None for *_, e in seen.values()) > 0 and opened  # the loop really trades


def test_flash_move_halts_new_entries_for_the_cooldown(live_cfg, universe):
    eng = LiveEngine(live_cfg, rest=FakeREST(universe), notifier=CaptureNotifier(), store=StateStore(":memory:"))
    sym = "BTCUSDT"
    eng._flash_check(sym, 100.0, 1000.0)
    eng._flash_check(sym, 102.0, 1010.0)
    assert sym not in eng.account.symbol_halts  # 2% is a normal move
    eng._flash_check(sym, 100.0 * (1 + live_cfg.risk.flash_move_pct + 0.01), 1020.0)
    assert sym in eng.account.symbol_halts and any("flash" in m.lower() for m in eng.notifier.messages)
    assert not eng._entry_ok(sym) and eng._entry_ok("ETHUSDT")
    eng.account.symbol_halts[sym] = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=1)).isoformat()
    assert eng._entry_ok(sym)  # cooldown over
    halted_until = eng.account.symbol_halts[sym]
    eng._flash_check(sym, 100.0, 1020.0 + live_cfg.risk.flash_window_s + 5)  # the spike left the window
    assert eng.account.symbol_halts[sym] == halted_until and eng._entry_ok(sym)
