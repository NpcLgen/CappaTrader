"""Live (mode: live) engine logic against a fake futures broker: orders, protection, reconciliation."""

import asyncio
import os
from dataclasses import dataclass

import pandas as pd
import pytest

from cappatrader.config import Secrets
from cappatrader.live.lock import RunLock, running_pid
from cappatrader.live.runner import LiveEngine
from cappatrader.live.state import StateStore
from cappatrader.models import SHORT, Decision, Fill
from cappatrader.strategy.pipeline import prepare_universe
from tests.test_live import CaptureNotifier, FakeREST


@dataclass
class Bal:
    wallet: float
    available: float
    unrealized: float = 0.0

    @property
    def equity(self):
        return self.wallet + self.unrealized


@dataclass
class XPos:
    symbol: str
    side: int
    qty: float
    entry_price: float
    unrealized: float = 0.0
    leverage: int = 2


class FakeFutures:
    """Duck-typed stand-in for BinanceFuturesBroker recording every call."""

    def __init__(self, positions=None, wallet=5000.0, fail_protection=False, already_closed=False):
        self.calls = []
        self.pos = dict(positions or {})
        self.wallet = wallet
        self.unavailable = []
        self.warnings = []
        self.tp_warning = False
        self.fail_close = False
        self.fail_cancel = False
        self.fail_protection = fail_protection
        self.already_closed = already_closed

    async def connect(self):
        self.calls.append(("connect",))

    async def balance(self):
        return Bal(self.wallet, self.wallet * 0.9)

    async def positions(self, symbols=None):
        self.calls.append(("positions", tuple(symbols or ())))
        return {k: v for k, v in self.pos.items() if symbols is None or k in symbols}

    async def market_order(self, symbol, side, qty, ref_price, reduce_only=False):
        self.calls.append(("market", symbol, side, qty, reduce_only))
        self.pos[symbol] = XPos(symbol, side, qty, ref_price * (1 + side * 0.0002))
        return Fill(symbol, side, qty, ref_price * (1 + side * 0.0002), ref_price, qty * ref_price * 0.0005, 2.0, 50.0,
                    pd.Timestamp.now(tz="UTC"))

    async def place_protection(self, symbol, position_side, qty, stop_price, take_profit):
        self.calls.append(("protect", symbol, position_side, qty, stop_price, take_profit))
        if self.fail_protection:
            raise RuntimeError("algo order rejected")
        if self.tp_warning:
            self.warnings.append(f"{symbol}: borsa kâr al emri konamadı, yalnızca stop aktif")
        return ["s1", "t1"]

    async def cancel_protection(self, symbol, order_ids=None):
        self.calls.append(("cancel", symbol) if order_ids is None else ("cancel_ids", symbol, tuple(order_ids)))
        if self.fail_cancel:
            raise RuntimeError("binanceusdm RequestTimeout")

    async def close_position(self, symbol, position_side, qty, ref_price):
        self.calls.append(("close", symbol, position_side, qty))
        if self.fail_close:
            raise RuntimeError("binanceusdm RequestTimeout")
        if self.already_closed or symbol not in self.pos:
            return None
        self.pos.pop(symbol)
        return Fill(symbol, -position_side, qty, ref_price, ref_price, qty * ref_price * 0.0005, 0.0, 40.0,
                    pd.Timestamp.now(tz="UTC"))

    async def last_close_fill(self, symbol, since_ms, position_side):
        self.calls.append(("last_close_fill", symbol))
        return Fill(symbol, -position_side, 0.01, 101.0, 101.0, 0.05, 0.0, 0.0, pd.Timestamp.now(tz="UTC"))

    async def close(self):
        self.calls.append(("close_session",))


@pytest.fixture
def live_mode_cfg(cfg):
    c = cfg.with_overrides({"mode": "live", "data.tick_max_deviation": 0.5, "account.allow_short": True})
    c.symbols = ["BTCUSDT", "ETHUSDT"]
    c.pairs = [["ETHUSDT", "BTCUSDT"]]
    c.secrets = Secrets("demo-key", "demo-secret", "", "")
    return c


def _engine(cfg, universe, fut):
    return LiveEngine(cfg, rest=FakeREST(universe), notifier=CaptureNotifier(), store=StateStore(":memory:"),
                      futures=fut)


def _decision(eng, cfg, sym, action="ENTER_LONG"):
    row = prepare_universe({sym: eng.bars[sym]}, cfg)[sym].iloc[-1]
    px = eng.last_price[sym]
    d = Decision(sym, row.name, action, 0.5, 75.0, "TREND_UP", "NORMAL", px, float(row["atr"]), style="SQZ")
    return d, row


async def _book(eng, sym, px):
    await eng.on_stream(f"{sym.lower()}@depth20@100ms",
                        {"e": "depthUpdate", "s": sym, "b": [[str(px * 0.9999), "50"]], "a": [[str(px * 1.0001), "50"]]})


def test_live_mode_uses_futures_market_and_separate_state(live_mode_cfg):
    assert live_mode_cfg.data_market == "futures"
    assert live_mode_cfg.state_db.endswith("_live_demo.sqlite")
    assert "DEMO" in live_mode_cfg.mode_label


def test_startup_adopts_orphan_and_protects(live_mode_cfg, universe):
    orphan = XPos("ETHUSDT", SHORT, 0.5, float(universe["ETHUSDT"]["close"].iloc[-2]))
    fut = FakeFutures(positions={"ETHUSDT": orphan})

    async def scenario():
        eng = _engine(live_mode_cfg, universe, fut)
        await eng.bootstrap()
        return eng

    eng = asyncio.run(scenario())
    pos = eng.account.positions["ETHUSDT"]
    assert pos.side == SHORT and pos.qty == 0.5 and pos.stop_price > pos.entry_price > pos.take_profit
    assert ("protect", "ETHUSDT", SHORT, 0.5, pos.stop_price, pos.take_profit) in fut.calls
    assert eng.wallet == 5000.0 and eng.equity() == pytest.approx(5000.0 + pos.unrealized(eng.last_price["ETHUSDT"]))
    assert any("Sahipsiz pozisyon" in m for m in eng.notifier.messages)
    assert eng.account.starting_balance == pytest.approx(eng.equity(), rel=1e-6)  # fresh live account seeded


def test_live_entry_places_protection_and_exit_cancels(live_mode_cfg, universe):
    fut = FakeFutures()

    async def scenario():
        eng = _engine(live_mode_cfg, universe, fut)
        await eng.bootstrap()
        sym = "BTCUSDT"
        await _book(eng, sym, eng.last_price[sym])
        d, row = _decision(eng, live_mode_cfg, sym)
        await eng.open_position(sym, d, row, {})
        pos = eng.account.positions[sym]
        assert pos.take_profit == pytest.approx(pos.entry_price + 1.1 * pos.risk_per_unit)
        await eng.close_position(sym, "Momentum satıcıya döndü")
        return eng

    eng = asyncio.run(scenario())
    kinds = [c[0] for c in fut.calls]
    first_market = kinds.index("market")
    assert kinds.index("cancel") < first_market  # leftover orders of an earlier trade are cleared first
    assert first_market < kinds.index("protect") < kinds.index("close") < len(kinds) - 1 - kinds[::-1].index("cancel")
    assert len(eng.account.trades) == 1 and not eng.account.positions
    entry, exit_ = eng.notifier.messages[-2:]
    assert "CANLI · DEMO" in entry and "ALIŞ" in entry and "SATIŞ" in exit_ and "Miktar" in exit_


def test_protection_failure_closes_position(live_mode_cfg, universe):
    fut = FakeFutures(fail_protection=True)

    async def scenario():
        eng = _engine(live_mode_cfg, universe, fut)
        await eng.bootstrap()
        sym = "BTCUSDT"
        await _book(eng, sym, eng.last_price[sym])
        d, row = _decision(eng, live_mode_cfg, sym)
        await eng.open_position(sym, d, row, {})
        return eng

    eng = asyncio.run(scenario())
    assert not eng.account.positions and len(eng.account.trades) == 1
    assert "güvenlik kapanışı" in eng.account.trades[0].exit_reason
    assert any("Koruma emri yerleştirilemedi" in m for m in eng.notifier.messages)


def test_exchange_side_close_is_reconciled(live_mode_cfg, universe):
    fut = FakeFutures()

    async def scenario():
        eng = _engine(live_mode_cfg, universe, fut)
        await eng.bootstrap()
        sym = "BTCUSDT"
        await _book(eng, sym, eng.last_price[sym])
        d, row = _decision(eng, live_mode_cfg, sym)
        await eng.open_position(sym, d, row, {})
        fut.pos.pop(sym)  # the exchange-side stop fired
        await eng.reconcile_positions()
        return eng

    eng = asyncio.run(scenario())
    assert not eng.account.positions and len(eng.account.trades) == 1
    assert "borsada kapandı" in eng.account.trades[0].exit_reason.lower()
    assert ("last_close_fill", "BTCUSDT") in fut.calls and ("cancel", "BTCUSDT") in fut.calls


def test_close_when_exchange_already_closed(live_mode_cfg, universe):
    fut = FakeFutures()

    async def scenario():
        eng = _engine(live_mode_cfg, universe, fut)
        await eng.bootstrap()
        sym = "BTCUSDT"
        await _book(eng, sym, eng.last_price[sym])
        d, row = _decision(eng, live_mode_cfg, sym)
        await eng.open_position(sym, d, row, {})
        fut.already_closed = True
        await eng.close_position(sym, "Zarar-kes (stop-loss) tetiklendi")
        return eng

    eng = asyncio.run(scenario())
    assert len(eng.account.trades) == 1 and "zaten kapanmıştı" in eng.account.trades[0].exit_reason


def test_kill_switch_during_entry_flattens(live_mode_cfg, universe):
    fut = FakeFutures()

    async def scenario():
        eng = _engine(live_mode_cfg, universe, fut)
        await eng.bootstrap()
        sym = "BTCUSDT"
        await _book(eng, sym, eng.last_price[sym])
        d, row = _decision(eng, live_mode_cfg, sym)
        orig = fut.market_order

        async def market_then_halt(*a, **k):
            f = await orig(*a, **k)
            eng.account.halted = True  # kill-switch fires while the order is in flight
            return f

        fut.market_order = market_then_halt
        await eng.open_position(sym, d, row, {})
        return eng

    eng = asyncio.run(scenario())
    assert not eng.account.positions and "Kill-switch" in eng.account.trades[0].exit_reason


def test_symbols_unavailable_on_futures_are_dropped(live_mode_cfg, universe):
    fut = FakeFutures()
    fut.unavailable = ["ETHUSDT"]

    async def scenario():
        eng = _engine(live_mode_cfg, universe, fut)
        await eng.bootstrap()
        return eng

    eng = asyncio.run(scenario())
    assert eng.symbols == ["BTCUSDT"] and any("ETH/USDT" in w for w in eng.warnings)


def _hold_lock_in_child(db: str, pid_text: str = "1"):
    """Another process holding the engine lock; it writes `pid_text` like PID 1 in its own container."""
    import subprocess
    import sys
    import time

    code = ("import fcntl, os, sys, time; fd = os.open(sys.argv[1] + '.lock', os.O_RDWR | os.O_CREAT, 0o644); "
            "fcntl.flock(fd, fcntl.LOCK_EX); os.ftruncate(fd, 0); os.write(fd, sys.argv[2].encode()); "
            "print('locked', flush=True); time.sleep(60)")
    child = subprocess.Popen([sys.executable, "-c", code, db, pid_text], stdout=subprocess.PIPE, text=True)
    assert child.stdout.readline().strip() == "locked"
    time.sleep(0.05)
    return child


def test_run_lock_blocks_second_engine(tmp_path):
    db = str(tmp_path / "s.sqlite")
    child = _hold_lock_in_child(db, "1")  # PID 1 also in our view: PIDs alone cannot tell containers apart
    try:
        assert running_pid(db) == 1
        with pytest.raises(RuntimeError, match="zaten çalışan bir bot"):
            RunLock(db).acquire()
    finally:
        child.kill()
        child.wait()
    assert running_pid(db) is None  # the kernel released the lock with the process: a leftover file is not a lock
    lock = RunLock(db)
    lock.acquire()
    with pytest.raises(RuntimeError):
        RunLock(db).acquire()  # a second engine in the same process / container is refused too
    assert running_pid(db) == os.getpid()
    lock.release()
    assert running_pid(db) is None


def test_reset_and_resume_refuse_while_running(tmp_path, monkeypatch, capsys):
    from cappatrader import cli

    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(f"runtime:\n  state_db: {tmp_path / 'st.sqlite'}\n  log_dir: {tmp_path / 'logs'}\n")
    child = _hold_lock_in_child(str(tmp_path / "st.sqlite"), str(os.getppid()))
    try:
        assert cli.main(["--config", str(cfg_file), "reset", "--yes"]) == 1
        assert cli.main(["--config", str(cfg_file), "resume"]) == 1
    finally:
        child.kill()
        child.wait()
    assert "çalışıyor" in capsys.readouterr().out
    assert cli.main(["--config", str(cfg_file), "resume"]) == 0  # lock released with the process


def test_paper_balance_seeded_from_binance(cfg, universe, monkeypatch):
    import cappatrader.execution.binance_futures as bf

    async def fake_balance(c):
        return 2345.67

    monkeypatch.setattr(bf, "fetch_binance_balance", fake_balance)
    c = cfg.with_overrides({"account.balance_source": "binance"})
    c.symbols, c.pairs = ["BTCUSDT"], []
    c.secrets = Secrets("k", "s", "", "")

    async def scenario():
        eng = LiveEngine(c, rest=FakeREST(universe, restrictions={"enableWithdrawals": False, "ipRestrict": True}),
                         notifier=CaptureNotifier(), store=StateStore(":memory:"))
        await eng.bootstrap()
        return eng

    eng = asyncio.run(scenario())
    assert eng.account.starting_balance == pytest.approx(2345.67) and eng.account.cash == pytest.approx(2345.67)
    assert "2,345.67" in eng.binance_balance_note


def test_balance_command_reads_binance(tmp_path, monkeypatch, capsys):
    import cappatrader.execution.binance_futures as bf
    from cappatrader import cli

    class FakeReader:
        def __init__(self, cfg, *a, **k):
            self.env = "mainnet"

        async def balance(self):
            return bf.FuturesBalance(wallet=1200.0, available=1100.0, unrealized=-5.0, equity=1195.0)

        async def spot_usdt_balance(self):
            return 42.5

        async def close(self):
            pass

    monkeypatch.setattr(bf, "BinanceAccountReader", FakeReader)
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_SECRET_KEY", "s")
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(f"runtime:\n  state_db: {tmp_path / 'st.sqlite'}\n  log_dir: {tmp_path / 'logs'}\n")
    assert cli.main(["--config", str(cfg_file), "balance"]) == 0
    out = capsys.readouterr().out
    assert "1,195.00 USDT" in out and "42.50 USDT" in out and "-5.00 USDT" in out


def test_reconcile_tracks_held_symbols_and_reprotects(live_mode_cfg, universe):
    fut = FakeFutures()

    async def scenario():
        eng = _engine(live_mode_cfg, universe, fut)
        await eng.bootstrap()
        sym = "BTCUSDT"
        await _book(eng, sym, eng.last_price[sym])
        d, row = _decision(eng, live_mode_cfg, sym)
        await eng.open_position(sym, d, row, {})
        held = [c for c in fut.calls if c[0] == "positions"][-1][1]
        assert set(held) >= set(eng.symbols)
        # the exchange position shrank (e.g. partial manual close): local qty follows and the stop is re-placed
        fut.pos[sym].qty = eng.account.positions[sym].qty / 2
        fut.calls.clear()
        await eng.reconcile_positions()
        pos = eng.account.positions[sym]
        assert pos.qty == fut.pos[sym].qty
        # the new stop/target is placed first, then the old orders are cancelled by id: never unprotected
        assert [c[0] for c in fut.calls[1:3]] == ["protect", "cancel_ids"] and fut.calls[1][3] == pos.qty
        assert fut.calls[2][2] == ("s1", "t1")
        return eng

    asyncio.run(scenario())


def test_failed_protection_on_restart_is_retried_each_sync(live_mode_cfg, universe):
    orphan = XPos("ETHUSDT", SHORT, 0.5, float(universe["ETHUSDT"]["close"].iloc[-2]))
    fut = FakeFutures(positions={"ETHUSDT": orphan}, fail_protection=True)

    async def scenario():
        eng = _engine(live_mode_cfg, universe, fut)
        await eng.bootstrap()
        assert "ETHUSDT" in eng.unprotected and "ETHUSDT" in eng.account.positions
        assert sum("Koruma emri yerleştirilemedi" in m for m in eng.notifier.messages) == 1
        assert sum(c[0] == "protect" for c in fut.calls) == 1  # adopted once, not re-protected again at startup
        await eng.reconcile_positions()  # still failing: no alert spam
        assert sum("Koruma emri yerleştirilemedi" in m for m in eng.notifier.messages) == 1
        fut.fail_protection = False
        await eng.reconcile_positions()
        assert not eng.unprotected
        return eng

    asyncio.run(scenario())


def test_broker_warnings_after_connect_reach_telegram(live_mode_cfg, universe):
    fut = FakeFutures()
    fut.tp_warning = True

    async def scenario():
        eng = _engine(live_mode_cfg, universe, fut)
        await eng.bootstrap()
        sym = "BTCUSDT"
        await _book(eng, sym, eng.last_price[sym])
        d, row = _decision(eng, live_mode_cfg, sym)
        await eng.open_position(sym, d, row, {})
        return eng

    eng = asyncio.run(scenario())
    assert any("kâr al emri konamadı" in m for m in eng.notifier.messages) and not eng.futures.warnings


def test_live_reverse_closes_then_opens_the_opposite_side(live_mode_cfg, universe):
    import copy

    fut = FakeFutures()

    async def scenario():
        eng = _engine(live_mode_cfg, universe, fut)
        await eng.bootstrap()
        sym = "BTCUSDT"
        await _book(eng, sym, eng.last_price[sym])
        d, row = _decision(eng, live_mode_cfg, sym)
        await eng.open_position(sym, d, row, {})
        exit_d = copy.copy(d)
        exit_d.action, exit_d.exit_reason = "EXIT", "Momentum satıcıya döndü — hedefe ulaşmadan erken çıkış"
        exit_d.reverse = copy.copy(d)
        exit_d.reverse.action = "ENTER_SHORT"
        fut.calls.clear()
        await eng._exit_then_enter(sym, exit_d, row, {})
        return eng

    eng = asyncio.run(scenario())
    kinds = [c[0] for c in fut.calls]
    assert kinds.index("close") < kinds.index("market") < kinds.index("protect")
    pos = eng.account.positions["BTCUSDT"]
    assert pos.side == SHORT and pos.stop_price > pos.entry_price > pos.take_profit
    assert len(eng.account.trades) == 1 and eng.account.trades[0].exit_reason.startswith("Momentum satıcıya")


def test_live_ledger_reseeds_from_binance_after_reset_or_failed_start(live_mode_cfg, universe):
    store = StateStore(":memory:")
    store.reset(1000.0)  # e.g. `reset --yes`, or a first start that failed before connecting
    fut = FakeFutures(wallet=400.0)

    async def scenario():
        eng = LiveEngine(live_mode_cfg, rest=FakeREST(universe), notifier=CaptureNotifier(), store=store, futures=fut)
        await eng.bootstrap()
        await eng.check_guards()
        return eng

    eng = asyncio.run(scenario())
    a = eng.account
    assert a.starting_balance == pytest.approx(400.0) and a.peak_equity == pytest.approx(400.0)
    assert not a.halted


def test_live_resume_reseeds_peak_from_binance_equity(live_mode_cfg, universe, tmp_path, monkeypatch):
    from cappatrader import cli

    live_mode_cfg.runtime.state_db = str(tmp_path / "s.sqlite")
    db = live_mode_cfg.state_db  # the live/demo database the CLI uses too
    fut = FakeFutures(wallet=1000.0)

    async def run_once():
        eng = LiveEngine(live_mode_cfg, rest=FakeREST(universe), notifier=CaptureNotifier(), store=StateStore(db),
                         futures=fut)
        await eng.bootstrap()
        await _book(eng, "BTCUSDT", eng.last_price["BTCUSDT"])
        if "BTCUSDT" not in eng.account.positions:
            d, row = _decision(eng, live_mode_cfg, "BTCUSDT")
            await eng.open_position("BTCUSDT", d, row, {})
        await eng.check_guards()
        eng.store.save_account(eng.account)
        return eng

    first = asyncio.run(run_once())
    first.account.halted, first.account.halt_reason = True, "test"
    first.store.save_account(first.account)
    first.store.close()
    monkeypatch.setattr(cli, "_engine_running", lambda c: False)

    class Args:
        pass

    assert cli.cmd_resume(Args(), live_mode_cfg) == 0
    second = asyncio.run(run_once())
    a = second.account
    # the old code planted cash (free margin) + collateral as the peak and fired a spurious kill-switch
    assert not a.halted and a.peak_equity == pytest.approx(second.equity(), rel=1e-9)
    assert not second.store.get_kv("reseed_peak")


def test_mainnet_refuses_a_key_with_withdrawals_enabled(live_mode_cfg, universe):
    from cappatrader.config import MAINNET_CONFIRM_PHRASE

    c = live_mode_cfg.with_overrides({"live.env": "mainnet", "live.mainnet_confirm": MAINNET_CONFIRM_PHRASE})
    c.symbols, c.pairs, c.secrets = live_mode_cfg.symbols, live_mode_cfg.pairs, live_mode_cfg.secrets
    fut = FakeFutures()

    async def scenario(restrictions):
        eng = LiveEngine(c, rest=FakeREST(universe, restrictions=restrictions), notifier=CaptureNotifier(),
                         store=StateStore(":memory:"), futures=fut)
        await eng.bootstrap()
        return eng

    with pytest.raises(RuntimeError, match="PARA ÇEKME"):
        asyncio.run(scenario({"enableWithdrawals": True, "ipRestrict": True}))
    assert ("connect",) not in fut.calls  # refused before touching the futures account
    eng = asyncio.run(scenario({"enableWithdrawals": False, "ipRestrict": False}))
    assert any("IP kısıtlaması yok" in w for w in eng.warnings)


def test_research_downloads_the_traded_market(live_mode_cfg, monkeypatch, tmp_path):
    import cappatrader.data.history as history

    seen = []

    class Rest:
        def __init__(self, ex, secrets=None, session=None, market="spot"):
            seen.append(market)

        async def klines_range(self, *a, **k):
            return pd.DataFrame()

        async def close(self):
            pass

    monkeypatch.setattr(history, "BinanceREST", Rest)
    c = live_mode_cfg.with_overrides({"data.data_dir": str(tmp_path)})
    asyncio.run(history._download(c, ["BTCUSDT"], 0, 1))
    assert seen == ["futures"] and history._cache_path(c, "BTCUSDT").name == "BTCUSDT_1h_futures.csv"


def test_failed_stop_and_failed_safety_close_are_retried_and_reported_honestly(live_mode_cfg, universe):
    fut = FakeFutures(fail_protection=True)

    async def scenario():
        eng = _engine(live_mode_cfg, universe, fut)
        await eng.bootstrap()
        sym = "BTCUSDT"
        await _book(eng, sym, eng.last_price[sym])
        d, row = _decision(eng, live_mode_cfg, sym)
        fut.fail_close = True
        await eng.open_position(sym, d, row, {})
        assert sym in eng.account.positions and sym in eng.unprotected and sym in eng.close_retry
        assert any("borsada stop emri YOK" in m for m in eng.notifier.messages)
        assert "stop emri YOK" in eng._protection_text()
        fut.fail_close = False
        await eng._retry_pending()
        await asyncio.gather(*eng.tasks)
        return eng

    eng = asyncio.run(scenario())
    assert not eng.account.positions and not eng.close_retry and not eng.unprotected
    assert "güvenlik kapanışı" in eng.account.trades[0].exit_reason


def test_kill_switch_flattens_in_flight_entries_and_retries_failed_closes(live_mode_cfg, universe):
    fut = FakeFutures()

    async def scenario():
        eng = _engine(live_mode_cfg, universe, fut)
        await eng.bootstrap()
        sym = "BTCUSDT"
        await _book(eng, sym, eng.last_price[sym])
        d, row = _decision(eng, live_mode_cfg, sym)
        orig = fut.place_protection

        async def protect_then_halt(*a, **k):
            ids = await orig(*a, **k)
            eng.account.halted, eng.account.halt_reason = True, "test"  # fires while the stop is being placed
            return ids

        fut.place_protection = protect_then_halt
        await eng.open_position(sym, d, row, {})
        assert sym not in eng.account.positions  # closed right after the stop was placed
        # a restored position on a halted account with a failing close is retried every tick
        eng.account.halted = False
        fut.place_protection = orig
        await _book(eng, "ETHUSDT", eng.last_price["ETHUSDT"])
        d2, row2 = _decision(eng, live_mode_cfg, "ETHUSDT")
        await eng.open_position("ETHUSDT", d2, row2, {})
        eng.account.halted, eng.account.halt_reason = True, "test"
        fut.fail_close = True
        eng._flatten_if_halted()
        await asyncio.gather(*list(eng.tasks))
        assert "ETHUSDT" in eng.account.positions
        fut.fail_close = False
        eng._flatten_if_halted()
        await asyncio.gather(*list(eng.tasks))
        return eng

    eng = asyncio.run(scenario())
    assert not eng.account.positions and all("Kill-switch" in t.exit_reason for t in eng.account.trades)


def test_adopted_position_far_from_entry_is_protected_from_the_current_price(live_mode_cfg, universe):
    px = float(universe["ETHUSDT"]["close"].iloc[-2])
    orphan = XPos("ETHUSDT", 1, 0.5, px / 1.06)  # long entered 6% lower: entry-based target already crossed
    fut = FakeFutures(positions={"ETHUSDT": orphan})

    async def scenario():
        eng = _engine(live_mode_cfg, universe, fut)
        await eng.bootstrap()
        await eng.on_price("ETHUSDT", eng.last_price["ETHUSDT"])
        return eng

    eng = asyncio.run(scenario())
    pos = eng.account.positions["ETHUSDT"]  # still open: not closed on its first tick
    assert pos.stop_price < eng.last_price["ETHUSDT"] < pos.take_profit
    assert "güncel fiyata" in pos.reasons[0]


def test_restart_places_new_protection_before_cancelling_the_old(live_mode_cfg, universe, tmp_path):
    live_mode_cfg.runtime.state_db = str(tmp_path / "s.sqlite")
    fut = FakeFutures()

    async def first():
        eng = LiveEngine(live_mode_cfg, rest=FakeREST(universe), notifier=CaptureNotifier(),
                         store=StateStore(live_mode_cfg.state_db), futures=fut)
        await eng.bootstrap()
        await _book(eng, "BTCUSDT", eng.last_price["BTCUSDT"])
        d, row = _decision(eng, live_mode_cfg, "BTCUSDT")
        await eng.open_position("BTCUSDT", d, row, {})
        eng.store.close()

    async def second():
        fut.calls.clear()
        eng = LiveEngine(live_mode_cfg, rest=FakeREST(universe), notifier=CaptureNotifier(),
                         store=StateStore(live_mode_cfg.state_db), futures=fut)
        await eng.bootstrap()
        return eng

    asyncio.run(first())
    eng = asyncio.run(second())
    kinds = [c[0] for c in fut.calls if c[0] in ("protect", "cancel", "cancel_ids")]
    assert kinds == ["protect", "cancel_ids"]  # persisted ids: the position is never without an exchange stop
    assert eng.protection_ids["BTCUSDT"] == ["s1", "t1"]


def test_stale_protection_is_retried_and_cleared_before_a_new_entry(live_mode_cfg, universe):
    fut = FakeFutures()

    async def scenario():
        eng = _engine(live_mode_cfg, universe, fut)
        await eng.bootstrap()
        sym = "BTCUSDT"
        await _book(eng, sym, eng.last_price[sym])
        d, row = _decision(eng, live_mode_cfg, sym)
        await eng.open_position(sym, d, row, {})
        fut.fail_cancel = True
        await eng.close_position(sym, "test")
        assert eng.stale_orders == {sym: None}
        await eng.open_position(sym, d, row, {})  # cannot clear the old orders: no new entry
        assert sym not in eng.account.positions
        fut.fail_cancel = False
        await eng._retry_pending()
        assert not eng.stale_orders
        return eng

    asyncio.run(scenario())
