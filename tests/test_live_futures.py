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
        self.fail_protection = fail_protection
        self.already_closed = already_closed

    async def connect(self):
        self.calls.append(("connect",))

    async def balance(self):
        return Bal(self.wallet, self.wallet * 0.9)

    async def positions(self):
        return dict(self.pos)

    async def market_order(self, symbol, side, qty, ref_price, reduce_only=False):
        self.calls.append(("market", symbol, side, qty, reduce_only))
        self.pos[symbol] = XPos(symbol, side, qty, ref_price * (1 + side * 0.0002))
        return Fill(symbol, side, qty, ref_price * (1 + side * 0.0002), ref_price, qty * ref_price * 0.0005, 2.0, 50.0,
                    pd.Timestamp.now(tz="UTC"))

    async def place_protection(self, symbol, position_side, qty, stop_price, take_profit):
        self.calls.append(("protect", symbol, position_side, qty, stop_price, take_profit))
        if self.fail_protection:
            raise RuntimeError("algo order rejected")
        return ["s1", "t1"]

    async def cancel_protection(self, symbol):
        self.calls.append(("cancel", symbol))

    async def close_position(self, symbol, position_side, qty, ref_price):
        self.calls.append(("close", symbol, position_side, qty))
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
    assert kinds.index("market") < kinds.index("protect") < kinds.index("close") < kinds.index("cancel")
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


def test_run_lock_blocks_second_engine(tmp_path):
    db = str(tmp_path / "s.sqlite")
    (tmp_path / "s.sqlite.lock").write_text(str(os.getppid()))  # a live process (our parent) holds it
    assert running_pid(db) == os.getppid()
    with pytest.raises(RuntimeError):
        RunLock(db).acquire()
    (tmp_path / "s.sqlite.lock").write_text("999999999")  # stale lock from a dead process
    assert running_pid(db) is None
    lock = RunLock(db)
    lock.acquire()
    lock.release()
    assert not (tmp_path / "s.sqlite.lock").exists()


def test_reset_and_resume_refuse_while_running(tmp_path, monkeypatch, capsys):
    from cappatrader import cli

    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(f"runtime:\n  state_db: {tmp_path / 'st.sqlite'}\n  log_dir: {tmp_path / 'logs'}\n")
    (tmp_path / "st.sqlite.lock").write_text(str(os.getppid()))
    assert cli.main(["--config", str(cfg_file), "reset", "--yes"]) == 1
    assert cli.main(["--config", str(cfg_file), "resume"]) == 1
    assert "çalışıyor" in capsys.readouterr().out


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
