"""Live engine: paper trading on real market data, or real orders on Binance USD-M futures.

WebSocket ticks/depth/klines -> cleaning -> (on bar close) online HMM + GARCH + pipeline
-> strategy decision (squeeze-VWAP or ensemble) -> risk -> broker -> SQLite + Telegram.
Stops, take-profits, trailing stops (ensemble only) and the flash-move guard run on
every tick. In live mode the exchange also holds STOP_MARKET / TAKE_PROFIT_MARKET
orders, balances come from Binance, and positions are reconciled with the exchange
at startup and every `live.sync_interval_s`.
"""

from __future__ import annotations

import asyncio
import logging
import math
import signal
import time
from collections import deque

import aiohttp
import numpy as np
import pandas as pd

from cappatrader import fmt
from cappatrader.config import Config
from cappatrader.data.alt_data import AltDataHub
from cappatrader.data.binance_rest import BinanceREST
from cappatrader.data.binance_ws import BinanceStream, market_streams
from cappatrader.data.cleaning import TickFilter, clean_klines
from cappatrader.data.orderbook import OrderBook, SpoofDetector, TradeFlow, orderflow_signal
from cappatrader.execution.costs import CostModel
from cappatrader.execution.orders import build_position
from cappatrader.execution.paper import PaperBroker
from cappatrader.features.indicators import atr as atr_ind
from cappatrader.live.lock import RunLock
from cappatrader.live.state import StateStore
from cappatrader.models import LONG, Decision, Fill, Position
from cappatrader.notify import messages as msg
from cappatrader.notify.telegram import TelegramNotifier
from cappatrader.risk.manager import RiskManager
from cappatrader.risk.sizing import size_position
from cappatrader.risk.stops import stop_reason, update_trailing
from cappatrader.signals.ml import OnlineModel
from cappatrader.strategy.decision import FrameView
from cappatrader.strategy.factory import (decide, explain_decision, is_squeeze, make_decider, strategy_risk,
                                          uses_trailing)
from cappatrader.strategy.pipeline import prepare_universe

log = logging.getLogger(__name__)

HMM_COLS = ["hmm_state", "hmm_label", "hmm_prob"]
GARCH_COLS = ["garch_vol", "garch_longrun", "garch_ratio", "garch_size_mult"]


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _ms(ts: str | pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).value // 1_000_000)


class LiveEngine:
    def __init__(self, cfg: Config, rest: BinanceREST | None = None, notifier: TelegramNotifier | None = None,
                 store: StateStore | None = None, http: aiohttp.ClientSession | None = None, futures=None):
        self.cfg = cfg
        self.live = cfg.mode == "live"
        self.market = cfg.data_market
        self.rest = rest or BinanceREST(cfg.exchange, cfg.secrets, market=self.market)
        self.notifier = notifier or TelegramNotifier(
            cfg.secrets.telegram_token, cfg.secrets.telegram_chat_id,
            enabled=cfg.telegram.enabled, min_interval_s=cfg.telegram.min_interval_s)
        self.store = store or StateStore(cfg.state_db)
        self._fresh_account = not self.store.has_account()
        self.account = self.store.load_account(cfg.account.starting_balance)
        self.costs = CostModel(cfg.costs, cfg.bar_ms)
        self.rcfg = strategy_risk(cfg)
        self.risk = RiskManager(self.rcfg)
        self.decider = make_decider(cfg)
        self.trailing = uses_trailing(cfg)
        self.symbols = list(cfg.symbols)
        for s in self.account.positions:  # a position restored for a symbol removed from the config
            if s not in self.symbols:
                self.symbols.append(s)
        self.bars: dict[str, pd.DataFrame] = {}
        self.books = {s: OrderBook(s) for s in self.symbols}
        self.flow = {s: TradeFlow() for s in self.symbols}
        self.spoof = {s: SpoofDetector() for s in self.symbols}
        self.ticks = {s: TickFilter(max_dev=cfg.data.tick_max_deviation) for s in self.symbols}
        self.flash = {s: deque() for s in self.symbols}
        self.use_ml = cfg.signals.ml.enabled and not is_squeeze(cfg)  # ML only votes in the ensemble
        self.ml = {s: OnlineModel(cfg.signals.ml) for s in self.symbols}
        self.hmm = self._online("hmm") if cfg.regime.method == "hmm" else {}
        self.garch = self._online("garch") if cfg.regime.vol_method == "garch" else {}
        self.last_price: dict[str, float] = {}
        self.last_seen: dict[str, float] = {}
        self.last_atr: dict[str, float] = {}
        self.filters: dict[str, dict] = {}
        self.alt = AltDataHub(self.rest, cfg.data.news_feeds, self.symbols) if cfg.data.alt_data_enabled else None
        self.broker = PaperBroker(self.costs, self.books, self.last_price)
        self.futures = futures
        self.wallet: float | None = None
        self.available: float | None = None
        self.binance_balance_note = ""
        self.entry_lock = asyncio.Lock()
        self.fit_lock = asyncio.Lock()  # model fits touch process-global state (threadpool limits, warnings)
        self.unprotected: set[str] = set()  # live positions whose exchange stop could not be placed (retried each sync)
        self.close_retry: dict[str, str] = {}  # positions the bot failed to close: retried each sync / tick
        # protection orders that could not be cancelled: symbol -> order ids, or None = all of a flat symbol
        self.stale_orders: dict[str, list[str] | None] = {}
        # exchange order ids of each position's stop / target (persisted: a restart replaces them by id)
        self.protection_ids: dict[str, list[str]] = dict(self.store.get_kv("protection_ids") or {})
        self._closed_at: dict[str, float] = {}  # monotonic time of the bot's last close per symbol
        self._task_alerted: dict[str, float] = {}  # rate limit for "order task failed" alerts
        self.inflight: set[str] = set()
        self.tasks: set[asyncio.Task] = set()
        self.protection = False
        self.ws_connected = False
        self.last_eval: dict[str, pd.Timestamp] = {}
        self.pending_entries: set[str] = set()
        self.manage_only: set[str] = set()  # symbols tracked only because an open position needs its stop
        self.warnings: list[str] = []
        self.dirty = False
        self.crashed = False
        hm, gc = cfg.regime.hmm, cfg.regime.garch
        # a whole anchored refit chunk must fit: training window + the chunk itself (+ margin)
        self.max_bars = max(cfg.runtime.history_bars,
                            cfg.signals.ml.train_bars + 300 if self.use_ml else 0,
                            hm.train_bars + hm.feature_window + hm.retrain_every + 50,
                            gc.train_bars + gc.retrain_every + 50,
                            cfg.strategy.squeeze.length * 4 + 50)
        self.stop_event: asyncio.Event | None = None
        self.lock = RunLock(cfg.state_db)
        self._http = http
        self._last_alt = 0.0
        self._last_equity_rec = 0.0
        self._last_heartbeat = 0.0
        self._last_sync = 0.0
        self._summary_day: str | None = None

    def _online(self, kind: str) -> dict:
        if kind == "hmm":
            from cappatrader.regime.hmm import OnlineHMM

            return {s: OnlineHMM(self.cfg.regime.hmm) for s in self.symbols}
        from cappatrader.regime.garch import OnlineGARCH

        return {s: OnlineGARCH(self.cfg.regime.garch) for s in self.symbols}

    # -- helpers -----------------------------------------------------------
    def equity(self) -> float:
        if self.live and self.wallet is not None:
            return self.wallet + sum(p.unrealized(self.last_price.get(s, p.entry_price))
                                     for s, p in self.account.positions.items())
        return self.account.equity(self.last_price)

    def cash(self) -> float:
        return self.available if (self.live and self.available is not None) else self.account.cash

    def alert(self, title: str, text: str, kind: str = "ALERT") -> None:
        log.warning("%s: %s", title, text)
        self.store.record_event(_now().isoformat(), kind, f"{title}: {text}")
        # paper and live accounts can report to the same chat: say which one is talking
        self.notifier.notify(msg.alert_message(f"{title} · {self.cfg.mode_label}", text))

    def _spawn(self, sym: str, coro) -> None:
        if sym in self.inflight:
            coro.close()
            return
        self.inflight.add(sym)
        task = asyncio.create_task(coro)
        self.tasks.add(task)

        def done(t: asyncio.Task) -> None:
            self.tasks.discard(t)
            self.inflight.discard(sym)
            if not t.cancelled() and t.exception() is not None:
                log.error("Emir görevi hata verdi (%s)", sym, exc_info=t.exception())
                now = time.monotonic()
                if now - self._task_alerted.get(sym, -math.inf) >= 600:  # retries must not flood the chat
                    self._task_alerted[sym] = now
                    self.alert("Emir görevi hata verdi",
                               f"{fmt.pair(sym)}: {type(t.exception()).__name__}: {t.exception()}")

        task.add_done_callback(done)

    def _quote_volume(self, sym: str) -> float | None:
        df = self.bars.get(sym)
        if df is None or df.empty or "quote_volume" not in df:
            return None
        return float(df["quote_volume"].iloc[-1])

    def _mark_manage_only(self) -> None:
        for s in self.symbols:
            if s not in self.cfg.symbols:
                self.manage_only.add(s)
                self.warnings.append(f"{fmt.pair(s)} ayarlarda yok ama açık pozisyonu var: yalnızca stop/hedef takibi "
                                     "yapılıyor, yeni işlem açılmayacak")

    def _drop_symbol(self, sym: str, why: str) -> None:
        if sym in self.symbols and sym not in self.account.positions:
            self.symbols.remove(sym)
            self.warnings.append(f"{fmt.pair(sym)} devre dışı: {why}")

    # -- bootstrap -----------------------------------------------------------
    async def bootstrap(self) -> None:
        cfg = self.cfg
        self._mark_manage_only()
        try:
            self.filters = await self.rest.exchange_filters(self.symbols)
        except Exception as exc:  # filters only refine rounding; keep going
            self.warnings.append(f"Borsa filtreleri alınamadı ({type(exc).__name__}); miktar yuvarlama kapalı")
        if self.filters:
            market = "USD-M futures" if self.market == "futures" else "spot"
            for s in [s for s in self.symbols if s not in self.filters]:
                self._drop_symbol(s, f"Binance {market} piyasasında bulunamadı")
        for s in list(self.symbols):
            status = self.filters.get(s, {}).get("status")
            if status and status != "TRADING":
                self.warnings.append(f"{fmt.pair(s)} borsada işlem durumunda değil ({status})")
            df = await self.rest.recent_klines(s, cfg.timeframe, self.max_bars)
            df, rep = clean_klines(df, cfg.timeframe, cfg.data.outlier_mad_k)
            if df.empty:
                raise RuntimeError(f"{s} için geçmiş veri alınamadı")
            self.bars[s] = df
            self.last_price[s] = float(df["close"].iloc[-1])
            self.last_seen[s] = time.monotonic()
            atr0 = float(atr_ind(df["high"], df["low"], df["close"], cfg.regime.atr_period).iloc[-1]) if len(df) else np.nan
            if math.isfinite(atr0):
                self.last_atr[s] = atr0
            log.info("%s: %d bar yüklendi (%s)", s, len(df), rep)
        for s in self.symbols:
            if self.use_ml:
                ok = await asyncio.to_thread(self.ml[s].fit, self.bars[s])
                log.info("%s ML modeli %s", s, "eğitildi" if ok else "eğitilemedi (yetersiz veri)")
            for kind, models in (("HMM rejim", self.hmm), ("GARCH", self.garch)):
                if s in models:  # the backtest scores the current chunk with a fit on the bars before it
                    ok = await asyncio.to_thread(models[s].fit_chunk, self.bars[s])
                    log.info("%s %s modeli %s", s, kind, "eğitildi" if ok else "eğitilemedi (yetersiz veri)")
        if self.live:
            if cfg.live.env == "mainnet":  # demo.binance.com keys have no /sapi endpoint
                restrictions = await self._check_api_key()
                if not restrictions:
                    raise RuntimeError("API anahtarının yetkileri okunamadı (para çekme kapalı mı doğrulanamadı); "
                                       "gerçek parayla işlem başlatılmadı. Bağlantıyı kontrol edip tekrar deneyin: "
                                       "python -m cappatrader check")
                if restrictions.get("enableWithdrawals"):
                    raise RuntimeError("API anahtarında PARA ÇEKME yetkisi açık: gerçek parayla işlem bu anahtarla "
                                       "başlatılmaz. Binance > API Yönetimi'nden para çekmeyi kapatın ve IP kısıtlaması "
                                       "ekleyin (PRD §6).")
            await self._connect_futures()
        else:
            await self._check_api_key()
            await self._seed_paper_balance()
        if self.alt is not None:
            await self.alt.refresh(await self.http())
            self._last_alt = time.time()

    async def _check_api_key(self) -> dict:
        """Warn about risky API-key permissions (PRD §6); returns Binance's apiRestrictions ({} if unknown)."""
        if not self.cfg.secrets.has_binance:
            log.info("Binance API anahtarı yok: paper modda gerekli değil")
            return {}
        try:
            r = await self.rest.api_restrictions()
        except Exception as exc:
            self.warnings.append(f"API anahtarı yetki kontrolü yapılamadı: {exc}")
            return {}
        if r.get("enableWithdrawals"):
            self.warnings.append("API anahtarında PARA ÇEKME yetkisi AÇIK — Binance'te hemen kapatın (PRD güvenlik)")
        if not r.get("ipRestrict"):
            self.warnings.append("API anahtarında IP kısıtlaması yok — sunucu IP'sine kısıtlamanız önerilir")
        if not self.live and (r.get("enableSpotAndMarginTrading") or r.get("enableFutures")):
            self.warnings.append("API anahtarında işlem yetkisi açık; paper mod için yalnızca okuma yetkisi yeterli")
        return r

    async def _seed_paper_balance(self) -> None:
        """balance_source "binance": a brand-new paper account starts from the real Binance balance."""
        if self.cfg.account.balance_source != "binance":
            return
        from cappatrader.execution.binance_futures import fetch_binance_balance

        try:
            bal = await fetch_binance_balance(self.cfg)
        except Exception as exc:
            self.warnings.append(f"Binance bakiyesi okunamadı ({type(exc).__name__}: {exc}); sanal bakiye kullanılıyor")
            return
        wallet = self.cfg.account.binance_wallet
        self.binance_balance_note = f"Binance {wallet} bakiyesi: {fmt.usd(bal)}"
        fresh = self._fresh_account or (not self.account.trades and not self.account.positions)
        if fresh and bal > 0:
            self._reset_account_balance(bal)
            log.info("Kağıt hesap Binance bakiyesiyle başlatıldı: %.2f USDT", bal)

    def _reset_account_balance(self, balance: float) -> None:
        a = self.account
        a.starting_balance = a.cash = a.peak_equity = a.day_start_equity = float(balance)
        self.store.save_account(a)

    async def _connect_futures(self) -> None:
        if self.futures is None:
            from cappatrader.execution.binance_futures import BinanceFuturesBroker

            self.futures = BinanceFuturesBroker(self.cfg)
        await self.futures.connect()
        unavailable = getattr(self.futures, "unavailable", None) or {}
        reasons = unavailable if isinstance(unavailable, dict) else {}
        for s in list(unavailable):
            self._drop_symbol(s, reasons.get(s) or "Binance USD-M futures'ta işlem görmüyor")
        for w in getattr(self.futures, "warnings", None) or []:
            if not any(fmt.pair(s) in w for s in unavailable):  # dropped symbols are already reported
                self.warnings.append(w)
        self._broker_warnings(drain_only=True)
        await self.refresh_balance()
        seed = (self._fresh_account or (not self.account.trades and not self.account.positions)
                or bool(self.store.get_kv("seed_pending")))
        if seed:  # durable: a start interrupted after adopting positions must still seed on the next one
            self.store.set_kv("seed_pending", "1")
        self.binance_balance_note = (f"Binance futures: cüzdan {fmt.usd(self.wallet or 0)}, "
                                     f"kullanılabilir {fmt.usd(self.available or 0)}")
        await self.reconcile_positions(startup=True)
        if seed:
            # a new (or reset, or never-traded) live ledger starts from the real futures equity, measured after
            # adopting the exchange's positions so their open P&L is not read as a drawdown
            self._reset_account_balance(self.equity())
            self.account.cash = self.available if self.available is not None else self.account.cash
            self.store.save_account(self.account)
            self.store.set_kv("seed_pending", "")
        if self.store.get_kv("reseed_peak"):
            # `resume` cannot know the live equity offline: restart the drawdown / daily windows from Binance's
            a, eq = self.account, self.equity()
            a.peak_equity = a.day_start_equity = eq
            a.day = _now().strftime("%Y-%m-%d")
            self.store.set_kv("reseed_peak", "")
            self.store.save_account(a)
            log.info("Zirve ve gün başı bakiye Binance özsermayesinden yeniden başlatıldı: %.2f USDT", eq)

    async def refresh_balance(self) -> None:
        bal = await self.futures.balance()
        self.wallet, self.available = float(bal.wallet), float(bal.available)
        self.account.cash = self.available

    async def http(self) -> aiohttp.ClientSession:
        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession(trust_env=True)
        return self._http

    # -- exchange reconciliation (live) ------------------------------------------
    def _synthetic_fill(self, sym: str, side: int, qty: float) -> Fill:
        px = self.last_price.get(sym) or self.account.positions[sym].entry_price
        return Fill(symbol=sym, side=side, qty=qty, price=px, ref_price=px, fee=0.0, slippage_bps=0.0,
                    latency_ms=0.0, timestamp=_now())

    async def _record_close(self, sym: str, fill: Fill, reason: str) -> None:
        if sym not in self.account.positions:
            return
        self._closed_at[sym] = time.monotonic()
        self.unprotected.discard(sym)
        self.close_retry.pop(sym, None)
        if self.protection_ids.pop(sym, None) is not None:
            self._save_protection_ids()
        trade = self.account.close_position(sym, fill, reason, _now().isoformat())
        if self.live and self.available is not None:
            self.account.cash = self.available
        self.store.record_trade(trade)
        self.store.save_account(self.account)
        self.notifier.notify(msg.exit_message(
            trade, fill, self.equity(), self.cash(), self.account.starting_balance,
            len(self.account.positions), self.account.trades, mode=self.cfg.mode_label))

    def _adopt(self, sym: str, side: int, qty: float, entry: float) -> Position:
        atr = self.last_atr.get(sym, 0.0)
        if not (math.isfinite(atr) and atr > 0):  # a brand-new listing has no ATR yet: 1% of the entry
            atr = entry * 0.01
        dist = self.rcfg.atr_stop_mult * atr
        tp_r = self.rcfg.take_profit_r
        px = self.last_price.get(sym) or entry
        note = "Borsada bulunan sahipsiz pozisyon devralındı"
        stop, tp = entry - side * dist, (entry + side * tp_r * dist if tp_r else None)
        # a level the price has already passed would close the position on its first tick: move only that
        # level to the current price (the target therefore never lands on the losing side of the entry)
        if (px - stop) * side <= 0:
            stop = px - side * dist
            note += f" (fiyat giriş bazlı stopun ötesinde: stop güncel fiyata ({fmt.price(px)}) göre kondu)"
        if tp is not None and (tp - px) * side <= 0:
            tp = px + side * tp_r * dist
            note += f" (fiyat giriş bazlı hedefi geçmiş: hedef güncel fiyata ({fmt.price(px)}) göre kondu)"
        pos = Position(symbol=sym, side=side, qty=qty, entry_price=entry, entry_time=_now().isoformat(),
                       stop_price=stop, initial_stop=stop, risk_per_unit=dist,
                       best_price=entry, entry_regime="UNKNOWN", entry_confidence=0.0, entry_fee=0.0,
                       entry_slippage_cost=0.0, reasons=[note],
                       take_profit=tp, style="SQZ" if is_squeeze(self.cfg) else "TREND")
        self.account.open_position(pos, self._synthetic_fill_for(sym, side, qty, entry), enforce_cash=False)
        return pos

    @staticmethod
    def _synthetic_fill_for(sym: str, side: int, qty: float, price: float) -> Fill:
        return Fill(symbol=sym, side=side, qty=qty, price=price, ref_price=price, fee=0.0, slippage_bps=0.0,
                    latency_ms=0.0, timestamp=_now())

    def _broker_warnings(self, drain_only: bool = False) -> None:
        """Forward warnings the broker collected after connect (e.g. take-profit not placed) to Telegram."""
        pending = getattr(self.futures, "warnings", None)
        if not isinstance(pending, list):
            return
        items, pending[:] = list(pending), []
        if not drain_only:
            for w in items:
                self.alert("Borsa uyarısı", w)

    def _exchange_protected(self, sym: str) -> bool:
        return self.live and self.cfg.live.place_exchange_stops and sym not in self.unprotected

    def _protection_text(self) -> str:
        """What actually protects the open positions while the bot is not watching them."""
        if not self.live or not self.account.positions:
            return "" if self.live else "Kağıt hesapta pozisyonlar yalnızca bot çalışırken izlenir."
        naked = [s for s in self.account.positions if not self._exchange_protected(s)]
        if not naked:
            return "Borsadaki stop emirleri açık pozisyonları korumaya devam ediyor."
        return ("DİKKAT: " + ", ".join(fmt.pair(s) for s in naked) + " için borsada stop emri YOK; "
                "botu hemen yeniden başlatın veya pozisyonu Binance'te elle koruyun.")

    def _save_protection_ids(self) -> None:
        self.store.set_kv("protection_ids", self.protection_ids)

    async def _guarded(self, coro):
        """Run an exchange action that must not be interrupted half-way (e.g. by SIGTERM): it is a tracked
        task that shutdown waits for, and cancelling the caller does not cancel it."""
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return await asyncio.shield(task)

    async def _protect(self, pos: Position) -> None:
        if not self.cfg.live.place_exchange_stops:
            return
        try:
            ids = await self.futures.place_protection(pos.symbol, pos.side, pos.qty, pos.stop_price, pos.take_profit)
        finally:
            self._broker_warnings()
        self.protection_ids[pos.symbol] = [str(x) for x in ids or []]
        self._save_protection_ids()
        self.unprotected.discard(pos.symbol)

    async def _swap_protection(self, pos: Position) -> None:
        old: list[str] | None = list(self.protection_ids.get(pos.symbol) or [])
        if not old:  # unknown (adopted position, reset ledger, first start): ask the exchange what is working
            lister = getattr(self.futures, "protection_order_ids", None)
            try:
                old = list(await lister(pos.symbol)) if lister is not None else None
            except Exception as exc:
                log.warning("%s açık koruma emirleri listelenemedi: %s", pos.symbol, exc)
                old = None
        if old is not None:
            if not old:
                await self._protect(pos)
                return
            # place the new stop/target first and only then cancel the old ones by id: never a moment
            # without an exchange stop (reduce-only orders cannot over-close the position)
            await self._protect(pos)
            try:
                await self.futures.cancel_protection(pos.symbol, old)
            except Exception as exc:
                self.stale_orders[pos.symbol] = old
                log.warning("%s eski koruma emirleri iptal edilemedi (yeniden denenecek): %s", pos.symbol, exc)
        else:  # the working orders could not be listed: clear the symbol, then place (shielded by _guarded)
            await self.futures.cancel_protection(pos.symbol)
            await self._protect(pos)

    async def _reprotect(self, pos: Position, why: str) -> None:
        """Replace the exchange stop/target; on failure keep tracking the stop locally and retry each sync."""
        if not self.cfg.live.place_exchange_stops:
            return
        try:
            await self._guarded(self._swap_protection(pos))
        except Exception as exc:
            first = pos.symbol not in self.unprotected
            self.unprotected.add(pos.symbol)
            if first:
                self.alert("Koruma emri yerleştirilemedi",
                           f"{fmt.pair(pos.symbol)} ({why}): {type(exc).__name__}: {exc}. Stop ve hedef bot tarafından "
                           "yerel olarak izleniyor; borsa emri her senkronda yeniden denenecek.")
            else:
                log.warning("%s koruma emri yine yerleştirilemedi: %s", pos.symbol, exc)

    async def _cancel_flat(self, sym: str) -> None:
        """Cancel every protection order of a symbol that has just gone flat; retried each sync on failure."""
        try:
            await self.futures.cancel_protection(sym)
            self.stale_orders.pop(sym, None)
        except Exception as exc:
            if sym not in self.stale_orders:
                self.alert("Eski koruma emirleri iptal edilemedi",
                           f"{fmt.pair(sym)}: {type(exc).__name__}: {exc}. Her senkronda yeniden denenecek; bu paritede "
                           "yeni giriş, emirler temizlenene kadar açılmaz.")
            self.stale_orders[sym] = None

    async def _retry_pending(self) -> None:
        """Retry what failed before: closes the bot wanted (safety / kill-switch), stale protection orders."""
        for sym, ids in list(self.stale_orders.items()):
            if sym in self.inflight or (ids is None and sym in self.account.positions):
                continue
            try:
                await self.futures.cancel_protection(sym, ids) if ids else await self.futures.cancel_protection(sym)
                self.stale_orders.pop(sym, None)
            except Exception as exc:
                log.warning("%s eski koruma emirleri yine iptal edilemedi: %s", sym, exc)
        for sym, reason in list(self.close_retry.items()):
            if sym not in self.account.positions:
                self.close_retry.pop(sym, None)
            elif sym not in self.inflight:
                self._spawn(sym, self.close_position(sym, reason))

    def _flatten_if_halted(self) -> None:
        """While the kill-switch is active the book must be flat: (re)issue closes the first pass missed
        (a symbol with an order in flight, a failed close, positions restored or adopted after a restart)."""
        if not self.account.halted:
            return
        for sym in list(self.account.positions):
            if sym not in self.inflight:
                self._spawn(sym, self.close_position(sym, f"Kill-switch: {self.account.halt_reason}"))

    async def reconcile_positions(self, startup: bool = False) -> None:
        """Make the local ledger agree with the exchange: record positions the exchange closed,
        adopt exchange positions the bot does not know (no orphan positions, PRD §6)."""
        held = list(dict.fromkeys([*self.symbols, *self.account.positions]))
        snapshot_at = time.monotonic()
        ex = await self.futures.positions(held)
        adopted: set[str] = set()
        for sym, pos in list(self.account.positions.items()):
            if sym in self.inflight:
                continue
            e = ex.get(sym)
            if e is None or e.side != pos.side:
                fill = await self.futures.last_close_fill(sym, _ms(pos.entry_time), pos.side)
                note = ""
                if fill is None:
                    fill = self._synthetic_fill(sym, -pos.side, pos.qty)
                    note = " — kapanış fiyatı bulunamadı, son fiyat kullanıldı"
                await self._cancel_flat(sym)
                await self._record_close(sym, fill, "Pozisyon borsada kapandı (borsa stop/hedef emri veya manuel)" + note)
            elif abs(e.qty - pos.qty) > 1e-12 * max(1.0, pos.qty):
                log.warning("%s miktar uyuşmazlığı: yerel %s, borsa %s — borsa esas alındı", sym, pos.qty, e.qty)
                pos.qty = e.qty
                self.dirty = True
                await self._reprotect(pos, "miktar borsaya göre güncellendi")
            elif sym in self.unprotected:
                await self._reprotect(pos, "yeniden deneme")
        for sym, e in ex.items():
            if sym in self.account.positions or sym in self.inflight:
                continue
            if self._closed_at.get(sym, -math.inf) >= snapshot_at:
                continue  # the bot closed it while this pass was running: the snapshot is stale, not an orphan
            if sym not in self.symbols:
                self.alert("Yönetilmeyen pozisyon",
                           f"Borsada {fmt.pair(sym)} pozisyonu var ama ayarlarda bu parite yok; bot dokunmuyor.")
                continue
            pos = self._adopt(sym, e.side, e.qty, e.entry_price)
            self.store.save_account(self.account)
            self.alert("Sahipsiz pozisyon devralındı",
                       f"{fmt.pair(sym)} {'LONG' if e.side == LONG else 'SHORT'} {fmt.qty(e.qty)} @ {fmt.price(e.entry_price)}; "
                       f"stop {fmt.price(pos.stop_price)}" + (f", hedef {fmt.price(pos.take_profit)}" if pos.take_profit else ""),
                       "ADOPT")
            adopted.add(sym)
            await self._reprotect(pos, "devralınan pozisyon")
        if startup:
            for sym, pos in list(self.account.positions.items()):
                if sym not in adopted:
                    await self._reprotect(pos, "yeniden başlatma")

    # -- stream handlers -----------------------------------------------------
    async def on_stream(self, stream: str, data: dict) -> None:
        if "@" not in stream:
            return
        sym_l, kind = stream.split("@", 1)
        sym = sym_l.upper()
        if sym not in self.books or sym not in self.symbols:
            return
        if kind.startswith("kline"):
            await self.on_kline(sym, data["k"])
        elif kind == "aggTrade":
            self.flow[sym].add(time.monotonic(), float(data["p"]), float(data["q"]), bool(data.get("m")))
            await self.on_price(sym, float(data["p"]))
        elif kind.startswith("depth"):
            now = time.monotonic()
            # spot partial depth: bids/asks; futures depth stream: b/a
            bids = data.get("bids", data.get("b", []))
            asks = data.get("asks", data.get("a", []))
            self.books[sym].update(bids, asks, data.get("lastUpdateId", data.get("u", 0)), now)
            self.spoof[sym].on_book(self.books[sym], now)
            self.last_seen[sym] = now

    async def on_ws_state(self, connected: bool) -> None:
        self.ws_connected = connected
        if not connected and not self.protection:
            self.protection = True
            self.alert("Veri bağlantısı koptu",
                       "WebSocket bağlantısı kesildi. Koruma modu: yeni işlem açılmıyor, açık pozisyonların stopları "
                       "REST fiyatıyla izleniyor. " + self._protection_text()
                       + " Otomatik yeniden bağlanılıyor.", "PROTECTION")
        elif connected:
            for s in list(self.symbols):
                await self._backfill(s)
                last = self.bars.get(s)
                if last is not None and len(last) and s in self.last_eval and last.index[-1] > self.last_eval[s]:
                    # a bar closed while the stream was down: decide on it now instead of skipping it
                    await self.on_bar_close(s, last.index[-1])

    async def on_price(self, sym: str, price: float, from_stream: bool = True) -> None:
        now = time.monotonic()
        if not self.ticks[sym].accept(price):
            return
        self.last_price[sym] = price
        if from_stream:  # a REST fallback price must not make a dead stream look alive
            self.last_seen[sym] = now
        self.spoof[sym].on_trade(price)
        self._flash_check(sym, price, now)
        pos = self.account.positions.get(sym)
        if pos is None:
            return
        long = pos.side == LONG
        if (long and price <= pos.stop_price) or (not long and price >= pos.stop_price):
            self._spawn(sym, self.close_position(sym, stop_reason(pos)))
            return
        tp = pos.take_profit
        if tp is not None and ((long and price >= tp) or (not long and price <= tp)):
            self._spawn(sym, self.close_position(sym, "Hedef kâr (take-profit) seviyesine ulaşıldı"))
            return
        if self.trailing and update_trailing(pos, price, price, self.last_atr.get(sym, 0.0), self.rcfg):
            self.dirty = True

    def _flash_check(self, sym: str, price: float, now: float) -> None:
        dq = self.flash[sym]
        if dq and now - dq[-1][0] < 0.5:
            return
        dq.append((now, price))
        while dq and now - dq[0][0] > self.cfg.risk.flash_window_s:
            dq.popleft()
        prices = [p for _, p in dq]
        move = max(prices) / min(prices) - 1.0
        halt = self.account.symbol_halts.get(sym)
        if move >= self.cfg.risk.flash_move_pct and (halt is None or _now() >= pd.Timestamp(halt)):
            until = _now() + pd.Timedelta(minutes=self.cfg.risk.flash_cooldown_min)
            self.account.symbol_halts[sym] = until.isoformat()
            self.dirty = True
            self.alert("Ani fiyat hareketi (flash crash/pump)",
                       f"{fmt.pair(sym)} {self.cfg.risk.flash_window_s} sn içinde %{move * 100:.1f} oynadı. "
                       f"{self.cfg.risk.flash_cooldown_min} dk yeni işlem açılmayacak; açık pozisyon stopla korunuyor.",
                       "FLASH")

    async def on_kline(self, sym: str, k: dict) -> None:
        if not k.get("x"):
            return
        ts = pd.Timestamp(int(k["t"]), unit="ms", tz="UTC")
        row = pd.DataFrame({
            "open": [float(k["o"])], "high": [float(k["h"])], "low": [float(k["l"])], "close": [float(k["c"])],
            "volume": [float(k["v"])], "quote_volume": [float(k["q"])], "taker_buy_volume": [float(k["V"])],
        }, index=pd.DatetimeIndex([ts], name="open_time"))
        df = self.bars.get(sym)
        if df is not None and len(df):
            last = df.index[-1]
            if ts < last:
                return
            if ts > last + pd.Timedelta(milliseconds=self.cfg.bar_ms):
                await self._backfill(sym)
                df = self.bars[sym]
            df = df[df.index < ts]
        merged = pd.concat([df, row]) if df is not None and len(df) else row
        merged, _ = clean_klines(merged.tail(self.max_bars), self.cfg.timeframe, self.cfg.data.outlier_mad_k)
        self.bars[sym] = merged
        await self.on_bar_close(sym, ts)

    async def _backfill(self, sym: str) -> None:
        df = self.bars.get(sym)
        if df is None or df.empty:
            return
        start_ms = int(df.index[-1].value // 1_000_000)
        try:
            new = await self.rest.klines_range(sym, self.cfg.timeframe, start_ms, int(time.time() * 1000))
        except Exception as exc:
            log.warning("%s eksik bar tamamlanamadı: %s", sym, exc)
            return
        if not new.empty:
            merged = pd.concat([df[df.index < new.index[0]], new])
            self.bars[sym], rep = clean_klines(merged.tail(self.max_bars), self.cfg.timeframe, self.cfg.data.outlier_mad_k)
            log.info("%s eksik barlar REST ile tamamlandı (%s)", sym, rep)

    # -- decisions -------------------------------------------------------------
    async def _live_layers(self, sym: str, bars: pd.DataFrame) -> dict:
        """Single-bar HMM / GARCH layers from the online models (only the last row is filled).

        Refits happen on the walk-forward's anchored bars and only see the bars before the chunk."""
        layers: dict = {}
        if sym in self.hmm:
            m = self.hmm[sym]
            m.bars_since_fit += 1
            if m.due(bars):
                async with self.fit_lock:
                    await asyncio.to_thread(m.fit_chunk, bars)
            label, prob, state = m.infer_last(bars)
            h = pd.DataFrame({"hmm_state": np.nan, "hmm_label": "UNKNOWN", "hmm_prob": np.nan}, index=bars.index)
            h.iloc[-1] = [state, label, prob]
            layers["hmm"] = h
        if sym in self.garch:
            g = self.garch[sym]
            g.bars_since_fit += 1
            if g.due(bars):
                async with self.fit_lock:
                    await asyncio.to_thread(g.fit_chunk, bars)
            vals = g.forecast_last(bars)
            gdf = pd.DataFrame(np.nan, index=bars.index, columns=GARCH_COLS)
            gdf.iloc[-1] = [vals.get(c, np.nan) for c in GARCH_COLS]
            layers["garch"] = gdf
        return layers

    async def _await_partners(self, partners: set[str], ts: pd.Timestamp, timeout_s: float = 3.0) -> None:
        """Pairs scores need the partner's bar for the same timestamp; Binance does not order klines
        across symbols, so give the partners a moment to arrive (keeps live equal to the backtest)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if all(p in self.bars and len(self.bars[p]) and self.bars[p].index[-1] >= ts for p in partners):
                return
            await asyncio.sleep(0.1)

    async def on_bar_close(self, sym: str, ts: pd.Timestamp) -> None:
        cfg = self.cfg
        self.last_eval[sym] = ts
        partners: set[str] = set()
        if not is_squeeze(cfg):  # only the ensemble's pairs-trading signal reads partner prices
            partners = ({b for a, b in cfg.pairs if a == sym} | {a for a, b in cfg.pairs if b == sym}) - {sym}
            if partners:
                await self._await_partners(partners, ts)
        data = {s: self.bars[s] for s in {sym} | partners if s in self.bars}
        bars = self.bars[sym]
        proba = None
        if self.use_ml:
            model = self.ml[sym]
            model.bars_since_fit += 1
            if model.needs_fit():
                async with self.fit_lock:
                    await asyncio.to_thread(model.fit, bars)
            p = model.predict_last(bars)
            proba = pd.Series(np.nan, index=bars.index)
            proba.iloc[-1] = p
        layers = {s: {} for s in data}  # partners: no HMM/GARCH needed, only their prices for pairs
        layers[sym] = await self._live_layers(sym, bars)
        now = time.monotonic()
        of_score, of_note = orderflow_signal(self.books[sym], self.flow[sym], self.spoof[sym], now, fmt.base_asset(sym))
        sent_score, sent_note = self.alt.sentiment(sym) if self.alt is not None else (None, "")
        frames = await asyncio.to_thread(
            prepare_universe, data, cfg, {sym: proba} if proba is not None else None,
            {sym: {"orderflow": of_score, "sentiment": sent_score}}, layers)
        f = frames[sym]
        v = FrameView(f)
        i = len(f) - 1
        atr_now = float(v["atr"][i])
        if np.isfinite(atr_now):
            self.last_atr[sym] = atr_now
        pos = self.account.positions.get(sym)
        if pos is not None:
            pos.bars_held += 1
            if self.trailing:
                update_trailing(pos, float(v["high"][i]), float(v["low"][i]), self.last_atr.get(sym, 0.0),
                                self.rcfg)  # no finite ATR yet (new listing): update_trailing skips it
            self.dirty = True
        last_exit = self.account.last_exit.get(sym)
        # bars since the exit, counted like the backtest (exit inside bar k -> 0 at k's close, 1 at k+1's)
        bse = (max(0, math.ceil((ts - pd.Timestamp(last_exit)) / pd.Timedelta(milliseconds=cfg.bar_ms)))
               if last_exit else None)
        d = decide(self.decider, cfg, sym, v, i, pos, bse)
        log.info("%s bar %s | rejim %s/%s | skor %+.2f | %s", sym, ts, d.regime, d.vol_state, d.composite, d.action)
        notes = {"orderflow": of_note, "sentiment": sent_note}
        if d.action == "EXIT":
            if d.reverse is not None:  # stop-and-reverse, as in the backtest: open the opposite side after the close
                self._spawn(sym, self._exit_then_enter(sym, d, f.iloc[i], notes))
            else:
                self._spawn(sym, self.close_position(sym, d.exit_reason))
        elif d.action in ("ENTER_LONG", "ENTER_SHORT"):
            if self._entry_ok(sym):
                self.pending_entries.add(sym)
                self._spawn(sym, self._entry_task(sym, d, f.iloc[i], notes))
        await self.check_guards()
        self.store.save_account(self.account)

    def _entry_ok(self, sym: str) -> bool:
        ok, why = self.risk.entry_allowed(self.account, sym, _now(), len(self.pending_entries - {sym}))
        if self.protection:
            ok, why = False, "Koruma modu (veri akışı sorunlu)"
        if sym in self.manage_only:
            ok, why = False, "parite ayarlarda yok (yalnızca açık pozisyon takibi)"
        if not ok:
            log.info("%s giriş sinyali reddedildi: %s", sym, why)
        return ok

    async def _entry_task(self, sym: str, d: Decision, row: pd.Series, notes: dict[str, str]) -> None:
        try:
            await self.open_position(sym, d, row, notes)
        finally:
            self.pending_entries.discard(sym)

    async def _exit_then_enter(self, sym: str, d: Decision, row: pd.Series, notes: dict[str, str]) -> None:
        await self.close_position(sym, d.exit_reason)
        if sym in self.account.positions or not self._entry_ok(sym):  # the close failed or entries are blocked
            return
        self.pending_entries.add(sym)
        await self._entry_task(sym, d.reverse, row, notes)

    async def open_position(self, sym: str, d: Decision, row: pd.Series, notes: dict[str, str]) -> None:
        cfg = self.cfg
        async with self.entry_lock:
            if sym in self.account.positions or self.account.halted:
                return
            if self.live:
                await self.refresh_balance()
            ref = self.broker.reference_price(sym)
            eq = self.equity()
            qv = self._quote_volume(sym)
            flt = self.filters.get(sym, {})
            # futures: notional is limited by margin x leverage; spot paper: by cash
            budget = (self.available or 0.0) * cfg.live.leverage if self.live else self.account.cash
            sz = size_position(self.rcfg, eq, budget, ref, d.atr, d.side, d.vol_state, self.account.trades,
                               step_size=flt.get("step_size"), min_notional=flt.get("min_notional"),
                               round_trip_cost_pct=self.costs.round_trip_pct(ref, d.atr, 0.1 * eq, qv),
                               size_mult=d.size_mult)
            if not sz.ok:
                log.info("%s pozisyon açılmadı: %s", sym, sz.rejected)
                return
            if self.live:
                try:  # a fresh position must not inherit reduce-only orders left from an earlier trade
                    await self.futures.cancel_protection(sym)
                    self.stale_orders.pop(sym, None)
                except Exception as exc:
                    log.warning("%s giriş atlandı: eski koruma emirleri temizlenemedi: %s", sym, exc)
                    self.stale_orders[sym] = None
                    return
                try:
                    fill = await self.futures.market_order(sym, d.side, sz.qty, ref)
                except Exception as exc:
                    unknown = type(exc).__name__ == "OrderStateUnknown"
                    self.alert("Emir gönderilemedi" if not unknown else "Emir durumu belirsiz",
                               f"{fmt.pair(sym)} {d.action}: {type(exc).__name__}: {exc}"
                               + (". Bir sonraki borsa senkronunda pozisyon varsa devralınıp stop konacak."
                                  if unknown else ""))
                    return
            else:
                fill = await self.broker.market_order(sym, d.side, sz.qty, ref, d.atr, qv)
                if fill.notional + fill.fee > self.account.cash:
                    log.warning("%s: dolum sonrası nakit yetersiz, emir iptal", sym)
                    return
            reasons = explain_decision(cfg, row, d.side, notes)
            pos = build_position(d, fill, sz, self.rcfg, reasons, _now().isoformat())
            self.account.open_position(pos, fill, enforce_cash=not self.live)
            self.store.save_account(self.account)
            self.store.record_event(_now().isoformat(), "ENTRY", f"{sym} {d.action} {fill.qty}@{fill.price}")
            if self.account.halted:
                # The kill-switch fired while the order was travelling: undo it immediately.
                await self.close_position(sym, "Kill-switch emir iletimi sırasında tetiklendi — pozisyon hemen kapatıldı")
                return
            if self.live:
                try:
                    await self._guarded(self._protect(pos))
                except Exception as exc:
                    self.unprotected.add(sym)  # if the safety close fails too, the sync retries both
                    self.alert("Koruma emri yerleştirilemedi",
                               f"{fmt.pair(sym)} pozisyonu güvenlik için kapatılıyor: {type(exc).__name__}: {exc}")
                    await self.close_position(sym, "Borsa tarafı stop emri yerleştirilemedi (güvenlik kapanışı)")
                    return
                try:
                    await self.refresh_balance()
                except Exception as exc:
                    log.warning("Bakiye güncellenemedi: %s", exc)
                if self.account.halted:  # the kill-switch fired while the stop was being placed
                    await self.close_position(sym, "Kill-switch emir iletimi sırasında tetiklendi — pozisyon hemen kapatıldı")
                    return
            prob = row.get("hmm_prob") if cfg.regime.method == "hmm" else None
            self.notifier.notify(msg.entry_message(
                pos, fill, d, sz.notes, self.equity(), self.cash(), self.account.starting_balance,
                len(self.account.positions), mode=cfg.mode_label, hmm_prob=prob))

    async def close_position(self, sym: str, reason: str) -> None:
        pos = self.account.positions.get(sym)
        if pos is None:
            return
        ref = self.broker.reference_price(sym)
        if self.live:
            try:
                fill = await self.futures.close_position(sym, pos.side, pos.qty, ref)
            except Exception as exc:
                first = sym not in self.close_retry
                self.close_retry[sym] = reason
                guard = ("Borsadaki stop emri pozisyonu korumaya devam ediyor" if self._exchange_protected(sym) else
                         "DİKKAT: borsada stop emri YOK, stop yalnızca bot tarafından izleniyor")
                if first:
                    self.alert("Pozisyon kapatılamadı", f"{fmt.pair(sym)}: {type(exc).__name__}: {exc}. {guard}; "
                               "kapatma her senkronda yeniden denenecek.")
                else:
                    log.warning("%s yine kapatılamadı: %s", sym, exc)
                return
            finally:
                self._broker_warnings()
            if fill is None:
                try:
                    fill = await self.futures.last_close_fill(sym, _ms(pos.entry_time), pos.side)
                except Exception as exc:  # the position is flat either way: record it rather than retry forever
                    log.warning("%s kapanış dolumu okunamadı: %s", sym, exc)
                    fill = None
                reason = f"{reason} (pozisyon borsada zaten kapanmıştı)"
                if fill is None:
                    fill = self._synthetic_fill(sym, -pos.side, pos.qty)
            await self._cancel_flat(sym)
            try:
                await self.refresh_balance()
            except Exception as exc:
                log.warning("Bakiye güncellenemedi: %s", exc)
        else:
            fill = await self.broker.market_order(sym, -pos.side, pos.qty, ref, self.last_atr.get(sym, 0.0),
                                                  self._quote_volume(sym))
        await self._record_close(sym, fill, reason)

    async def check_guards(self) -> None:
        for ev, text in self.risk.update_guards(self.account, self.equity(), _now()):
            self.dirty = True
            if ev == "KILL_SWITCH":
                self.alert("KILL-SWITCH — işlemler durduruldu",
                           f"{text}. Tüm pozisyonlar kapatılıyor. Yeniden başlatmak için botu durdurup: "
                           "python -m cappatrader resume", ev)
                for s in list(self.account.positions):
                    self._spawn(s, self.close_position(s, f"Kill-switch: {text}"))
            else:
                self.alert("Günlük zarar limiti", text, ev)

    # -- background loops ----------------------------------------------------------
    async def watchdog(self) -> None:
        while True:
            await asyncio.sleep(5)
            now = time.monotonic()
            stale = [s for s in self.symbols if now - self.last_seen.get(s, 0.0) > self.cfg.data.stale_after_s]
            if stale and not self.protection:
                self.protection = True
                self.alert("Veri akışı gecikti",
                           f"{', '.join(fmt.pair(s) for s in stale)} için {self.cfg.data.stale_after_s:.0f} sn veri yok. "
                           "Koruma modu: yeni işlem yok, stoplar REST ile izleniyor.", "PROTECTION")
            elif not stale and self.protection and self.ws_connected:
                self.protection = False
                self.alert("Veri akışı normale döndü", "Koruma modu kapatıldı, işlemler devam ediyor.", "PROTECTION")
            if self.protection:
                for s in [s for s in self.symbols if s in self.account.positions]:
                    try:
                        await self.on_price(s, await self.rest.ticker_price(s), from_stream=False)
                    except Exception as exc:
                        log.warning("%s REST fiyatı alınamadı: %s", s, exc)

    async def periodic(self) -> None:
        cfg = self.cfg
        while True:
            await asyncio.sleep(10)
            t = time.time()
            if self.live and t - self._last_sync >= cfg.live.sync_interval_s:
                self._last_sync = t
                try:
                    await self.refresh_balance()
                    await self.reconcile_positions()
                    await self._retry_pending()
                except Exception as exc:
                    log.warning("Borsa senkronizasyonu başarısız: %s", exc)
            await self.check_guards()
            self._flatten_if_halted()
            if self.dirty:
                self.store.save_account(self.account)
                self.dirty = False
            if self.alt is not None and t - self._last_alt >= cfg.data.alt_refresh_min * 60:
                self._last_alt = t
                try:
                    await self.alt.refresh(await self.http())
                except Exception as exc:
                    log.warning("Alternatif veri güncellenemedi: %s", exc)
            if t - self._last_equity_rec >= cfg.bar_ms / 1000:
                self._last_equity_rec = t
                self.store.record_equity(_now().isoformat(), self.equity(), self.cash())
            if t - self._last_heartbeat >= 900:
                self._last_heartbeat = t
                log.info("Kalp atışı: bakiye %.2f USDT, nakit %.2f, açık %d, koruma=%s",
                         self.equity(), self.cash(), len(self.account.positions), self.protection)
            now = _now()
            today = now.strftime("%Y-%m-%d")
            if self._summary_day is None:
                self._summary_day = self.store.get_kv("summary_day")
            if now.hour == cfg.telegram.daily_summary_hour_utc and self._summary_day != today:
                self._summary_day = today
                self.store.set_kv("summary_day", today)
                day_trades = [x for x in self.account.trades if x.exit_time.startswith(today)]
                self.notifier.notify(msg.daily_summary(
                    today, self.equity(), self.cash(), self.account.starting_balance,
                    self.account.day_start_equity, self.account.positions, self.last_price, day_trades,
                    mode=cfg.mode_label))

    # -- lifecycle ------------------------------------------------------------------
    def _startup_text(self) -> str:
        sq = self.cfg.strategy.squeeze
        if is_squeeze(self.cfg):
            strat = (f"Strateji: Squeeze Momentum + VWAP (stop {sq.atr_stop_mult:g}×ATR, hedef {sq.take_profit_r:g}R, "
                     f"momentum dönüşünde erken çıkış)")
        else:
            strat = "Strateji: rejim ağırlıklı ensemble"
        regime = ("HMM" if self.cfg.regime.method == "hmm" else "ADX") + " rejim · " + (
            "GARCH volatilite" if self.cfg.regime.vol_method == "garch" else "ATR volatilite")
        lines = [strat, regime]
        if self.binance_balance_note:
            lines.append(self.binance_balance_note)
        return "\n".join(lines)

    async def run(self) -> None:
        self.stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.stop_event.set)
            except (NotImplementedError, RuntimeError):
                pass
        self.lock.acquire()
        await self.notifier.start()
        boot = asyncio.create_task(self.bootstrap(), name="bootstrap")
        stop_wait = asyncio.create_task(self.stop_event.wait())
        await asyncio.wait([boot, stop_wait], return_when=asyncio.FIRST_COMPLETED)
        if not boot.done():  # Ctrl+C / SIGTERM while still loading history and fitting models
            boot.cancel()
            await asyncio.gather(boot, return_exceptions=True)
            log.info("Başlatma sırasında durduruldu")
            await self.shutdown(send_message=False)
            return
        stop_wait.cancel()
        try:
            boot.result()
        except Exception as exc:
            self.notifier.notify(msg.alert_message(f"Başlatma hatası · {self.cfg.mode_label}",
                                                   f"{type(exc).__name__}: {exc}"))
            await self.shutdown(send_message=False)
            raise
        if self.cfg.telegram.send_startup:
            self.notifier.notify(msg.startup_message(
                self.symbols, self.cfg.timeframe, self.equity(), self.cash(), self.account.starting_balance,
                self.account.positions, self.warnings, mode=self.cfg.mode_label, extra=self._startup_text()))
        for w in self.warnings:
            log.warning(w)
        ws_base = self.cfg.exchange.futures_ws_base if self.market == "futures" else self.cfg.exchange.ws_base
        stream = BinanceStream(ws_base, market_streams(self.symbols, self.cfg.timeframe), self.on_stream,
                               self.on_ws_state)
        loops = [asyncio.create_task(stream.run(self.stop_event), name="ws"),
                 asyncio.create_task(self.watchdog(), name="watchdog"),
                 asyncio.create_task(self.periodic(), name="periodic")]
        stopper = asyncio.create_task(self.stop_event.wait())
        done, _ = await asyncio.wait([stopper, *loops], return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            if t is not stopper and not t.cancelled() and t.exception() is not None:
                self.crashed = True
                log.error("Arka plan görevi çöktü: %s", t.get_name(), exc_info=t.exception())
                self.notifier.notify(msg.alert_message(
                    f"Bot beklenmedik şekilde durdu · {self.cfg.mode_label}",
                    f"'{t.get_name()}' görevi çöktü: {type(t.exception()).__name__}: {t.exception()}. "
                    + self._protection_text() + " "
                    + "Botu yeniden başlatın."))
        self.stop_event.set()
        stopper.cancel()
        for t in loops:
            t.cancel()
        await asyncio.gather(*loops, stopper, return_exceptions=True)
        await self.shutdown()

    async def shutdown(self, send_message: bool = True) -> None:
        """Best effort: every step runs even if an earlier one fails."""
        unfinished = 0
        if self.tasks:  # orders in flight must finish (fill -> stop placed); docker compose grants 30 s
            try:
                _, pending = await asyncio.wait(list(self.tasks), timeout=25)
                unfinished = len(pending)
            except Exception as exc:
                log.error("Açık emir görevleri beklenemedi: %s", exc)
        try:
            self.store.save_account(self.account)
        except Exception as exc:
            log.error("Hesap kaydedilemedi: %s", exc)
        if send_message and not self.crashed:
            self.notifier.notify(msg.alert_message(
                f"CappaTrader durduruldu · {self.cfg.mode_label}",
                f"Bakiye {self.equity():,.2f} USDT, açık pozisyon {len(self.account.positions)} "
                "(pozisyonlar kayıtlı; yeniden başlatınca takip devam eder)."
                + (f" {self._protection_text()}" if self.account.positions else "")
                + (f" DİKKAT: {unfinished} emir görevi bitmeden durduruldu; borsadaki pozisyonları kontrol edin."
                   if unfinished else "")))
        for name, closer in (("telegram", self.notifier.stop), ("rest", self.rest.close)):
            try:
                await closer()
            except Exception as exc:
                log.error("Kapatma adımı başarısız (%s): %s", name, exc)
        if self.futures is not None:
            try:
                await self.futures.close()
            except Exception as exc:
                log.error("Futures bağlantısı kapatılamadı: %s", exc)
        if self._http is not None and not self._http.closed:
            await self._http.close()
        try:
            self.store.close()
        except Exception:
            pass
        self.lock.release()
