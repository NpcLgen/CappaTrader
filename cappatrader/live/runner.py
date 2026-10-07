"""Live paper-trading engine.

WebSocket ticks/depth/klines -> cleaning -> (on bar close) pipeline + ML + order
flow + sentiment -> decision -> risk -> paper broker -> SQLite + Telegram.
Stops, trailing stops and the flash-move guard run on every tick.
"""

from __future__ import annotations

import asyncio
import logging
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
from cappatrader.live.state import StateStore
from cappatrader.models import LONG, Decision
from cappatrader.notify import messages as msg
from cappatrader.notify.telegram import TelegramNotifier
from cappatrader.risk.manager import RiskManager
from cappatrader.risk.sizing import size_position
from cappatrader.risk.stops import stop_reason, update_trailing
from cappatrader.signals.ml import OnlineModel
from cappatrader.strategy.decision import DecisionEngine, FrameView
from cappatrader.strategy.explain import explain
from cappatrader.strategy.pipeline import prepare_universe

log = logging.getLogger(__name__)


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


class LiveEngine:
    def __init__(self, cfg: Config, rest: BinanceREST | None = None, notifier: TelegramNotifier | None = None,
                 store: StateStore | None = None, http: aiohttp.ClientSession | None = None):
        self.cfg = cfg
        self.rest = rest or BinanceREST(cfg.exchange, cfg.secrets)
        self.notifier = notifier or TelegramNotifier(
            cfg.secrets.telegram_token, cfg.secrets.telegram_chat_id,
            enabled=cfg.telegram.enabled, min_interval_s=cfg.telegram.min_interval_s)
        self.store = store or StateStore(cfg.runtime.state_db)
        self.account = self.store.load_account(cfg.account.starting_balance)
        self.costs = CostModel(cfg.costs, cfg.bar_ms)
        self.risk = RiskManager(cfg.risk)
        self.decider = DecisionEngine(cfg)
        self.symbols = list(cfg.symbols)
        self.bars: dict[str, pd.DataFrame] = {}
        self.books = {s: OrderBook(s) for s in self.symbols}
        self.flow = {s: TradeFlow() for s in self.symbols}
        self.spoof = {s: SpoofDetector() for s in self.symbols}
        self.ticks = {s: TickFilter(max_dev=cfg.data.tick_max_deviation) for s in self.symbols}
        self.flash = {s: deque() for s in self.symbols}
        self.ml = {s: OnlineModel(cfg.signals.ml) for s in self.symbols}
        self.last_price: dict[str, float] = {}
        self.last_seen: dict[str, float] = {}
        self.last_atr: dict[str, float] = {}
        self.filters: dict[str, dict] = {}
        self.alt = AltDataHub(self.rest, cfg.data.news_feeds, self.symbols) if cfg.data.alt_data_enabled else None
        self.broker = PaperBroker(self.costs, self.books, self.last_price)
        self.entry_lock = asyncio.Lock()
        self.inflight: set[str] = set()
        self.tasks: set[asyncio.Task] = set()
        self.protection = False
        self.warnings: list[str] = []
        self.dirty = False
        self.max_bars = max(cfg.runtime.history_bars, cfg.signals.ml.train_bars + 300)
        self.stop_event: asyncio.Event | None = None
        self._http = http
        self._last_alt = 0.0
        self._last_equity_rec = 0.0
        self._last_heartbeat = 0.0
        self._summary_day: str | None = None

    # -- helpers -----------------------------------------------------------
    def equity(self) -> float:
        return self.account.equity(self.last_price)

    def alert(self, title: str, text: str, kind: str = "ALERT") -> None:
        log.warning("%s: %s", title, text)
        self.store.record_event(_now().isoformat(), kind, f"{title}: {text}")
        self.notifier.notify(msg.alert_message(title, text))

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

        task.add_done_callback(done)

    def _quote_volume(self, sym: str) -> float | None:
        df = self.bars.get(sym)
        if df is None or df.empty or "quote_volume" not in df:
            return None
        return float(df["quote_volume"].iloc[-1])

    # -- bootstrap -----------------------------------------------------------
    async def bootstrap(self) -> None:
        cfg = self.cfg
        try:
            self.filters = await self.rest.exchange_filters(self.symbols)
        except Exception as exc:  # filters only refine rounding; keep going
            self.warnings.append(f"Borsa filtreleri alınamadı ({type(exc).__name__}); miktar yuvarlama kapalı")
        for s in self.symbols:
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
            self.last_atr[s] = float(atr_ind(df["high"], df["low"], df["close"], cfg.regime.atr_period).iloc[-1])
            log.info("%s: %d bar yüklendi (%s)", s, len(df), rep)
        if cfg.signals.ml.enabled:
            for s in self.symbols:
                ok = await asyncio.to_thread(self.ml[s].fit, self.bars[s])
                log.info("%s ML modeli %s", s, "eğitildi" if ok else "eğitilemedi (yetersiz veri)")
        await self._check_api_key()
        if self.alt is not None:
            await self.alt.refresh(await self.http())
            self._last_alt = time.time()

    async def _check_api_key(self) -> None:
        if not self.cfg.secrets.has_binance:
            log.info("Binance API anahtarı yok: paper modda gerekli değil")
            return
        try:
            r = await self.rest.api_restrictions()
        except Exception as exc:
            self.warnings.append(f"API anahtarı yetki kontrolü yapılamadı: {exc}")
            return
        if r.get("enableWithdrawals"):
            self.warnings.append("API anahtarında PARA ÇEKME yetkisi AÇIK — Binance'te hemen kapatın (PRD güvenlik)")
        if not r.get("ipRestrict"):
            self.warnings.append("API anahtarında IP kısıtlaması yok — sunucu IP'sine kısıtlamanız önerilir")
        if r.get("enableSpotAndMarginTrading") or r.get("enableFutures"):
            self.warnings.append("API anahtarında işlem yetkisi açık; paper mod için yalnızca okuma yetkisi yeterli")

    async def http(self) -> aiohttp.ClientSession:
        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession(trust_env=True)
        return self._http

    # -- stream handlers -----------------------------------------------------
    async def on_stream(self, stream: str, data: dict) -> None:
        if "@" not in stream:
            return
        sym_l, kind = stream.split("@", 1)
        sym = sym_l.upper()
        if sym not in self.books:
            return
        if kind.startswith("kline"):
            await self.on_kline(sym, data["k"])
        elif kind == "aggTrade":
            self.flow[sym].add(time.monotonic(), float(data["p"]), float(data["q"]), bool(data.get("m")))
            await self.on_price(sym, float(data["p"]))
        elif kind.startswith("depth"):
            now = time.monotonic()
            self.books[sym].update(data.get("bids", []), data.get("asks", []), data.get("lastUpdateId", 0), now)
            self.spoof[sym].on_book(self.books[sym], now)
            self.last_seen[sym] = now

    async def on_ws_state(self, connected: bool) -> None:
        if not connected and not self.protection:
            self.protection = True
            self.alert("Veri bağlantısı koptu",
                       "WebSocket bağlantısı kesildi. Koruma modu: yeni işlem açılmıyor, açık pozisyonların stopları "
                       "REST fiyatıyla izleniyor. Otomatik yeniden bağlanılıyor.", "PROTECTION")
        elif connected:
            for s in self.symbols:
                await self._backfill(s)

    async def on_price(self, sym: str, price: float) -> None:
        now = time.monotonic()
        if not self.ticks[sym].accept(price):
            return
        self.last_price[sym] = price
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
        if update_trailing(pos, price, price, self.last_atr.get(sym, 0.0), self.cfg.risk):
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
    async def on_bar_close(self, sym: str, ts: pd.Timestamp) -> None:
        cfg = self.cfg
        partners = {b for a, b in cfg.pairs if a == sym} | {a for a, b in cfg.pairs if b == sym}
        data = {s: self.bars[s] for s in {sym} | partners if s in self.bars}
        bars = self.bars[sym]
        proba = None
        if cfg.signals.ml.enabled:
            model = self.ml[sym]
            model.bars_since_fit += 1
            if model.needs_fit():
                await asyncio.to_thread(model.fit, bars)
            p = model.predict_last(bars)
            proba = pd.Series(np.nan, index=bars.index)
            proba.iloc[-1] = p
        now = time.monotonic()
        of_score, of_note = orderflow_signal(self.books[sym], self.flow[sym], self.spoof[sym], now, fmt.base_asset(sym))
        sent_score, sent_note = self.alt.sentiment(sym) if self.alt is not None else (None, "")
        frames = await asyncio.to_thread(
            prepare_universe, data, cfg, {sym: proba} if proba is not None else None,
            {sym: {"orderflow": of_score, "sentiment": sent_score}})
        f = frames[sym]
        v = FrameView(f)
        i = len(f) - 1
        atr_now = float(v["atr"][i])
        if np.isfinite(atr_now):
            self.last_atr[sym] = atr_now
        pos = self.account.positions.get(sym)
        if pos is not None:
            pos.bars_held += 1
            update_trailing(pos, float(v["high"][i]), float(v["low"][i]), self.last_atr[sym], cfg.risk)
            self.dirty = True
        last_exit = self.account.last_exit.get(sym)
        bse = int((ts - pd.Timestamp(last_exit)) / pd.Timedelta(milliseconds=cfg.bar_ms)) if last_exit else None
        d = self.decider.evaluate(sym, v, i, pos, bse)
        log.info("%s bar %s | rejim %s/%s | skor %+.2f | %s", sym, ts, d.regime, d.vol_state, d.composite, d.action)
        if d.action == "EXIT":
            self._spawn(sym, self.close_position(sym, d.exit_reason))
        elif d.action in ("ENTER_LONG", "ENTER_SHORT"):
            ok, why = self.risk.entry_allowed(self.account, sym, _now(), len(self.inflight - {sym}))
            if self.protection:
                ok, why = False, "Koruma modu (veri akışı sorunlu)"
            if ok:
                notes = {"orderflow": of_note, "sentiment": sent_note}
                self._spawn(sym, self.open_position(sym, d, f.iloc[i], notes))
            else:
                log.info("%s giriş sinyali reddedildi: %s", sym, why)
        await self.check_guards()
        self.store.save_account(self.account)

    async def open_position(self, sym: str, d: Decision, row: pd.Series, notes: dict[str, str]) -> None:
        cfg = self.cfg
        async with self.entry_lock:
            if sym in self.account.positions or self.account.halted:
                return
            ref = self.broker.reference_price(sym)
            eq = self.equity()
            qv = self._quote_volume(sym)
            flt = self.filters.get(sym, {})
            sz = size_position(cfg.risk, eq, self.account.cash, ref, d.atr, d.side, d.vol_state, self.account.trades,
                               step_size=flt.get("step_size"), min_notional=flt.get("min_notional"),
                               round_trip_cost_pct=self.costs.round_trip_pct(ref, d.atr, 0.1 * eq, qv))
            if not sz.ok:
                log.info("%s pozisyon açılmadı: %s", sym, sz.rejected)
                return
            fill = await self.broker.market_order(sym, d.side, sz.qty, ref, d.atr, qv)
            if fill.notional + fill.fee > self.account.cash:
                log.warning("%s: dolum sonrası nakit yetersiz, emir iptal", sym)
                return
            reasons = explain(row, d.side, cfg, notes)
            pos = build_position(d, fill, sz, cfg.risk, reasons, _now().isoformat())
            self.account.open_position(pos, fill)
            self.store.save_account(self.account)
            self.store.record_event(_now().isoformat(), "ENTRY", f"{sym} {d.action} {fill.qty}@{fill.price}")
            self.notifier.notify(msg.entry_message(
                pos, fill, d, sz.notes, self.equity(), self.account.cash, self.account.starting_balance,
                len(self.account.positions)))

    async def close_position(self, sym: str, reason: str) -> None:
        pos = self.account.positions.get(sym)
        if pos is None:
            return
        ref = self.broker.reference_price(sym)
        fill = await self.broker.market_order(sym, -pos.side, pos.qty, ref, self.last_atr.get(sym, 0.0),
                                              self._quote_volume(sym))
        if sym not in self.account.positions:
            return
        trade = self.account.close_position(sym, fill, reason, _now().isoformat())
        self.store.record_trade(trade)
        self.store.save_account(self.account)
        self.notifier.notify(msg.exit_message(
            trade, fill, self.equity(), self.account.cash, self.account.starting_balance,
            len(self.account.positions), self.account.trades))

    async def check_guards(self) -> None:
        for ev, text in self.risk.update_guards(self.account, self.equity(), _now()):
            self.dirty = True
            if ev == "KILL_SWITCH":
                self.alert("KILL-SWITCH — işlemler durduruldu",
                           f"{text}. Tüm pozisyonlar kapatılıyor. Yeniden başlatmak için: python -m cappatrader resume",
                           ev)
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
            elif not stale and self.protection:
                self.protection = False
                self.alert("Veri akışı normale döndü", "Koruma modu kapatıldı, işlemler devam ediyor.", "PROTECTION")
            if self.protection:
                for s in [s for s in stale if s in self.account.positions]:
                    try:
                        await self.on_price(s, await self.rest.ticker_price(s))
                    except Exception as exc:
                        log.warning("%s REST fiyatı alınamadı: %s", s, exc)

    async def periodic(self) -> None:
        cfg = self.cfg
        while True:
            await asyncio.sleep(10)
            t = time.time()
            await self.check_guards()
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
                self.store.record_equity(_now().isoformat(), self.equity(), self.account.cash)
            if t - self._last_heartbeat >= 900:
                self._last_heartbeat = t
                log.info("Kalp atışı: bakiye %.2f USDT, nakit %.2f, açık %d, koruma=%s",
                         self.equity(), self.account.cash, len(self.account.positions), self.protection)
            now = _now()
            today = now.strftime("%Y-%m-%d")
            if now.hour == cfg.telegram.daily_summary_hour_utc and self._summary_day != today:
                self._summary_day = today
                day_trades = [x for x in self.account.trades if x.exit_time.startswith(today)]
                self.notifier.notify(msg.daily_summary(
                    today, self.equity(), self.account.cash, self.account.starting_balance,
                    self.account.day_start_equity, self.account.positions, self.last_price, day_trades))

    # -- lifecycle ------------------------------------------------------------------
    async def run(self) -> None:
        self.stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.stop_event.set)
            except (NotImplementedError, RuntimeError):
                pass
        await self.notifier.start()
        try:
            await self.bootstrap()
        except Exception as exc:
            self.notifier.notify(msg.alert_message("Başlatma hatası", f"{type(exc).__name__}: {exc}"))
            await self.shutdown(send_message=False)
            raise
        if self.cfg.telegram.send_startup:
            self.notifier.notify(msg.startup_message(
                self.symbols, self.cfg.timeframe, self.equity(), self.account.cash, self.account.starting_balance,
                self.account.positions, self.warnings))
        for w in self.warnings:
            log.warning(w)
        stream = BinanceStream(self.cfg.exchange.ws_base, market_streams(self.symbols, self.cfg.timeframe),
                               self.on_stream, self.on_ws_state)
        loops = [asyncio.create_task(stream.run(self.stop_event), name="ws"),
                 asyncio.create_task(self.watchdog(), name="watchdog"),
                 asyncio.create_task(self.periodic(), name="periodic")]
        stopper = asyncio.create_task(self.stop_event.wait())
        done, _ = await asyncio.wait([stopper, *loops], return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            if t is not stopper and t.exception() is not None:
                log.error("Arka plan görevi çöktü: %s", t.get_name(), exc_info=t.exception())
        self.stop_event.set()
        for t in loops:
            t.cancel()
        await asyncio.gather(*loops, return_exceptions=True)
        await self.shutdown()

    async def shutdown(self, send_message: bool = True) -> None:
        if self.tasks:
            await asyncio.wait(list(self.tasks), timeout=10)
        self.store.save_account(self.account)
        if send_message:
            self.notifier.notify(msg.alert_message(
                "CappaTrader durduruldu",
                f"Bakiye {self.equity():,.2f} USDT, açık pozisyon {len(self.account.positions)} "
                "(pozisyonlar kayıtlı; yeniden başlatınca stop takibi devam eder)."))
        await self.notifier.stop()
        await self.rest.close()
        if self._http is not None and not self._http.closed:
            await self._http.close()
        self.store.close()
