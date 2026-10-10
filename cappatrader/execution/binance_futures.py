"""Real order execution and balances on Binance USD-M futures through ccxt.

Safety design (this module moves real money when live.env == "mainnet"):
- The broker refuses to exist outside mode: live, without keys, or on mainnet without the
  verbatim confirmation phrase; the phrase is re-checked before every order.
- Every order carries a client order id ("cappa-..."). After a timeout or any other ambiguous
  failure the order is looked up by that id before anything is re-sent, so a retry can never
  double a position. Errors that cannot have executed an order (insufficient margin, invalid
  order, authentication) are never retried.
- Only one-way position mode is supported; hedge mode is refused at connect().
- Stops and take-profits are placed on the exchange (reduce-only conditional algo orders), so
  open positions stay protected if the bot dies.
- Messages are Turkish (shown in Telegram) and never contain the API key or secret.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Any

import ccxt
import pandas as pd

from cappatrader import fmt
from cappatrader.config import MAINNET_CONFIRM_PHRASE, Config
from cappatrader.models import LONG, SHORT, Fill

log = logging.getLogger(__name__)

CLIENT_ID_PREFIX = "cappa-"
_WEEK_MS = 7 * 86_400_000
_FILL_POLLS = 4
_TERMINAL = ("closed", "canceled", "expired", "rejected")
_CODE_RE = re.compile(r'"code"\s*:\s*"?(-?\d+)')
_SIGNATURE_RE = re.compile(r"signature=[0-9a-fA-F]+")

HEDGE_MODE_MESSAGE = (
    "Binance vadeli hesabınız Hedge (çift yönlü) pozisyon modunda. CappaTrader yalnızca Tek Yönlü "
    "(One-way) modda çalışır: Binance Futures > Ayarlar > Pozisyon Modu > 'Tek Yönlü Mod' seçin "
    "(açık pozisyon ve emir yokken), sonra botu yeniden başlatın."
)


class FuturesError(RuntimeError):
    """Base error of the futures broker; the message is Turkish and safe to show to the user."""


class LiveTradingRefused(FuturesError):
    """Configuration or account state does not allow real order execution."""


class OrderRejected(FuturesError):
    """The order was not executed: a pre-trade check failed or the exchange rejected it.

    code: min_qty | max_qty | min_notional | invalid | unavailable | insufficient_funds |
          would_trigger | no_position | not_filled | network | rejected
    """

    def __init__(self, message: str, code: str = "rejected"):
        super().__init__(message)
        self.code = code


class OrderStateUnknown(FuturesError):
    """The order may or may not have executed; reconcile with positions() before acting again."""

    def __init__(self, message: str, client_id: str):
        super().__init__(message)
        self.client_id = client_id


@dataclass
class FuturesBalance:
    wallet: float
    available: float
    unrealized: float
    equity: float


@dataclass
class ExchangePosition:
    symbol: str
    side: int
    qty: float
    entry_price: float
    unrealized: float
    leverage: float


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


def _backoff(attempt: int) -> float:
    return float(2 ** attempt)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _codes(exc: BaseException) -> set[int]:
    return {int(c) for c in _CODE_RE.findall(str(exc))}


def _has_code(exc: BaseException, *codes: int) -> bool:
    return bool(_codes(exc) & set(codes))


def _is_transient(exc: BaseException) -> bool:
    """Retryable: network/server-side failures and clock drift. Never margin, order or auth errors."""
    if not isinstance(exc, ccxt.BaseError):
        return False
    if _has_code(exc, -2023):  # account in liquidation mode
        return False
    return isinstance(exc, ccxt.OperationFailed) or _is_clock_error(exc)


def _is_clock_error(exc: BaseException) -> bool:
    return isinstance(exc, ccxt.InvalidNonce) or _has_code(exc, -1021, -5028)


def _no_change(exc: BaseException) -> bool:
    return (isinstance(exc, ccxt.NoChange) or _has_code(exc, -4046, -4059)
            or "no need to change" in str(exc).lower())


def _nothing_to_cancel(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (isinstance(exc, ccxt.OrderNotFound) or _has_code(exc, -2011, -2013, -20124)
            or "unknown order" in text or "does not exist" in text or "no open order" in text)


def _order_side(side: int) -> str:
    if side == LONG:
        return "buy"
    if side == SHORT:
        return "sell"
    raise OrderRejected(f"Geçersiz emir yönü: {side} (+1 alış, -1 satış olmalı)", "invalid")


def client_order_id(symbol: str, side: int, tag: str, ts_ms: int, seq: int = 0) -> str:
    """Deterministic Binance client order id (<= 36 chars, [A-Za-z0-9-]) for one logical order."""
    digest = hashlib.sha1(f"{symbol}|{side}|{tag}|{ts_ms}|{seq}".encode()).hexdigest()[:16]
    return f"{CLIENT_ID_PREFIX}{tag}-{digest}"


def build_exchange(cfg: Config) -> Any:
    """ccxt USD-M client from cfg. No network; demo endpoints are switched on by the caller."""
    import ccxt.async_support as ccxt_async

    return ccxt_async.binanceusdm({
        "apiKey": cfg.secrets.binance_api_key,
        "secret": cfg.secrets.binance_secret_key,
        "enableRateLimit": True,
        "timeout": int(cfg.exchange.request_timeout_s * 1000),
        "options": {
            "defaultType": "future",
            "adjustForTimeDifference": True,
            "recvWindow": int(cfg.exchange.recv_window_ms),
            # Currencies come from a spot sapi endpoint we do not need (and demo trading lacks).
            "fetchCurrencies": False,
        },
    })


def parse_futures_balance(raw: dict, quote: str = "USDT") -> FuturesBalance:
    """USD-M account (ccxt fetch_balance on fapi v3 /account) -> wallet/available/unrealized/equity."""
    info = raw.get("info") if isinstance(raw.get("info"), dict) else {}
    row = next((a for a in info.get("assets") or [] if a.get("asset") == quote), None)
    if row is not None:
        wallet = _f(row.get("walletBalance"))
        unrealized = _f(row.get("unrealizedProfit"))
        available = _f(row.get("availableBalance"))
    else:
        acct = raw.get(quote) or {}
        wallet, unrealized, available = _f(acct.get("total")), 0.0, _f(acct.get("free"))
    return FuturesBalance(wallet=wallet, available=available, unrealized=unrealized, equity=wallet + unrealized)


def parse_spot_balance(raw: dict, quote: str = "USDT") -> float:
    acct = raw.get(quote) or {}
    return _f(acct.get("free")) + _f(acct.get("used"))


def _scrub(text: str, secrets: tuple[str, ...]) -> str:
    for s in secrets:
        if s:
            text = text.replace(s, "***")
    return _SIGNATURE_RE.sub("signature=***", text)


def _aggregate(trades: list[dict]) -> tuple[float, float, int | None]:
    qty = sum(_f(t.get("amount")) for t in trades)
    if qty <= 0:
        return 0.0, 0.0, None
    avg = sum(_f(t.get("price")) * _f(t.get("amount")) for t in trades) / qty
    stamps = [int(t["timestamp"]) for t in trades if t.get("timestamp") is not None]
    return qty, avg, max(stamps) if stamps else None


def _fee_items(obj: dict) -> list[dict]:
    fees = [f for f in obj.get("fees") or [] if f]
    if not fees and obj.get("fee"):
        fees = [obj["fee"]]
    return [f for f in fees if f.get("cost") is not None]


def _timestamp(ms: Any) -> pd.Timestamp:
    if ms is None:
        return pd.Timestamp.now(tz="UTC")
    return pd.Timestamp(int(ms), unit="ms", tz="UTC")


class BinanceAccountReader:
    """Read-only USDT balances (futures and spot). Needs only read permission, works in any mode."""

    def __init__(self, cfg: Config, exchange: Any = None, env: str | None = None):
        self.cfg = cfg
        self.env = env or (cfg.live.env if cfg.mode == "live" else "mainnet")
        if self.env not in ("demo", "mainnet"):
            raise LiveTradingRefused("live.env: 'demo' veya 'mainnet' olmalı")
        if exchange is None and not cfg.secrets.has_binance:
            raise LiveTradingRefused("Binance hesabına erişmek için .env içinde BINANCE_API_KEY ve "
                                     "BINANCE_SECRET_KEY tanımlı olmalı.")
        self.quote = cfg.account.quote_asset
        self.ex = exchange
        self._demo_on = False

    # -- client ------------------------------------------------------------------
    def _client(self) -> Any:
        if self.ex is None:
            self.ex = build_exchange(self.cfg)
        if self.env == "demo" and not self._demo_on:
            # demo-fapi.binance.com (keys from demo.binance.com); the old futures testnet is deprecated in ccxt
            if not (getattr(self.ex, "options", None) or {}).get("enableDemoTrading"):
                self.ex.enable_demo_trading(True)
            self._demo_on = True
        self._check_endpoint()
        return self.ex

    def _check_endpoint(self) -> None:
        urls = getattr(self.ex, "urls", None)
        api = urls.get("api") if isinstance(urls, dict) else None
        fapi = api.get("fapiPrivate") if isinstance(api, dict) else None
        if not isinstance(fapi, str):
            return
        is_demo = "demo-fapi" in fapi or "testnet" in fapi
        if self.env == "demo" and not is_demo:
            raise LiveTradingRefused("live.env=demo ama ccxt istemcisi gerçek (mainnet) uç noktasına bağlı; "
                                     "güvenlik için işlem yapılmıyor.")
        if self.env == "mainnet" and is_demo:
            raise LiveTradingRefused("live.env=mainnet ama ccxt istemcisi demo/testnet uç noktasına bağlı.")

    def _secrets(self) -> tuple[str, ...]:
        return (self.cfg.secrets.binance_api_key, self.cfg.secrets.binance_secret_key)

    async def close(self) -> None:
        if self.ex is not None and hasattr(self.ex, "close"):
            await self.ex.close()

    # -- request helpers -----------------------------------------------------------
    async def _resync_clock(self) -> None:
        if hasattr(self.ex, "load_time_difference"):
            try:
                await self.ex.load_time_difference()
            except ccxt.BaseError as e:
                log.warning("Binance saat farkı güncellenemedi: %s", _scrub(str(e), self._secrets()))

    async def _read(self, fn, *args, **kwargs):
        """Idempotent request with exponential backoff on transient errors; raises raw ccxt errors."""
        retries = max(0, int(self.cfg.live.max_retries))
        for attempt in range(retries + 1):
            try:
                return await fn(*args, **kwargs)
            except ccxt.BaseError as e:
                if not _is_transient(e) or attempt == retries:
                    raise
                if _is_clock_error(e):
                    await self._resync_clock()
                log.warning("Binance geçici hata (%s), %.0f sn sonra tekrar denenecek: %s",
                            getattr(fn, "__name__", "istek"), _backoff(attempt), _scrub(str(e), self._secrets()))
                await _sleep(_backoff(attempt))

    async def _call(self, fn, *args, **kwargs):
        try:
            return await self._read(fn, *args, **kwargs)
        except ccxt.BaseError as e:
            raise self._translate(e) from e

    def _translate(self, exc: BaseException, symbol: str | None = None) -> FuturesError:
        text = _scrub(str(exc), self._secrets())
        where = f"{fmt.pair(symbol)}: " if symbol else ""
        if isinstance(exc, ccxt.InsufficientFunds):
            return OrderRejected(f"{where}Binance teminat/bakiye yetersiz diyor ({text})", "insufficient_funds")
        if isinstance(exc, ccxt.OrderImmediatelyFillable):
            return OrderRejected(f"{where}tetik fiyatı mevcut fiyatın ötesinde, emir hemen tetiklenirdi ({text})",
                                 "would_trigger")
        if _has_code(exc, -2022) or "reduceonly order is rejected" in text.lower():
            return OrderRejected(f"{where}azaltılacak pozisyon yok, reduce-only emir reddedildi ({text})",
                                 "no_position")
        if isinstance(exc, ccxt.AuthenticationError):
            return LiveTradingRefused(f"{where}Binance API anahtarı reddedildi veya yetkisi yok (vadeli işlem "
                                      f"izni, IP kısıtlaması?): {text}")
        if isinstance(exc, (ccxt.InvalidOrder, ccxt.BadRequest, ccxt.OperationRejected)):
            return OrderRejected(f"{where}Binance emri reddetti: {text}", "rejected")
        if _is_transient(exc):
            return FuturesError(f"{where}Binance'e ulaşılamadı (geçici hata): {text}")
        return FuturesError(f"{where}Binance hatası: {text}")

    # -- balances --------------------------------------------------------------------
    async def balance(self) -> FuturesBalance:
        """USD-M futures USDT balance: wallet, available, unrealized PnL, equity = wallet + unrealized."""
        raw = await self._call(self._client().fetch_balance)
        return parse_futures_balance(raw, self.quote)

    async def spot_usdt_balance(self) -> float:
        """Spot wallet USDT, free + locked."""
        # binanceusdm's defaultSubType "linear" would route type=spot to the futures account;
        # a non-linear subType is dropped by handle_sub_type_and_params, so /api/v3/account is used.
        raw = await self._call(self._client().fetch_balance, {"type": "spot", "subType": "spot"})
        return parse_spot_balance(raw, self.quote)


async def fetch_binance_balance(cfg: Config, exchange: Any = None, env: str | None = None) -> float:
    """Real USDT balance used to seed a paper account: futures equity or spot free+locked.

    Read-only (a key with only "Enable Reading" is enough); works whatever cfg.mode is. In paper mode
    the real (mainnet) account is read unless env is given.
    """
    reader = BinanceAccountReader(cfg, exchange, env)
    try:
        if cfg.account.binance_wallet == "spot":
            return await reader.spot_usdt_balance()
        return (await reader.balance()).equity
    finally:
        await reader.close()


class BinanceFuturesBroker(BinanceAccountReader):
    """Market orders, exchange-side protection and position sync on Binance USD-M perpetuals."""

    def __init__(self, cfg: Config, exchange: Any = None):
        self._assert_allowed(cfg)
        super().__init__(cfg, exchange, cfg.live.env)
        self.symbol_map: dict[str, str] = {}
        self.unavailable: dict[str, str] = {}
        self.leverage: dict[str, float] = {}
        self.warnings: list[str] = []
        self.connected = False
        self._seq = 0

    @staticmethod
    def _assert_allowed(cfg: Config) -> None:
        if cfg.mode != "live":
            raise LiveTradingRefused(f"Gerçek emir iletimi yalnızca mode: live iken açılır (şu an: {cfg.mode}).")
        if not cfg.secrets.has_binance:
            raise LiveTradingRefused("Gerçek emir için .env içinde BINANCE_API_KEY ve BINANCE_SECRET_KEY "
                                     "tanımlı olmalı.")
        if cfg.live.env not in ("demo", "mainnet"):
            raise LiveTradingRefused("live.env: 'demo' veya 'mainnet' olmalı")
        if cfg.live.env == "mainnet" and cfg.live.mainnet_confirm != MAINNET_CONFIRM_PHRASE:
            raise LiveTradingRefused("Mainnet GERÇEK PARA ile işlem yapar: live.mainnet_confirm alanına tam olarak "
                                     f"'{MAINNET_CONFIRM_PHRASE}' yazılmadan emir gönderilmez.")
        if not 1 <= int(cfg.live.leverage) <= 20:
            raise LiveTradingRefused("live.leverage 1-20 arasında olmalı")
        if cfg.live.margin_mode not in ("isolated", "cross"):
            raise LiveTradingRefused("live.margin_mode: 'isolated' veya 'cross' olmalı")

    # -- connection ----------------------------------------------------------------
    async def connect(self) -> None:
        """Load markets, map symbols, refuse hedge mode, set margin mode and leverage per symbol."""
        self._assert_allowed(self.cfg)
        self.connected = False
        self.warnings.clear()
        ex = self._client()
        await self._call(ex.load_markets)
        self._map_symbols()
        mode = await self._call(ex.fetch_position_mode)
        if (mode or {}).get("hedged"):
            raise LiveTradingRefused(HEDGE_MODE_MESSAGE)
        for raw, usym in list(self.symbol_map.items()):
            await self._configure_symbol(raw, usym)
        if not self.symbol_map:
            raise LiveTradingRefused("Yapılandırılan sembollerin hiçbiri Binance USD-M vadeli işlemde "
                                     "işlem yapılabilir değil.")
        self.connected = True
        log.info("Binance USD-M (%s) bağlandı: %s", self.env, ", ".join(self.symbol_map.values()))

    def _map_symbols(self) -> None:
        by_id = {m["id"]: m for m in (getattr(self.ex, "markets", None) or {}).values()
                 if m.get("swap") and m.get("linear") and m.get("settle") == self.quote}
        self.symbol_map.clear()
        self.unavailable.clear()
        for raw in self.cfg.symbols:
            market = by_id.get(raw)
            if market is None:
                self._mark_unavailable(raw, "Binance USD-M'de vadesiz (perpetual) kontratı yok "
                                            "(farklı adla listelenmiş olabilir, ör. 1000SHIBUSDT)")
            elif market.get("active") is False:
                self._mark_unavailable(raw, "vadeli kontratı şu an işleme kapalı")
            else:
                self.symbol_map[raw] = market["symbol"]

    def _mark_unavailable(self, raw: str, reason: str) -> None:
        self.symbol_map.pop(raw, None)
        self.unavailable[raw] = reason
        text = f"{fmt.pair(raw)}: {reason}; bu sembolde işlem yapılmayacak."
        self.warnings.append(text)
        log.warning(text)

    async def _configure_symbol(self, raw: str, usym: str) -> None:
        live = self.cfg.live
        try:
            await self._read(self.ex.set_margin_mode, live.margin_mode, usym)
        except ccxt.BaseError as e:
            if isinstance(e, ccxt.AuthenticationError) or _is_transient(e):
                raise self._translate(e, raw) from e
            if _has_code(e, -4168):
                raise LiveTradingRefused("Binance hesabı Multi-Assets modunda; izole marjin kullanılamaz. "
                                         "live.margin_mode: cross yapın veya Tek Varlık (Single-Asset) "
                                         "moduna geçin.") from e
            if _has_code(e, -4047, -4048):
                self._warn(f"{fmt.pair(raw)}: açık pozisyon/emir olduğu için marjin modu {live.margin_mode} "
                           "yapılamadı, mevcut mod korunuyor.")
            elif not _no_change(e):
                self._mark_unavailable(raw, f"marjin modu ayarlanamadı ({_scrub(str(e), self._secrets())})")
                return
        try:
            await self._read(self.ex.set_leverage, int(live.leverage), usym)
            self.leverage[raw] = float(live.leverage)
        except ccxt.BaseError as e:
            if isinstance(e, ccxt.AuthenticationError) or _is_transient(e):
                raise self._translate(e, raw) from e
            if _no_change(e):
                self.leverage[raw] = float(live.leverage)
            elif await self._has_position(usym):
                self._warn(f"{fmt.pair(raw)}: açık pozisyon varken kaldıraç {live.leverage}x yapılamadı, "
                           "mevcut kaldıraç korunuyor.")
            else:
                self._mark_unavailable(raw, f"kaldıraç {live.leverage}x ayarlanamadı "
                                            f"({_scrub(str(e), self._secrets())})")

    def _warn(self, text: str) -> None:
        self.warnings.append(text)
        log.warning(text)

    async def _has_position(self, usym: str) -> bool:
        try:
            return abs(await self._signed_amount(usym)) > 0
        except ccxt.BaseError:
            return False

    # -- helpers ---------------------------------------------------------------------
    def _unified(self, symbol: str) -> str:
        if symbol in self.symbol_map:
            return self.symbol_map[symbol]
        if symbol in self.unavailable:
            raise OrderRejected(f"{fmt.pair(symbol)}: {self.unavailable[symbol]}", "unavailable")
        if not self.connected:
            raise FuturesError("Binance vadeli bağlantısı kurulmadı: önce connect() çağrılmalı.")
        raise OrderRejected(f"{fmt.pair(symbol)} yapılandırılmış semboller arasında değil.", "unavailable")

    def _raw(self, usym: str) -> str | None:
        return next((raw for raw, u in self.symbol_map.items() if u == usym), None)

    def _new_client_id(self, symbol: str, side: int, tag: str) -> str:
        self._seq += 1
        return client_order_id(symbol, side, tag, _now_ms(), self._seq)

    def _amount(self, symbol: str, usym: str, qty: float) -> float:
        """Exchange-valid quantity, truncated to the step size (never above qty); enforces amount limits."""
        q = _f(qty, -1.0)
        if q <= 0:
            raise OrderRejected(f"{fmt.pair(symbol)}: geçersiz miktar {qty}", "invalid")
        q = float(f"{q:.12g}")  # strip float noise (0.0029999999 -> 0.003) before truncating
        market = self.ex.market(usym)
        limits = market.get("limits") or {}
        mins = [_f((limits.get(k) or {}).get("min")) for k in ("amount", "market")]
        maxs = [_f((limits.get(k) or {}).get("max")) for k in ("amount", "market")]
        min_qty = max(mins)
        max_qty = min((m for m in maxs if m > 0), default=0.0)
        try:
            amount = float(self.ex.amount_to_precision(usym, q))
        except ccxt.InvalidOrder:
            amount = 0.0
        if amount > q * (1 + 1e-9):
            raise FuturesError(f"{fmt.pair(symbol)}: miktar yuvarlaması istenenden büyük çıktı ({amount} > {q})")
        if amount <= 0 or amount < min_qty:
            raise OrderRejected(f"{fmt.pair(symbol)}: miktar {fmt.qty(q)}, borsanın en küçük emir miktarının "
                                f"({fmt.qty(min_qty or _f((market.get('precision') or {}).get('amount')))}) "
                                "altında", "min_qty")
        if max_qty and amount > max_qty:
            raise OrderRejected(f"{fmt.pair(symbol)}: miktar {fmt.qty(amount)}, piyasa emri üst sınırının "
                                f"({fmt.qty(max_qty)}) üstünde", "max_qty")
        return amount

    def _check_min_cost(self, symbol: str, usym: str, amount: float, ref_price: float) -> None:
        min_cost = _f(((self.ex.market(usym).get("limits") or {}).get("cost") or {}).get("min"))
        if ref_price <= 0:
            raise OrderRejected(f"{fmt.pair(symbol)}: referans fiyat geçersiz ({ref_price})", "invalid")
        if min_cost and amount * ref_price < min_cost:
            raise OrderRejected(f"{fmt.pair(symbol)}: emir tutarı {fmt.usd(amount * ref_price)}, Binance en küçük "
                                f"emir tutarının ({fmt.usd(min_cost)}) altında", "min_notional")

    def _taker_rate(self, usym: str) -> float:
        rate = _f(self.ex.market(usym).get("taker"), -1.0)
        return rate if rate >= 0 else self.cfg.costs.taker_fee

    async def _signed_amount(self, usym: str) -> float:
        rows = await self._read(self.ex.fetch_positions, [usym])
        total = 0.0
        for p in rows or []:
            if p.get("symbol") == usym:
                total += self._position_amount(p)
        return total

    @staticmethod
    def _position_amount(p: dict) -> float:
        amt = (p.get("info") or {}).get("positionAmt")
        if amt is not None:
            return _f(amt)
        contracts = _f(p.get("contracts"))
        return contracts if p.get("side") == "long" else -contracts if p.get("side") == "short" else 0.0

    async def _find_order(self, usym: str, cid: str, conditional: bool = False) -> dict | None:
        params: dict[str, Any] = {"clientOrderId": cid}
        if conditional:
            params["trigger"] = True  # fetch_order -> fapiPrivateGetAlgoOrder(clientAlgoId=...)
        try:
            return await self.ex.fetch_order(None, usym, params)
        except ccxt.BaseError as e:
            if isinstance(e, ccxt.OrderNotFound) or _has_code(e, -2013):
                return None
            raise

    async def _submit(self, usym: str, otype: str, side: str, amount: float, price: float | None,
                      params: dict[str, Any], cid: str, conditional: bool = False) -> dict:
        """create_order with idempotent retries: after any ambiguous failure, look up `cid` first."""
        self._assert_allowed(self.cfg)
        raw = self._raw(usym) or usym
        retries = max(0, int(self.cfg.live.max_retries))
        maybe_sent = False
        for attempt in range(retries + 1):
            if attempt:
                await _sleep(_backoff(attempt - 1))
            if maybe_sent:
                try:
                    found = await self._find_order(usym, cid, conditional)
                except ccxt.BaseError as e:
                    if not _is_transient(e):
                        raise OrderStateUnknown(f"{fmt.pair(raw)}: emir {cid} durumu sorgulanamadı; pozisyonu "
                                                f"Binance'te kontrol edin ({_scrub(str(e), self._secrets())})",
                                                cid) from e
                    log.warning("%s: emir %s sorgulanamadı, tekrar gönderilmeden beklenecek: %s",
                                fmt.pair(raw), cid, _scrub(str(e), self._secrets()))
                    continue
                if found is not None:
                    log.warning("%s: emir %s borsaya ulaşmış, tekrar gönderilmiyor", fmt.pair(raw), cid)
                    return found
            try:
                return await self.ex.create_order(usym, otype, side, amount, price, dict(params))
            except ccxt.BaseError as e:
                if not _is_transient(e):
                    if maybe_sent:  # e.g. duplicate client id or margin used up: an earlier send may have landed
                        found = await self._lookup_or_unknown(usym, cid, conditional, raw)
                        if found is not None:
                            return found
                    raise self._translate(e, raw) from e
                if _is_clock_error(e):
                    await self._resync_clock()
                maybe_sent = True
                log.warning("%s: emir %s için geçici/belirsiz hata (deneme %d/%d): %s", fmt.pair(raw), cid,
                            attempt + 1, retries + 1, _scrub(str(e), self._secrets()))
        found = await self._lookup_or_unknown(usym, cid, conditional, raw)
        if found is not None:
            return found
        raise OrderRejected(f"{fmt.pair(raw)}: Binance'e {retries + 1} denemede ulaşılamadı; emrin iletilmediği "
                            f"doğrulandı ({cid})", "network")

    async def _lookup_or_unknown(self, usym: str, cid: str, conditional: bool, raw: str) -> dict | None:
        try:
            return await self._read(self._find_order, usym, cid, conditional)
        except ccxt.BaseError as e:
            raise OrderStateUnknown(f"{fmt.pair(raw)}: emir {cid} durumu doğrulanamadı; pozisyonu Binance'te "
                                    f"kontrol edin ({_scrub(str(e), self._secrets())})", cid) from e

    async def _order_trades(self, usym: str, order_id: str | None) -> list[dict]:
        if not order_id:
            return []
        for attempt in range(2):
            try:
                trades = await self._read(self.ex.fetch_my_trades, usym, None, None, {"orderId": order_id})
            except ccxt.BaseError as e:
                log.warning("%s: emir işlemleri okunamadı: %s", usym, _scrub(str(e), self._secrets()))
                return []
            trades = [t for t in trades or [] if str(t.get("order")) == str(order_id)]
            if trades:
                return trades
            if attempt == 0:
                await _sleep(0.5)
        return []

    async def _fees_usdt(self, items: list[dict]) -> float | None:
        """Sum of fee items in USDT; non-USDT fees (e.g. BNB) converted at the last price, None if impossible."""
        total = 0.0
        for item in items:
            cost = _f(item.get("cost"))
            currency = item.get("currency") or self.quote
            if cost == 0:
                continue
            if currency == self.quote:
                total += cost
                continue
            try:
                ticker = await self._read(self.ex.fetch_ticker, f"{currency}/{self.quote}:{self.quote}")
            except ccxt.BaseError:
                return None
            px = _f((ticker or {}).get("last"))
            if px <= 0:
                return None
            total += cost * px
        return total

    async def _fill(self, symbol: str, usym: str, side: int, order: dict, ref_price: float,
                    latency_ms: float, cid: str) -> Fill | None:
        """Actual fill of an accepted market order. Never raises once the order is known to have executed."""
        oid = order.get("id")
        for i in range(_FILL_POLLS):
            status = order.get("status")
            if status in _TERMINAL and (status != "closed" or _f(order.get("filled")) > 0):
                break
            if not oid:
                break
            await _sleep(0.2 * (i + 1))
            try:
                order = await self._read(self.ex.fetch_order, oid, usym) or order
            except ccxt.BaseError as e:
                log.warning("%s: emir %s sorgulanamadı: %s", fmt.pair(symbol), oid, _scrub(str(e), self._secrets()))
                break
        status = order.get("status")
        filled, avg = _f(order.get("filled")), _f(order.get("average"))
        ts = order.get("lastTradeTimestamp") or order.get("timestamp")
        trades: list[dict] | None = None
        if filled <= 0 or avg <= 0:
            trades = await self._order_trades(usym, oid)
            if trades:
                filled, avg, trade_ts = _aggregate(trades)
                ts = trade_ts or ts
        if filled <= 0:
            if status in _TERMINAL:
                return None
            raise OrderStateUnknown(f"{fmt.pair(symbol)}: emir {cid} kabul edildi ama dolum bilgisi alınamadı "
                                    f"(durum: {status}); pozisyonu Binance'te kontrol edin", cid)
        if avg <= 0:
            cost = _f(order.get("cost"))
            avg = cost / filled if cost > 0 else ref_price
            log.warning("%s: ortalama dolum fiyatı okunamadı, %s kullanılıyor", fmt.pair(symbol), fmt.price(avg))
        order_fees = [f for f in _fee_items(order) if _f(f.get("cost")) > 0]
        fee = await self._fees_usdt(order_fees) if order_fees else None  # USD-M responses carry no fee
        if fee is None:
            if trades is None:
                trades = await self._order_trades(usym, oid)
            items = [f for t in trades for f in _fee_items(t)]
            fee = await self._fees_usdt(items) if items else None
        if fee is None:
            fee = filled * avg * self._taker_rate(usym)
            log.warning("%s: komisyon okunamadı, tahmini %s kullanılıyor", fmt.pair(symbol), fmt.usd(fee))
        slippage = side * (avg / ref_price - 1) * 1e4 if ref_price > 0 else 0.0
        return Fill(symbol=symbol, side=side, qty=filled, price=avg, ref_price=ref_price, fee=fee,
                    slippage_bps=slippage, latency_ms=latency_ms, timestamp=_timestamp(ts))

    # -- account -----------------------------------------------------------------------
    async def positions(self) -> dict[str, ExchangePosition]:
        """Open positions (non-zero contracts) of the configured symbols, keyed by raw symbol."""
        if not self.symbol_map:
            return {}
        rows = await self._call(self._client().fetch_positions, list(self.symbol_map.values()))
        out: dict[str, ExchangePosition] = {}
        for p in rows or []:
            raw = self._raw(p.get("symbol"))
            amount = self._position_amount(p)
            if raw is None or amount == 0:
                continue
            leverage = _f(p.get("leverage"))
            if leverage <= 0:
                margin, notional = _f(p.get("initialMargin")), abs(_f(p.get("notional")))
                leverage = (float(round(notional / margin)) if margin > 0 and notional > 0
                            else self.leverage.get(raw, float(self.cfg.live.leverage)))
            out[raw] = ExchangePosition(symbol=raw, side=LONG if amount > 0 else SHORT, qty=abs(amount),
                                        entry_price=_f(p.get("entryPrice")), unrealized=_f(p.get("unrealizedPnl")),
                                        leverage=leverage)
        return out

    # -- orders ---------------------------------------------------------------------------
    async def market_order(self, symbol: str, side: int, qty: float, ref_price: float,
                           reduce_only: bool = False, client_id: str | None = None) -> Fill:
        """MARKET order; returns the real fill (average price, filled qty, USDT fee, slippage vs ref_price).

        The filled qty can be smaller than requested (step rounding, partial fill): always use fill.qty.
        """
        usym = self._unified(symbol)
        oside = _order_side(side)
        amount = self._amount(symbol, usym, qty)
        if not reduce_only:  # Binance exempts reduce-only orders from the min notional filter
            self._check_min_cost(symbol, usym, amount, ref_price)
        cid = client_id or self._new_client_id(symbol, side, "c" if reduce_only else "o")
        params: dict[str, Any] = {"clientOrderId": cid}
        if reduce_only:
            params["reduceOnly"] = True
        t0 = time.perf_counter()
        order = await self._submit(usym, "market", oside, amount, None, params, cid)
        latency_ms = (time.perf_counter() - t0) * 1000
        fill = await self._fill(symbol, usym, side, order, ref_price, latency_ms, cid)
        if fill is None:
            raise OrderRejected(f"{fmt.pair(symbol)}: piyasa emri dolmadı (durum: {order.get('status')})",
                                "no_position" if reduce_only else "not_filled")
        log.info("%s %s %s @ %s (kayma %.1f bps, komisyon %s, %.0f ms)", fmt.pair(symbol), oside.upper(),
                 fmt.qty(fill.qty), fmt.price(fill.price), fill.slippage_bps, fmt.usd(fill.fee), latency_ms)
        return fill

    async def close_position(self, symbol: str, position_side: int, qty: float, ref_price: float) -> Fill | None:
        """Reduce-only market order against the position; None when the exchange has nothing to reduce."""
        fill, rejected = await self._try_close(symbol, position_side, qty, ref_price)
        if rejected is None:
            return fill
        try:
            current = await self._signed_amount(self._unified(symbol))
        except ccxt.BaseError as e:
            raise self._translate(e, symbol) from e
        if current == 0 or (current > 0) != (position_side == LONG):
            log.info("%s: borsada kapatılacak pozisyon yok (borsa stopu tetiklenmiş olabilir)", fmt.pair(symbol))
            return None
        if abs(current) < qty:
            fill, again = await self._try_close(symbol, position_side, abs(current), ref_price)
            if again is None:
                return fill
            raise again
        raise rejected

    async def _try_close(self, symbol: str, position_side: int, qty: float,
                         ref_price: float) -> tuple[Fill | None, OrderRejected | None]:
        try:
            return await self.market_order(symbol, -position_side, qty, ref_price, reduce_only=True), None
        except OrderRejected as e:
            if e.code in ("no_position", "not_filled", "insufficient_funds"):
                return None, e
            raise

    async def place_protection(self, symbol: str, position_side: int, qty: float, stop_price: float,
                               take_profit: float | None = None, working_type: str = "MARK_PRICE") -> list[str]:
        """Exchange-side STOP_MARKET (+ optional TAKE_PROFIT_MARKET), reduce-only; returns their order ids.

        The stop is mandatory: its failure raises (code "would_trigger" means the price is already past
        the stop, close the position now). A failed take-profit is logged and skipped.
        """
        usym = self._unified(symbol)
        close_side = _order_side(-position_side)
        if working_type not in ("MARK_PRICE", "CONTRACT_PRICE"):
            raise OrderRejected(f"Geçersiz tetik fiyat türü: {working_type} (MARK_PRICE | CONTRACT_PRICE)", "invalid")
        amount = self._amount(symbol, usym, qty)
        stop = _f(stop_price)
        tp = _f(take_profit) if take_profit is not None else None
        if stop <= 0 or (tp is not None and tp <= 0):
            raise OrderRejected(f"{fmt.pair(symbol)}: geçersiz stop/kâr al fiyatı", "invalid")
        if tp is not None and (tp - stop) * position_side <= 0:
            raise OrderRejected(f"{fmt.pair(symbol)}: kâr al ({fmt.price(tp)}) stopun ({fmt.price(stop)}) "
                                "yanlış tarafında", "invalid")
        ids = [await self._conditional(symbol, usym, close_side, amount, "sl", "stopLossPrice", stop,
                                       working_type, position_side)]
        if tp is not None:
            try:
                ids.append(await self._conditional(symbol, usym, close_side, amount, "tp", "takeProfitPrice", tp,
                                                   working_type, position_side))
            except OrderRejected as e:
                self._warn(f"{fmt.pair(symbol)}: borsa kâr al emri konamadı, yalnızca stop aktif: {e}")
        return ids

    async def _conditional(self, symbol: str, usym: str, side: str, amount: float, tag: str, key: str,
                           trigger: float, working_type: str, position_side: int) -> str:
        # stopLossPrice -> STOP_MARKET, takeProfitPrice -> TAKE_PROFIT_MARKET; create_order routes
        # conditional linear orders to fapiPrivatePostAlgoOrder (algoType CONDITIONAL, clientAlgoId).
        price = self.ex.price_to_precision(usym, trigger)
        cid = self._new_client_id(symbol, position_side, tag)
        params = {key: price, "reduceOnly": True, "workingType": working_type, "clientOrderId": cid}
        order = await self._submit(usym, "market", side, amount, None, params, cid, conditional=True)
        oid = order.get("id")
        if not oid:
            raise OrderStateUnknown(f"{fmt.pair(symbol)}: koruma emri {cid} için borsa kimliği alınamadı", cid)
        log.info("%s: borsa %s emri %s @ %s", fmt.pair(symbol), "stop" if tag == "sl" else "kâr al", oid, price)
        return str(oid)

    async def cancel_protection(self, symbol: str, order_ids: list[str] | None = None) -> None:
        """Cancel the given conditional orders, or all conditional orders on the symbol plus the bot's
        regular open orders. Missing orders ("nothing to cancel") are ignored."""
        usym = self._unified(symbol)
        if order_ids:
            for oid in order_ids:
                await self._cancel(usym, oid, {"trigger": True})
            return
        try:
            await self._read(self.ex.cancel_all_orders, usym, {"trigger": True})  # -> fapiPrivateDeleteAlgoOpenOrders
        except ccxt.BaseError as e:
            if not _nothing_to_cancel(e):
                raise self._translate(e, symbol) from e
        orders = await self._call(self.ex.fetch_open_orders, usym)
        for o in orders or []:
            if str(o.get("clientOrderId") or "").startswith(CLIENT_ID_PREFIX):
                await self._cancel(usym, o.get("id"), {})

    async def _cancel(self, usym: str, order_id: str | None, params: dict[str, Any]) -> None:
        if not order_id:
            return
        try:
            await self._read(self.ex.cancel_order, order_id, usym, params)
        except ccxt.BaseError as e:
            if not _nothing_to_cancel(e):
                raise self._translate(e, self._raw(usym)) from e

    async def last_close_fill(self, symbol: str, since_ms: int, position_side: int,
                              ref_price: float | None = None) -> Fill | None:
        """Exit fill rebuilt from account trades since since_ms (e.g. after an exchange-side stop)."""
        usym = self._unified(symbol)
        since = max(int(since_ms), _now_ms() - _WEEK_MS + 60_000)  # userTrades spans at most 7 days
        trades = await self._call(self.ex.fetch_my_trades, usym, since, 1000)
        close_side = _order_side(-position_side)
        exits = [t for t in trades or [] if t.get("side") == close_side and _f(t.get("timestamp")) >= since]
        qty, avg, ts = _aggregate(exits)
        if qty <= 0:
            return None
        items = [f for t in exits for f in _fee_items(t)]
        fee = await self._fees_usdt(items) if items else 0.0
        if fee is None:
            fee = qty * avg * self._taker_rate(usym)
        ref = ref_price if ref_price and ref_price > 0 else avg
        side = -position_side
        return Fill(symbol=symbol, side=side, qty=qty, price=avg, ref_price=ref, fee=fee,
                    slippage_bps=side * (avg / ref - 1) * 1e4, latency_ms=0.0, timestamp=_timestamp(ts))

    async def close(self) -> None:
        self.connected = False
        await super().close()
