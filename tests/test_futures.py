"""Binance USD-M futures broker against a ccxt-shaped fake exchange (no network)."""

import asyncio
import time
from urllib.parse import parse_qsl

import ccxt
import pytest

import cappatrader.execution.binance_futures as bf
from cappatrader.config import MAINNET_CONFIRM_PHRASE, Config, Secrets
from cappatrader.execution.binance_futures import (
    CLIENT_ID_PREFIX,
    BinanceFuturesBroker,
    ExchangePosition,
    FuturesBalance,
    FuturesError,
    LiveTradingRefused,
    OrderRejected,
    OrderStateUnknown,
    client_order_id,
    fetch_binance_balance,
)
from cappatrader.models import LONG, SHORT

API_KEY = "test-api-key-1234567890"
SECRET = "test-secret-key-abcdef-0987654321"
BTC, ETH, BNB = "BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT"
FAPI = "https://fapi.binance.com/fapi/v1"
DEMO_FAPI = "https://demo-fapi.binance.com/fapi/v1"


def _raw_market(mid, base, step="0.001", tick="0.10", notional="100", status="TRADING", market_max="120"):
    return {
        "symbol": mid, "pair": mid, "contractType": "PERPETUAL", "deliveryDate": 4133404800000, "status": status,
        "baseAsset": base, "quoteAsset": "USDT", "marginAsset": "USDT", "pricePrecision": 2,
        "quantityPrecision": 3, "baseAssetPrecision": 8, "quotePrecision": 8,
        "filters": [
            {"filterType": "PRICE_FILTER", "tickSize": tick, "minPrice": "0.01", "maxPrice": "1000000"},
            {"filterType": "LOT_SIZE", "stepSize": step, "minQty": step, "maxQty": "1000"},
            {"filterType": "MARKET_LOT_SIZE", "stepSize": step, "minQty": step, "maxQty": market_max},
            {"filterType": "MIN_NOTIONAL", "notional": notional},
        ],
        "orderTypes": ["LIMIT", "MARKET", "STOP", "STOP_MARKET", "TAKE_PROFIT", "TAKE_PROFIT_MARKET",
                       "TRAILING_STOP_MARKET"],
        "timeInForce": ["GTC", "IOC", "FOK", "GTX"],
    }


RAW_MARKETS = [
    _raw_market("BTCUSDT", "BTC"),
    _raw_market("ETHUSDT", "ETH", tick="0.01", notional="20"),
    _raw_market("BNBUSDT", "BNB", step="0.01", tick="0.010", notional="5"),
    _raw_market("OLDUSDT", "OLD", status="SETTLING"),
    {**_raw_market("BTCUSDC", "BTC"), "quoteAsset": "USDC", "marginAsset": "USDC"},
]


def _err(cls, code, msg):
    return cls(f'binanceusdm {{"code":{code},"msg":"{msg}"}}')


class FakeExchange:
    """Async methods with ccxt-shaped return values; records every call."""

    def __init__(self, hedged=False, demo_switch=True):
        self._px = ccxt.binanceusdm()
        self._px.set_markets([self._px.parse_market(m) for m in RAW_MARKETS])
        self.markets = None
        self.options = {}
        self.urls = {"api": {"fapiPrivate": FAPI}}
        self.hedged = hedged
        self.demo_switch = demo_switch
        self.calls = []
        self.orders = []
        self.algo = []
        self.open_regular = []
        self.trades = []
        self.pos = {}
        self.entry = {}
        self.fill_price = {BTC: 60_000.0, ETH: 3_000.0, BNB: 600.0}
        self.tickers = {"BNB/USDT:USDT": 600.0}
        self.fee_currency = "USDT"
        self.response_mode = "result"
        self.create_effects = []
        self.fetch_order_effects = []
        self.margin_errors = {}
        self.leverage_errors = {}
        self.balance_info = {"assets": [{"asset": "USDT", "walletBalance": "1000", "unrealizedProfit": "-12.5",
                                         "marginBalance": "987.5", "availableBalance": "800", "updateTime": 1}]}
        self.closed = False

    def _rec(self, name, *args):
        self.calls.append((name, args))

    def names(self):
        return [c[0] for c in self.calls]

    def args(self, name):
        return [c[1] for c in self.calls if c[0] == name]

    def enable_demo_trading(self, enable):
        self._rec("enable_demo_trading", enable)
        self.options["enableDemoTrading"] = enable
        if self.demo_switch:
            self.urls["api"]["fapiPrivate"] = DEMO_FAPI if enable else FAPI

    async def load_markets(self, reload=False, params={}):
        self._rec("load_markets")
        self.markets = self._px.markets
        return self.markets

    def market(self, symbol):
        return self._px.market(symbol)

    def amount_to_precision(self, symbol, amount):
        return self._px.amount_to_precision(symbol, amount)

    def price_to_precision(self, symbol, price):
        return self._px.price_to_precision(symbol, price)

    async def load_time_difference(self):
        self._rec("load_time_difference")

    async def fetch_position_mode(self, symbol=None, params={}):
        self._rec("fetch_position_mode")
        return {"info": {"dualSidePosition": self.hedged}, "hedged": self.hedged}

    async def set_margin_mode(self, mode, symbol=None, params={}):
        self._rec("set_margin_mode", mode, symbol)
        if symbol in self.margin_errors:
            raise self.margin_errors[symbol]
        return {"code": 200, "msg": "success"}

    async def set_leverage(self, leverage, symbol=None, params={}):
        self._rec("set_leverage", leverage, symbol)
        if symbol in self.leverage_errors:
            raise self.leverage_errors[symbol]
        return {"symbol": symbol, "leverage": leverage}

    async def fetch_balance(self, params={}):
        self._rec("fetch_balance", dict(params))
        if params.get("type") == "spot":
            return {"info": {}, "USDT": {"free": 5.0, "used": 1.0, "total": 6.0}}
        return {"info": self.balance_info, "USDT": {"free": 800.0, "used": 200.0, "total": 987.5}}

    async def fetch_positions(self, symbols=None, params={}):
        self._rec("fetch_positions", symbols)
        rows = []
        for sym, amount in self.pos.items():
            if amount == 0 or (symbols and sym not in symbols):
                continue
            entry = self.entry.get(sym, self.fill_price.get(sym, 1.0))
            rows.append({"symbol": sym, "contracts": abs(amount), "side": "long" if amount > 0 else "short",
                         "entryPrice": entry, "unrealizedPnl": 1.5, "leverage": None,
                         "initialMargin": abs(amount) * entry / 2, "notional": abs(amount) * entry,
                         "info": {"symbol": sym.split("/")[0] + "USDT", "positionAmt": str(amount)}})
        return rows

    async def create_order(self, symbol, type, side, amount, price=None, params={}):
        self._rec("create_order", symbol, type, side, amount, price, dict(params))
        effect = self.create_effects.pop(0) if self.create_effects else None
        if isinstance(effect, Exception):
            raise effect
        order = self._book(symbol, side, amount, params)
        if isinstance(effect, tuple):  # the order landed, then the connection dropped
            raise effect[1]
        return dict(order)

    def _book(self, symbol, side, amount, params):
        cid = params.get("clientOrderId")
        oid = str(1000 + len(self.orders) + len(self.algo))
        sign = 1 if side == "buy" else -1
        if "stopLossPrice" in params or "takeProfitPrice" in params:
            kind = "stop_market" if "stopLossPrice" in params else "take_profit_market"
            order = {"id": oid, "clientOrderId": cid, "symbol": symbol, "type": kind, "side": side,
                     "amount": amount, "filled": 0.0, "average": None, "status": "open",
                     "triggerPrice": float(params.get("stopLossPrice") or params.get("takeProfitPrice")),
                     "reduceOnly": params.get("reduceOnly"), "info": {"algoId": oid, "clientAlgoId": cid}}
            self.algo.append(order)
            return order
        if params.get("reduceOnly"):
            current = self.pos.get(symbol, 0.0)
            if current == 0 or current * sign > 0 or amount > abs(current) + 1e-12:
                raise _err(ccxt.InvalidOrder, -2022, "ReduceOnly Order is rejected.")
        px = self.fill_price[symbol]
        ts = int(time.time() * 1000)
        order = {"id": oid, "clientOrderId": cid, "symbol": symbol, "type": "market", "side": side,
                 "amount": amount, "filled": amount, "average": px, "cost": amount * px, "status": "closed",
                 "timestamp": ts, "lastTradeTimestamp": ts, "fee": None, "fees": [],
                 "reduceOnly": bool(params.get("reduceOnly")), "info": {"orderId": oid}}
        self.orders.append(order)
        fee_usdt = amount * px * 0.0005
        fee = fee_usdt if self.fee_currency == "USDT" else fee_usdt / self.tickers["BNB/USDT:USDT"]
        self.trades.append({"id": "t" + oid, "order": oid, "symbol": symbol, "side": side, "amount": amount,
                            "price": px, "cost": amount * px, "timestamp": ts,
                            "fee": {"cost": fee, "currency": self.fee_currency},
                            "fees": [{"cost": fee, "currency": self.fee_currency}]})
        self.pos[symbol] = round(self.pos.get(symbol, 0.0) + sign * amount, 12)
        self.entry.setdefault(symbol, px)
        if self.response_mode == "ack":
            return {**order, "status": "open", "filled": 0.0, "average": None, "cost": 0.0}
        return order

    async def fetch_order(self, id, symbol=None, params={}):
        self._rec("fetch_order", id, symbol, dict(params))
        if self.fetch_order_effects:
            raise self.fetch_order_effects.pop(0)
        cid = params.get("clientOrderId")
        pool = self.algo if params.get("trigger") else self.orders
        for order in pool:
            if (cid and order["clientOrderId"] == cid) or (not cid and order["id"] == id):
                return dict(order)
        raise _err(ccxt.OrderNotFound, -2013, "Order does not exist.")

    async def fetch_my_trades(self, symbol=None, since=None, limit=None, params={}):
        self._rec("fetch_my_trades", symbol, since, limit, dict(params))
        out = [t for t in self.trades if t["symbol"] == symbol]
        if "orderId" in params:
            out = [t for t in out if t["order"] == str(params["orderId"])]
        if since is not None:
            out = [t for t in out if t["timestamp"] >= since]
        return [dict(t) for t in out]

    async def fetch_ticker(self, symbol, params={}):
        self._rec("fetch_ticker", symbol)
        return {"symbol": symbol, "last": self.tickers[symbol]}

    async def cancel_all_orders(self, symbol=None, params={}):
        self._rec("cancel_all_orders", symbol, dict(params))
        if params.get("trigger"):
            self.algo = [o for o in self.algo if o["symbol"] != symbol]
        return [{"info": {"code": 200, "msg": "The operation of cancel all open order is done."}}]

    async def fetch_open_orders(self, symbol=None, since=None, limit=None, params={}):
        self._rec("fetch_open_orders", symbol, dict(params))
        return [dict(o) for o in self.open_regular if o["symbol"] == symbol]

    async def cancel_order(self, id, symbol=None, params={}):
        self._rec("cancel_order", id, symbol, dict(params))
        pool = self.algo if params.get("trigger") else self.open_regular
        for order in pool:
            if order["id"] == id:
                pool.remove(order)
                return {"id": id, "status": "canceled"}
        raise _err(ccxt.OrderNotFound, -2011, "Unknown order sent.")

    async def close(self):
        self.closed = True


def live_cfg(env="demo", symbols=("BTCUSDT", "ETHUSDT", "BNBUSDT", "PAXGUSDT", "OLDUSDT"), **live):
    c = Config()
    c.mode = "live"
    c.symbols = list(symbols)
    c.pairs = []
    c.live.env = env
    if env == "mainnet":
        c.live.mainnet_confirm = MAINNET_CONFIRM_PHRASE
    for key, value in live.items():
        setattr(c.live, key, value)
    c.secrets = Secrets(binance_api_key=API_KEY, binance_secret_key=SECRET)
    c.validate()
    return c


@pytest.fixture(autouse=True)
def sleeps(monkeypatch):
    delays = []

    async def fake_sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr(bf, "_sleep", fake_sleep)
    return delays


async def connected(cfg=None, ex=None):
    ex = ex or FakeExchange()
    broker = BinanceFuturesBroker(cfg or live_cfg(), ex)
    await broker.connect()
    return broker, ex


def run(coro):
    return asyncio.run(coro)


# -- construction / connection -------------------------------------------------------------
def test_refuses_paper_mode_missing_keys_and_unconfirmed_mainnet():
    ex = FakeExchange()
    paper = live_cfg()
    paper.mode = "paper"
    with pytest.raises(LiveTradingRefused, match="mode: live"):
        BinanceFuturesBroker(paper, ex)
    no_keys = live_cfg()
    no_keys.secrets = Secrets()
    with pytest.raises(LiveTradingRefused, match="BINANCE_API_KEY"):
        BinanceFuturesBroker(no_keys, ex)
    mainnet = live_cfg("mainnet")
    for phrase in ("", "evet", MAINNET_CONFIRM_PHRASE.lower()):
        mainnet.live.mainnet_confirm = phrase
        with pytest.raises(LiveTradingRefused, match="GERÇEK PARA"):
            BinanceFuturesBroker(mainnet, ex)
    assert ex.calls == []


def test_mainnet_confirm_is_rechecked_before_every_order():
    async def scenario():
        broker, ex = await connected(live_cfg("mainnet"))
        broker.cfg.live.mainnet_confirm = ""
        with pytest.raises(LiveTradingRefused):
            await broker.market_order("BTCUSDT", LONG, 0.002, 60_000.0)
        return ex

    ex = run(scenario())
    assert "enable_demo_trading" not in ex.names()
    assert ex.args("create_order") == []


def test_demo_mode_enables_demo_trading_and_mainnet_does_not():
    broker, ex = run(connected(live_cfg("demo")))
    assert ex.args("enable_demo_trading") == [(True,)]
    assert ex.urls["api"]["fapiPrivate"] == DEMO_FAPI
    assert broker.connected and broker.env == "demo"
    _, ex2 = run(connected(live_cfg("mainnet")))
    assert "enable_demo_trading" not in ex2.names()


def test_demo_env_refused_when_client_still_points_to_mainnet():
    with pytest.raises(LiveTradingRefused, match="mainnet"):
        run(connected(live_cfg("demo"), FakeExchange(demo_switch=False)))


def test_connect_maps_symbols_and_sets_margin_and_leverage():
    ex = FakeExchange()
    ex.margin_errors[ETH] = ccxt.MarginModeAlreadySet(
        'binanceusdm {"code":-4046,"msg":"No need to change margin type."}')
    broker, ex = run(connected(live_cfg(leverage=3, margin_mode="isolated"), ex))
    assert broker.symbol_map == {"BTCUSDT": BTC, "ETHUSDT": ETH, "BNBUSDT": BNB}
    assert set(broker.unavailable) == {"PAXGUSDT", "OLDUSDT"}
    assert "perpetual" in broker.unavailable["PAXGUSDT"] and "kapalı" in broker.unavailable["OLDUSDT"]
    assert any("PAXG/USDT" in w and "işlem yapılmayacak" in w for w in broker.warnings)
    assert ex.args("set_margin_mode") == [("isolated", BTC), ("isolated", ETH), ("isolated", BNB)]
    assert ex.args("set_leverage") == [(3, BTC), (3, ETH), (3, BNB)]
    assert broker.leverage == {"BTCUSDT": 3.0, "ETHUSDT": 3.0, "BNBUSDT": 3.0}
    with pytest.raises(OrderRejected) as info:
        run(broker.market_order("PAXGUSDT", LONG, 1.0, 2_000.0))
    assert info.value.code == "unavailable"


def test_connect_leverage_failure_drops_symbol_unless_a_position_exists():
    ex = FakeExchange()
    bad = _err(ccxt.BadRequest, -4028, "Leverage 3 is not valid")
    ex.leverage_errors = {BNB: bad, ETH: bad}
    ex.pos[ETH] = 0.5
    broker, ex = run(connected(live_cfg(leverage=3), ex))
    assert "BNBUSDT" in broker.unavailable and "kaldıraç" in broker.unavailable["BNBUSDT"]
    assert "ETHUSDT" in broker.symbol_map
    assert any("ETH/USDT" in w and "kaldıraç" in w for w in broker.warnings)


def test_connect_refuses_hedge_mode_and_multi_assets_isolated():
    with pytest.raises(LiveTradingRefused, match="Tek Yönlü"):
        run(connected(ex=FakeExchange(hedged=True)))
    ex = FakeExchange()
    ex.margin_errors[BTC] = _err(ccxt.BadRequest, -4168,
                                 "Unable to adjust to isolated-margin mode under the Multi-Assets mode.")
    with pytest.raises(LiveTradingRefused, match="Multi-Assets"):
        run(connected(ex=ex))


def test_connect_auth_error_is_refusal_without_secret():
    ex = FakeExchange()

    async def denied(symbol=None, params={}):
        raise _err(ccxt.AuthenticationError, -2015, f"Invalid API-key {SECRET}, IP, or permissions for action.")

    ex.fetch_position_mode = denied
    with pytest.raises(LiveTradingRefused) as info:
        run(connected(ex=ex))
    assert SECRET not in str(info.value) and "***" in str(info.value)


# -- balances / positions -----------------------------------------------------------------
def test_balance_spot_balance_and_positions_parsing():
    async def scenario():
        ex = FakeExchange()
        ex.pos = {BTC: 0.002, ETH: -0.5, BNB: 0.0, "SOL/USDT:USDT": 3.0}
        ex.entry = {BTC: 60_000.0, ETH: 3_000.0}
        broker, _ = await connected(ex=ex)
        return await broker.balance(), await broker.spot_usdt_balance(), await broker.positions(), ex

    bal, spot, positions, ex = run(scenario())
    assert bal == FuturesBalance(wallet=1000.0, available=800.0, unrealized=-12.5, equity=987.5)
    assert spot == pytest.approx(6.0)
    assert {"type": "spot", "subType": "spot"} in [a[0] for a in ex.args("fetch_balance")]
    assert set(positions) == {"BTCUSDT", "ETHUSDT"}
    assert positions["BTCUSDT"] == ExchangePosition("BTCUSDT", LONG, 0.002, 60_000.0, 1.5, 2.0)
    assert positions["ETHUSDT"].side == SHORT and positions["ETHUSDT"].qty == pytest.approx(0.5)


def test_fetch_binance_balance_works_in_paper_mode_read_only():
    cfg = Config()
    cfg.account.balance_source = "binance"
    ex = FakeExchange()
    assert run(fetch_binance_balance(cfg, ex)) == pytest.approx(987.5)
    assert "enable_demo_trading" not in ex.names() and ex.closed
    assert "create_order" not in ex.names()
    cfg.account.binance_wallet = "spot"
    assert run(fetch_binance_balance(cfg, FakeExchange())) == pytest.approx(6.0)
    with pytest.raises(LiveTradingRefused, match="BINANCE_API_KEY"):
        run(fetch_binance_balance(cfg))


# -- market orders ---------------------------------------------------------------------------
def test_market_order_rounds_down_and_extracts_fill():
    async def scenario():
        broker, ex = await connected()
        a = await broker.market_order("BTCUSDT", LONG, 0.0029, 59_990.0)
        b = await broker.market_order("BTCUSDT", LONG, 0.1 * 3 / 100, 59_990.0)  # 0.0030000000000000005
        c = await broker.market_order("ETHUSDT", SHORT, 0.0299999999999999996, 3_003.0)
        return a, b, c, ex

    a, b, c, ex = run(scenario())
    amounts = [args[3] for args in ex.args("create_order")]
    assert amounts == [0.002, 0.003, 0.03]
    sides = [args[2] for args in ex.args("create_order")]
    assert sides == ["buy", "buy", "sell"]
    assert a.qty == 0.002 and a.price == 60_000.0 and a.side == LONG and a.symbol == "BTCUSDT"
    assert a.fee == pytest.approx(0.002 * 60_000 * 0.0005)
    assert a.slippage_bps == pytest.approx((60_000 / 59_990 - 1) * 1e4)
    assert c.slippage_bps == pytest.approx(-(3_000 / 3_003 - 1) * 1e4)
    assert a.latency_ms >= 0 and str(a.timestamp.tz) == "UTC"
    cids = [args[5]["clientOrderId"] for args in ex.args("create_order")]
    assert all(cid.startswith(CLIENT_ID_PREFIX + "o-") and len(cid) <= 36 for cid in cids)
    assert len(set(cids)) == 3
    assert all("reduceOnly" not in args[5] for args in ex.args("create_order"))


def test_market_order_limits_reject_before_sending():
    async def scenario():
        broker, ex = await connected()
        codes = []
        for sym, qty, px in (("BTCUSDT", 0.001, 60_000.0), ("BTCUSDT", 0.0004, 60_000.0),
                             ("BTCUSDT", 130.0, 60_000.0), ("BTCUSDT", -1.0, 60_000.0)):
            with pytest.raises(OrderRejected) as info:
                await broker.market_order(sym, LONG, qty, px)
            codes.append(info.value.code)
        return codes, ex, str(info.value)

    codes, ex, _ = run(scenario())
    assert codes == ["min_notional", "min_qty", "max_qty", "invalid"]
    assert "create_order" not in ex.names()


def test_min_notional_message_is_turkish():
    async def scenario():
        broker, _ = await connected()
        with pytest.raises(OrderRejected) as info:
            await broker.market_order("BTCUSDT", LONG, 0.001, 60_000.0)
        return str(info.value)

    assert "en küçük emir tutarının" in run(scenario())


def test_fill_from_fetch_order_and_bnb_fee_conversion():
    async def scenario():
        ex = FakeExchange()
        ex.response_mode = "ack"
        ex.fee_currency = "BNB"
        broker, _ = await connected(ex=ex)
        return await broker.market_order("ETHUSDT", LONG, 0.05, 3_000.0), ex

    fill, ex = run(scenario())
    assert fill.qty == pytest.approx(0.05) and fill.price == pytest.approx(3_000.0)
    assert fill.fee == pytest.approx(0.05 * 3_000 * 0.0005)
    assert ex.args("fetch_order")[0][0] == ex.orders[0]["id"]
    assert ("fetch_ticker", ("BNB/USDT:USDT",)) in ex.calls


def test_fill_falls_back_to_my_trades_when_order_queries_fail():
    async def scenario():
        ex = FakeExchange()
        ex.response_mode = "ack"
        broker, _ = await connected(ex=ex)
        ex.fetch_order_effects = [ccxt.NetworkError("binanceusdm GET connection reset")] * 20
        return await broker.market_order("BTCUSDT", SHORT, 0.002, 60_000.0), ex

    fill, ex = run(scenario())
    assert fill.qty == pytest.approx(0.002) and fill.price == pytest.approx(60_000.0)
    assert fill.fee == pytest.approx(0.002 * 60_000 * 0.0005)
    assert ex.args("fetch_my_trades")[0][3] == {"orderId": ex.orders[0]["id"]}
    assert len(ex.orders) == 1


# -- idempotency / retries -------------------------------------------------------------------
def test_timeout_after_send_does_not_create_duplicate_order(sleeps):
    async def scenario():
        broker, ex = await connected()
        ex.create_effects = [("after", ccxt.RequestTimeout("binanceusdm POST https://fapi... timed out"))]
        return await broker.market_order("BTCUSDT", LONG, 0.002, 60_000.0), ex

    fill, ex = run(scenario())
    assert len(ex.args("create_order")) == 1
    assert len(ex.orders) == 1 and ex.pos[BTC] == pytest.approx(0.002)
    cid = ex.args("create_order")[0][5]["clientOrderId"]
    assert ex.args("fetch_order")[0][2] == {"clientOrderId": cid}
    assert fill.qty == pytest.approx(0.002)
    assert sleeps[0] == 1.0


def test_timeout_before_send_retries_with_the_same_client_id():
    async def scenario():
        broker, ex = await connected()
        ex.create_effects = [ccxt.RequestTimeout("binanceusdm timed out")]
        return await broker.market_order("BTCUSDT", LONG, 0.002, 60_000.0), ex

    fill, ex = run(scenario())
    cids = [args[5]["clientOrderId"] for args in ex.args("create_order")]
    assert len(cids) == 2 and cids[0] == cids[1]
    assert len(ex.orders) == 1 and fill.qty == pytest.approx(0.002)


def test_transient_errors_retried_with_exponential_backoff(sleeps):
    async def scenario():
        broker, ex = await connected()
        ex.create_effects = [ccxt.ExchangeNotAvailable("binanceusdm 503 Service Unavailable"),
                             ccxt.DDoSProtection("binanceusdm 429 Too Many Requests"),
                             ccxt.InvalidNonce('binanceusdm {"code":-1021,"msg":"Timestamp outside recvWindow"}')]
        return await broker.market_order("BTCUSDT", LONG, 0.002, 60_000.0), ex

    fill, ex = run(scenario())
    assert sleeps == [1.0, 2.0, 4.0]
    assert len(ex.args("create_order")) == 4 and len(ex.orders) == 1
    assert "load_time_difference" in ex.names()
    assert fill.qty == pytest.approx(0.002)


def test_retries_exhausted_but_verified_not_sent_is_rejected(sleeps):
    async def scenario():
        broker, ex = await connected(live_cfg(max_retries=2))
        ex.create_effects = [ccxt.RequestTimeout("binanceusdm timed out")] * 3
        with pytest.raises(OrderRejected) as info:
            await broker.market_order("BTCUSDT", LONG, 0.002, 60_000.0)
        return info.value, ex

    err, ex = run(scenario())
    assert err.code == "network"
    assert len(ex.args("create_order")) == 3 and ex.orders == []
    assert sleeps == [1.0, 2.0]


@pytest.mark.parametrize("exc,expected,code", [
    (_err(ccxt.InsufficientFunds, -2019, "Margin is insufficient."), OrderRejected, "insufficient_funds"),
    (_err(ccxt.InvalidOrder, -4164, "Order notional must be no smaller than 100"), OrderRejected, "rejected"),
    (_err(ccxt.AuthenticationError, -2015, "Invalid API-key, IP, or permissions for action."),
     LiveTradingRefused, None),
    (_err(ccxt.OperationFailed, -2023, "User in liquidation mode now."), FuturesError, None),
])
def test_permanent_errors_are_not_retried(sleeps, exc, expected, code):
    async def scenario():
        broker, ex = await connected()
        ex.create_effects = [exc]
        with pytest.raises(expected) as info:
            await broker.market_order("BTCUSDT", LONG, 0.002, 60_000.0)
        return info.value, ex

    err, ex = run(scenario())
    assert len(ex.args("create_order")) == 1 and "fetch_order" not in ex.names()
    assert sleeps == []
    if code:
        assert err.code == code


def test_unverifiable_order_raises_state_unknown_and_never_resends():
    async def scenario():
        broker, ex = await connected()
        ex.create_effects = [("after", ccxt.RequestTimeout("binanceusdm timed out"))]
        ex.fetch_order_effects = [ccxt.NetworkError("binanceusdm connection reset")] * 50
        with pytest.raises(OrderStateUnknown) as info:
            await broker.market_order("BTCUSDT", LONG, 0.002, 60_000.0)
        return info.value, ex

    err, ex = run(scenario())
    assert len(ex.args("create_order")) == 1 and len(ex.orders) == 1
    assert err.client_id == ex.args("create_order")[0][5]["clientOrderId"]
    assert "Binance'te" in str(err)


def test_lookup_rejected_after_ambiguous_send_is_state_unknown_not_rejection():
    async def scenario():
        broker, ex = await connected()
        ex.create_effects = [("after", ccxt.RequestTimeout("binanceusdm timed out"))]
        ex.fetch_order_effects = [_err(ccxt.BadRequest, -1102, "Mandatory parameter was not sent.")]
        with pytest.raises(OrderStateUnknown):
            await broker.market_order("BTCUSDT", LONG, 0.002, 60_000.0)
        return ex

    ex = run(scenario())
    assert len(ex.args("create_order")) == 1


def test_transient_failure_while_configuring_fails_connect(sleeps):
    ex = FakeExchange()
    ex.leverage_errors[BTC] = ccxt.ExchangeNotAvailable("binanceusdm 502 Bad Gateway")
    broker = BinanceFuturesBroker(live_cfg(max_retries=1), ex)
    with pytest.raises(FuturesError, match="geçici"):
        run(broker.connect())
    assert not broker.connected and "BTCUSDT" not in broker.unavailable
    assert ex.args("set_leverage") == [(2, BTC), (2, BTC)] and sleeps == [1.0]


def test_exchange_errors_are_scrubbed_of_secrets():
    async def scenario():
        broker, ex = await connected()
        ex.create_effects = [ccxt.ExchangeError(f"binanceusdm boom {SECRET} {API_KEY} signature=abcdef0123")]
        with pytest.raises(FuturesError) as info:
            await broker.market_order("BTCUSDT", LONG, 0.002, 60_000.0)
        return str(info.value)

    text = run(scenario())
    assert SECRET not in text and API_KEY not in text and "abcdef0123" not in text


# -- closing ------------------------------------------------------------------------------
def test_close_position_sends_reduce_only_opposite_order():
    async def scenario():
        broker, ex = await connected()
        ex.pos[BTC] = 0.002
        return await broker.close_position("BTCUSDT", LONG, 0.002, 60_010.0), ex

    fill, ex = run(scenario())
    _, otype, side, amount, _, params = ex.args("create_order")[0]
    assert (otype, side, amount) == ("market", "sell", 0.002)
    assert params["reduceOnly"] is True and params["clientOrderId"].startswith(CLIENT_ID_PREFIX + "c-")
    assert fill.side == SHORT and fill.qty == pytest.approx(0.002) and ex.pos[BTC] == 0


def test_close_position_returns_none_when_exchange_stop_already_closed_it():
    async def scenario():
        broker, ex = await connected()
        ex.pos[BTC] = 0.0
        return await broker.close_position("BTCUSDT", LONG, 0.002, 60_000.0), ex

    fill, ex = run(scenario())
    assert fill is None
    assert len(ex.args("create_order")) == 1 and ex.orders == []


def test_close_position_clamps_to_the_exchange_position():
    async def scenario():
        broker, ex = await connected()
        ex.pos[ETH] = -0.02
        return await broker.close_position("ETHUSDT", SHORT, 0.05, 3_000.0), ex

    fill, ex = run(scenario())
    assert [args[3] for args in ex.args("create_order")] == [0.05, 0.02]
    assert fill.qty == pytest.approx(0.02) and fill.side == LONG and ex.pos[ETH] == 0


# -- exchange-side protection -------------------------------------------------------------------
def test_place_protection_sends_reduce_only_stop_and_take_profit():
    async def scenario():
        broker, ex = await connected()
        long_ids = await broker.place_protection("BTCUSDT", LONG, 0.002, 58_950.04, 62_000.04)
        short_ids = await broker.place_protection("ETHUSDT", SHORT, 0.05, 3_100.004, None)
        return long_ids, short_ids, ex

    long_ids, short_ids, ex = run(scenario())
    stop, tp, short_stop = ex.args("create_order")
    assert stop[:5] == (BTC, "market", "sell", 0.002, None)
    assert stop[5]["stopLossPrice"] == "58950" and stop[5]["reduceOnly"] is True
    assert stop[5]["workingType"] == "MARK_PRICE" and stop[5]["clientOrderId"].startswith(CLIENT_ID_PREFIX + "sl-")
    assert tp[5]["takeProfitPrice"] == "62000" and tp[5]["reduceOnly"] is True and tp[2] == "sell"
    assert tp[5]["clientOrderId"].startswith(CLIENT_ID_PREFIX + "tp-")
    assert short_stop[2] == "buy" and short_stop[5]["stopLossPrice"] == "3100"
    assert "takeProfitPrice" not in short_stop[5]
    assert long_ids == [ex.algo[0]["id"], ex.algo[1]["id"]] and short_ids == [ex.algo[2]["id"]]


def test_place_protection_rejects_bad_prices_and_reports_immediate_trigger():
    async def scenario():
        broker, ex = await connected()
        with pytest.raises(OrderRejected) as wrong_side:
            await broker.place_protection("BTCUSDT", LONG, 0.002, 61_000.0, 60_000.0)
        ex.create_effects = [ccxt.OrderImmediatelyFillable(
            'binanceusdm {"code":-2021,"msg":"Order would immediately trigger."}')]
        with pytest.raises(OrderRejected) as trigger:
            await broker.place_protection("BTCUSDT", LONG, 0.002, 61_000.0, 65_000.0)
        return wrong_side.value, trigger.value, ex

    wrong_side, trigger, ex = run(scenario())
    assert wrong_side.code == "invalid" and trigger.code == "would_trigger"
    assert len(ex.args("create_order")) == 1


def test_take_profit_failure_keeps_the_stop():
    async def scenario():
        broker, ex = await connected()
        ex.create_effects = [None, _err(ccxt.BadRequest, -4183, "Price is higher than stop price multiplier cap.")]
        ids = await broker.place_protection("BTCUSDT", LONG, 0.002, 58_000.0, 90_000.0)
        return ids, broker, ex

    ids, broker, ex = run(scenario())
    assert ids == [ex.algo[0]["id"]] and len(ex.algo) == 1
    assert any("kâr al" in w for w in broker.warnings)


def test_cancel_protection_cancels_conditional_and_bot_orders_only():
    async def scenario():
        broker, ex = await connected()
        ids = await broker.place_protection("BTCUSDT", LONG, 0.002, 58_000.0, 62_000.0)
        ex.open_regular = [{"id": "77", "symbol": BTC, "clientOrderId": "cappa-o-123"},
                           {"id": "78", "symbol": BTC, "clientOrderId": "manual-order"}]
        await broker.cancel_protection("BTCUSDT")
        after_all = (list(ex.algo), list(ex.open_regular))
        more = await broker.place_protection("BTCUSDT", LONG, 0.002, 58_000.0, None)
        await broker.cancel_protection("BTCUSDT", more + ["does-not-exist"])
        return ids, after_all, ex

    ids, (algo_left, regular_left), ex = run(scenario())
    assert (BTC, {"trigger": True}) in ex.args("cancel_all_orders")
    assert algo_left == [] and [o["id"] for o in regular_left] == ["78"]
    assert ("77", BTC, {}) in ex.args("cancel_order")
    assert ("does-not-exist", BTC, {"trigger": True}) in ex.args("cancel_order")
    assert ex.algo == []


# -- exit reconstruction -----------------------------------------------------------------------
def test_last_close_fill_aggregates_exit_trades():
    now = int(time.time() * 1000)

    async def scenario():
        broker, ex = await connected()
        ex.trades = [
            {"symbol": BTC, "order": "1", "side": "buy", "amount": 0.003, "price": 60_000.0, "timestamp": now - 5_000,
             "fee": {"cost": 0.09, "currency": "USDT"}},
            {"symbol": BTC, "order": "2", "side": "sell", "amount": 0.001, "price": 58_000.0, "timestamp": now - 2_000,
             "fee": {"cost": 0.029, "currency": "USDT"}},
            {"symbol": BTC, "order": "2", "side": "sell", "amount": 0.002, "price": 57_970.0, "timestamp": now - 1_000,
             "fee": {"cost": 0.058, "currency": "USDT"}},
            {"symbol": BTC, "order": "0", "side": "sell", "amount": 1.0, "price": 1.0, "timestamp": now - 99_000,
             "fee": {"cost": 9.0, "currency": "USDT"}},
        ]
        fill = await broker.last_close_fill("BTCUSDT", now - 10_000, LONG, ref_price=58_000.0)
        none = await broker.last_close_fill("BTCUSDT", now - 500, LONG)
        return fill, none

    fill, none = run(scenario())
    assert none is None
    assert fill.side == SHORT and fill.qty == pytest.approx(0.003)
    assert fill.price == pytest.approx((0.001 * 58_000 + 0.002 * 57_970) / 0.003)
    assert fill.fee == pytest.approx(0.087)
    assert fill.slippage_bps == pytest.approx(-(fill.price / 58_000 - 1) * 1e4)
    assert fill.timestamp.value // 1_000_000 == now - 1_000


def test_client_order_id_is_deterministic_and_short():
    a = client_order_id("1000SHIBUSDT", LONG, "sl", 1_700_000_000_000, 99)
    assert a == client_order_id("1000SHIBUSDT", LONG, "sl", 1_700_000_000_000, 99)
    assert a != client_order_id("1000SHIBUSDT", LONG, "sl", 1_700_000_000_000, 100)
    assert a.startswith("cappa-sl-") and len(a) <= 36


def test_not_connected_broker_refuses_orders():
    broker = BinanceFuturesBroker(live_cfg(), FakeExchange())
    with pytest.raises(FuturesError, match="connect"):
        run(broker.market_order("BTCUSDT", LONG, 0.002, 60_000.0))


# -- real ccxt request building (offline: the HTTP layer is patched) -------------------------------
def test_real_ccxt_builds_expected_demo_requests_offline():
    cfg = live_cfg("demo", symbols=("BTCUSDT",), leverage=2, margin_mode="isolated")
    sent = []

    async def scenario():
        ex = bf.build_exchange(cfg)
        assert ex.options["defaultType"] == "future" and ex.options["adjustForTimeDifference"] is True
        assert ex.options["recvWindow"] == cfg.exchange.recv_window_ms and ex.enableRateLimit is True
        assert ex.apiKey == API_KEY and ex.secret == SECRET
        ex.enableRateLimit = False  # the throttler would only slow this offline test down
        ex.set_markets([ex.parse_market(RAW_MARKETS[0])])
        algo_ids = iter(range(501, 600))

        async def fetch(url, method="GET", headers=None, body=None):
            path, _, query = url.partition("?")
            params = dict(parse_qsl(body or query))
            sent.append((method, path, params, headers or {}))
            if path.endswith("/marginType"):
                raise ccxt.MarginModeAlreadySet('binanceusdm {"code":-4046,"msg":"No need to change margin type."}')
            if path.endswith("/leverage"):
                return {"symbol": "BTCUSDT", "leverage": int(params["leverage"]), "maxNotionalValue": "1000000"}
            if path.endswith("/positionSide/dual"):
                return {"dualSidePosition": False}
            if path.endswith("/fapi/v1/order") and method == "POST":
                return {"orderId": 123, "symbol": "BTCUSDT", "status": "FILLED",
                        "clientOrderId": params["newClientOrderId"], "price": "0", "avgPrice": "60012.30",
                        "origQty": params["quantity"], "executedQty": params["quantity"], "cumQuote": "120.0246",
                        "timeInForce": "GTC", "type": "MARKET", "reduceOnly": False, "closePosition": False,
                        "side": params["side"], "positionSide": "BOTH", "stopPrice": "0",
                        "workingType": "CONTRACT_PRICE", "priceProtect": False, "origType": "MARKET",
                        "updateTime": 1_700_000_000_000}
            if path.endswith("/userTrades"):
                return [{"symbol": "BTCUSDT", "id": 9, "orderId": 123, "side": "BUY", "price": "60012.30",
                         "qty": "0.002", "realizedPnl": "0", "quoteQty": "120.0246", "commission": "0.04800984",
                         "commissionAsset": "USDT", "time": 1_700_000_000_000, "buyer": True, "maker": False,
                         "positionSide": "BOTH"}]
            if path.endswith("/algoOrder") and method == "POST":
                return {"algoId": next(algo_ids), "clientAlgoId": params["clientAlgoId"], "algoType": "CONDITIONAL",
                        "orderType": params["type"], "symbol": "BTCUSDT", "side": params["side"],
                        "positionSide": "BOTH", "quantity": params["quantity"], "algoStatus": "NEW",
                        "triggerPrice": params["triggerPrice"], "workingType": params["workingType"],
                        "reduceOnly": True, "closePosition": False, "createTime": 1, "updateTime": 1}
            if path.endswith("/algoOpenOrders"):
                return {"code": 200, "msg": "The operation of cancel all open order is done."}
            if path.endswith("/openOrders"):
                return []
            raise AssertionError(f"unexpected request {method} {path}")

        ex.fetch = fetch
        broker = BinanceFuturesBroker(cfg, ex)
        await broker.connect()
        fill = await broker.market_order("BTCUSDT", LONG, 0.0025, 60_000.0)
        ids = await broker.place_protection("BTCUSDT", LONG, fill.qty, 58_950.04, 62_000.04)
        await broker.cancel_protection("BTCUSDT")
        await broker.close()
        return fill, ids

    fill, ids = run(scenario())
    assert all(path.startswith("https://demo-fapi.binance.com/fapi/") for _, path, _, _ in sent)
    by_path = {(m, p.rsplit("/fapi/", 1)[1]): params for m, p, params, _ in sent}
    assert by_path[("POST", "v1/marginType")]["marginType"] == "ISOLATED"
    assert by_path[("POST", "v1/leverage")] | {} and by_path[("POST", "v1/leverage")]["leverage"] == "2"
    order = by_path[("POST", "v1/order")]
    assert (order["symbol"], order["side"], order["type"], order["quantity"]) == ("BTCUSDT", "BUY", "MARKET", "0.002")
    assert order["newClientOrderId"].startswith(CLIENT_ID_PREFIX + "o-") and "reduceOnly" not in order
    assert by_path[("GET", "v1/userTrades")]["orderId"] == "123"
    algo = [params for m, p, params, _ in sent if m == "POST" and p.endswith("/algoOrder")]
    assert [a["type"] for a in algo] == ["STOP_MARKET", "TAKE_PROFIT_MARKET"]
    for a, trigger, tag in zip(algo, ("58950", "62000"), ("sl-", "tp-")):
        assert a["algoType"] == "CONDITIONAL" and a["side"] == "SELL" and a["quantity"] == "0.002"
        assert a["triggerPrice"] == trigger and a["reduceOnly"] == "true" and a["workingType"] == "MARK_PRICE"
        assert a["clientAlgoId"].startswith(CLIENT_ID_PREFIX + tag) and "closePosition" not in a
    assert by_path[("DELETE", "v1/algoOpenOrders")]["symbol"] == "BTCUSDT"
    signed = [params for m, p, params, _ in sent if "signature" in params]
    assert signed and all(params["recvWindow"] == str(cfg.exchange.recv_window_ms) for params in signed)
    assert all(SECRET not in str(params) for _, _, params, _ in sent)
    assert all(h.get("X-MBX-APIKEY") == API_KEY for m, p, params, h in sent if "signature" in params)
    assert fill.qty == pytest.approx(0.002) and fill.price == pytest.approx(60_012.3)
    assert fill.fee == pytest.approx(0.04800984)
    assert ids == ["501", "502"]
