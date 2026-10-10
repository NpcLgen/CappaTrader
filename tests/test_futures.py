"""Binance USD-M futures broker against a ccxt-shaped fake exchange (no network)."""

import asyncio
import itertools
import json
import time
from urllib.parse import parse_qsl, urlsplit

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
        self.partials = []  # per market order: (filled fraction, status[, fill price])
        self.hide_new = False  # new orders stay invisible to fetch_order until reveal()
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
        if isinstance(effect, BaseException):
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
        fraction, status, *price = self.partials.pop(0) if self.partials else (1.0, "closed")
        px = price[0] if price else self.fill_price[symbol]
        filled = round(amount * fraction, 12)
        ts = int(time.time() * 1000)
        order = {"id": oid, "clientOrderId": cid, "symbol": symbol, "type": "market", "side": side,
                 "amount": amount, "filled": filled, "average": px, "cost": filled * px, "status": status,
                 "timestamp": ts, "lastTradeTimestamp": ts, "fee": None, "fees": [],
                 "reduceOnly": bool(params.get("reduceOnly")), "info": {"orderId": oid}}
        if self.hide_new:
            order["_hidden"] = True
        self.orders.append(order)
        if status == "open":
            self.open_regular.append(order)
        fee_usdt = filled * px * 0.0005
        fee = fee_usdt if self.fee_currency == "USDT" else fee_usdt / self.tickers["BNB/USDT:USDT"]
        self.trades.append({"id": "t" + oid, "order": oid, "symbol": symbol, "side": side, "amount": filled,
                            "price": px, "cost": filled * px, "timestamp": ts,
                            "fee": {"cost": fee, "currency": self.fee_currency},
                            "fees": [{"cost": fee, "currency": self.fee_currency}]})
        self.pos[symbol] = round(self.pos.get(symbol, 0.0) + sign * filled, 12)
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
            if order.get("_hidden"):
                continue
            if (cid and order["clientOrderId"] == cid) or (not cid and order["id"] == id):
                return dict(order)
        raise _err(ccxt.OrderNotFound, -2013, "Order does not exist.")

    def reveal(self):
        for order in self.orders + self.algo:
            order.pop("_hidden", None)

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
                order["status"] = "canceled"
                return dict(order)
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


def test_demo_env_refused_when_any_endpoint_is_mainnet():
    ex = FakeExchange()
    ex.urls["api"]["fapiPrivateV3"] = "https://fapi.binance.com/fapi/v3"  # left out of the demo switch
    with pytest.raises(LiveTradingRefused, match="mainnet"):
        run(connected(live_cfg("demo"), ex))

    async def flipped_after_connect():
        broker, ex = await connected(live_cfg("demo"))
        ex.urls["api"]["fapiPrivate"] = FAPI
        with pytest.raises(LiveTradingRefused):
            await broker.market_order("BTCUSDT", LONG, 0.002, 60_000.0)
        return ex

    assert "create_order" not in run(flipped_after_connect()).names()


def test_real_ccxt_demo_switch_covers_every_endpoint_offline():
    async def scenario():
        demo = BinanceFuturesBroker(live_cfg("demo"), bf.build_exchange(live_cfg("demo")))
        demo._client()  # enables demo trading and checks every base URL
        urls = dict(demo.ex.urls["api"])
        demo.ex.urls["api"]["fapiPrivateV2"] = "https://fapi.binance.com/fapi/v2"
        with pytest.raises(LiveTradingRefused):
            demo._check_endpoint()
        main = BinanceFuturesBroker(live_cfg("mainnet"), bf.build_exchange(live_cfg("mainnet")))
        main._client()
        await demo.close()
        await main.close()
        return urls

    urls = run(scenario())
    assert {"fapiPrivate", "fapiPrivateV2", "fapiPrivateV3"} <= set(urls)
    assert all("demo-" in u for u in urls.values())


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


def test_partially_filled_working_market_order_is_settled_before_reporting():
    async def scenario():
        ex = FakeExchange()
        broker, _ = await connected(ex=ex)
        ex.partials = [(0.5, "open")]
        fill = await broker.market_order("BTCUSDT", LONG, 0.004, 60_000.0)
        return fill, ex

    fill, ex = run(scenario())
    oid = ex.orders[0]["id"]
    assert (oid, BTC, {}) in ex.args("cancel_order")  # the unfilled rest is cancelled first
    assert ex.orders[0]["status"] == "canceled" and ex.open_regular == []
    assert fill.qty == pytest.approx(0.002) and ex.pos[BTC] == pytest.approx(0.002)


def test_partial_fill_that_cannot_be_settled_is_state_unknown():
    async def scenario():
        ex = FakeExchange()
        broker, _ = await connected(ex=ex)

        async def cancel_down(id, symbol=None, params={}):
            raise ccxt.ExchangeNotAvailable("binanceusdm 503")

        ex.cancel_order = cancel_down
        ex.partials = [(0.5, "open")]
        with pytest.raises(OrderStateUnknown) as info:
            await broker.market_order("BTCUSDT", LONG, 0.004, 60_000.0)
        return info.value, ex

    err, ex = run(scenario())
    assert "kısmen" in str(err) and err.client_id == ex.orders[0]["clientOrderId"]
    assert len(ex.args("create_order")) == 1


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
    assert sleeps == [1.0, 2.0, 4.0]  # the last one precedes the final "was it sent?" lookup


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


def test_order_invisible_until_after_the_last_timeout_is_found_by_the_final_lookup(monkeypatch):
    async def scenario():
        broker, ex = await connected(live_cfg(max_retries=1))
        delays = []

        async def sleep_then_reveal(seconds):  # the exchange shows the order only after some delay
            delays.append(seconds)
            ex.reveal()

        monkeypatch.setattr(bf, "_sleep", sleep_then_reveal)
        ex.hide_new = True
        ex.create_effects = [ccxt.RequestTimeout("binanceusdm timed out"),
                             ("after", ccxt.RequestTimeout("binanceusdm timed out"))]
        return await broker.market_order("BTCUSDT", LONG, 0.002, 60_000.0), ex, delays

    fill, ex, delays = run(scenario())
    assert len(ex.orders) == 1 and fill.qty == pytest.approx(0.002) and ex.pos[BTC] == pytest.approx(0.002)
    assert delays[:2] == [1.0, 2.0]


def test_filled_order_missing_from_the_order_query_is_never_sent_twice():
    async def scenario():
        broker, ex = await connected()
        ex.hide_new = True  # executed, but GET /order lags behind the position
        ex.create_effects = [("after", ccxt.RequestTimeout("binanceusdm timed out"))]
        with pytest.raises(OrderStateUnknown) as info:
            await broker.market_order("BTCUSDT", LONG, 0.002, 60_000.0)
        return info.value, ex

    err, ex = run(scenario())
    assert len(ex.args("create_order")) == 1 and len(ex.orders) == 1 and ex.pos[BTC] == pytest.approx(0.002)
    assert "aynı yönde pozisyon" in str(err)


def test_codeless_transport_error_after_send_is_looked_up_not_reported_as_rejected(caplog):
    async def scenario():
        broker, ex = await connected()
        # aiohttp.ClientPayloadError while reading the response: ccxt raises a bare ExchangeError
        ex.create_effects = [("after", ccxt.ExchangeError(
            f"binanceusdm POST https://demo-fapi.binance.com/fapi/v1/order?signature=abc123 {SECRET}"))]
        return await broker.market_order("BTCUSDT", LONG, 0.002, 60_000.0), ex

    fill, ex = run(scenario())
    assert len(ex.args("create_order")) == 1 and len(ex.orders) == 1
    assert fill.qty == pytest.approx(0.002)
    assert SECRET not in caplog.text and "abc123" not in caplog.text


def test_protection_timeout_after_send_is_not_duplicated():
    async def scenario():
        broker, ex = await connected()
        ex.create_effects = [("after", ccxt.RequestTimeout("binanceusdm POST algoOrder timed out"))]
        return await broker.place_protection("BTCUSDT", LONG, 0.002, 58_000.0, 62_000.0), ex

    ids, ex = run(scenario())
    assert [o["type"] for o in ex.algo] == ["stop_market", "take_profit_market"]
    assert ids == [ex.algo[0]["id"], ex.algo[1]["id"]]
    sl_cid = ex.args("create_order")[0][5]["clientOrderId"]
    assert ex.args("fetch_order")[0][2] == {"clientOrderId": sl_cid, "trigger": True}
    assert len(ex.args("create_order")) == 2


def test_cancellation_while_an_order_is_in_flight_is_not_swallowed_or_resent():
    async def scenario():
        broker, ex = await connected()
        ex.create_effects = [asyncio.CancelledError()]
        with pytest.raises(asyncio.CancelledError):
            await broker.market_order("BTCUSDT", LONG, 0.002, 60_000.0)
        return ex

    ex = run(scenario())
    assert len(ex.args("create_order")) == 1 and "fetch_order" not in ex.names()


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
        ex.create_effects = [ccxt.ExchangeError(
            f'binanceusdm {{"code":-1000,"msg":"boom {SECRET} {API_KEY}"}} signature=abcdef0123')]
        with pytest.raises(FuturesError) as info:
            await broker.market_order("BTCUSDT", LONG, 0.002, 60_000.0)
        return str(info.value), ex

    text, ex = run(scenario())
    assert SECRET not in text and API_KEY not in text and "abcdef0123" not in text
    assert len(ex.args("create_order")) == 1 and "fetch_order" not in ex.names()


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


def test_close_position_completes_a_partially_filled_close():
    async def scenario():
        broker, ex = await connected()
        ex.pos[BTC] = 0.004
        ex.partials = [(0.5, "expired", 59_900.0), (1.0, "closed", 59_800.0)]
        return await broker.close_position("BTCUSDT", LONG, 0.004, 60_000.0), ex

    fill, ex = run(scenario())
    sent = ex.args("create_order")
    assert [(a[2], a[3], a[5]["reduceOnly"]) for a in sent] == [("sell", 0.004, True), ("sell", 0.002, True)]
    assert ex.pos[BTC] == 0
    assert fill.qty == pytest.approx(0.004) and fill.side == SHORT
    assert fill.price == pytest.approx((59_900.0 + 59_800.0) / 2)
    assert fill.fee == pytest.approx(0.002 * 59_900 * 0.0005 + 0.002 * 59_800 * 0.0005)
    assert fill.slippage_bps == pytest.approx(-(fill.price / 60_000 - 1) * 1e4)


def test_held_symbols_outside_the_tradeable_set_are_manageable_but_not_tradeable():
    async def scenario():
        ex = FakeExchange()
        ex.fill_price["OLD/USDT:USDT"] = 1.0
        ex.pos = {ETH: 0.5, "OLD/USDT:USDT": -40.0}
        broker, _ = await connected(live_cfg(symbols=("BTCUSDT", "OLDUSDT")), ex)
        hidden = await broker.positions()
        held = await broker.positions(["BTCUSDT", "ETHUSDT", "OLDUSDT", "PAXGUSDT"])
        codes = []
        for sym in ("ETHUSDT", "OLDUSDT"):
            with pytest.raises(OrderRejected) as info:
                await broker.market_order(sym, LONG, 1.0, 3_000.0)
            codes.append(info.value.code)
        ids = await broker.place_protection("ETHUSDT", LONG, 0.5, 2_800.0, None)
        await broker.cancel_protection("ETHUSDT")
        eth = await broker.close_position("ETHUSDT", LONG, 0.5, 3_000.0)
        old = await broker.close_position("OLDUSDT", SHORT, 40.0, 1.0)
        exit_fill = await broker.last_close_fill("ETHUSDT", 0, LONG)
        return hidden, held, codes, ids, eth, old, exit_fill, ex

    hidden, held, codes, ids, eth, old, exit_fill, ex = run(scenario())
    assert hidden == {}
    assert set(held) == {"ETHUSDT", "OLDUSDT"} and held["OLDUSDT"].side == SHORT
    assert codes == ["unavailable", "unavailable"]
    assert len(ids) == 1 and ex.algo == []
    assert eth.qty == pytest.approx(0.5) and old.qty == pytest.approx(40.0) and old.side == LONG
    assert all(a[5].get("reduceOnly") for a in ex.args("create_order") if "stopLossPrice" not in a[5])
    assert ex.pos[ETH] == 0 and ex.pos["OLD/USDT:USDT"] == 0
    assert exit_fill.qty == pytest.approx(0.5)


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


# -- real ccxt over a fake Binance HTTP session --------------------------------------------------------
# Only aiohttp's session is replaced: the installed ccxt binanceusdm builds, signs and routes every request,
# maps every Binance error body (handle_errors) and parses every response. Bodies follow Binance's USD-M
# shapes (fapi v1 order / algoOrder / userTrades, v3 account / positionRisk) and spot /api/v3/account.
UNSIGNED = {"/fapi/v1/exchangeInfo", "/fapi/v1/time", "/fapi/v1/ticker/24hr"}
EXCHANGE_INFO = {"timezone": "UTC", "serverTime": 1_700_000_000_000, "futuresType": "U_MARGINED", "rateLimits": [],
                 "exchangeFilters": [], "assets": [],
                 "symbols": [*RAW_MARKETS, {**_raw_market("BTCUSDT_261225", "BTC"), "pair": "BTCUSDT",
                                            "contractType": "CURRENT_QUARTER", "deliveryDate": 1_798_185_600_000}]}
HOSTS = {"demo": ("demo-fapi.binance.com", "demo-api.binance.com"), "mainnet": ("fapi.binance.com", "api.binance.com")}


def _binance_err(status, code, msg):
    return status, {"code": code, "msg": msg}


class _HTTPResponse:
    def __init__(self, status, payload):
        self.status, self.reason = status, "OK" if status < 400 else "Bad Request"
        self.headers = {"Content-Type": "application/json"}
        self._text = payload if isinstance(payload, str) else json.dumps(payload)

    async def text(self, errors="strict"):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class BinanceHTTP:
    """aiohttp-session stand-in answering like Binance USD-M; records (method, host, path, params, headers, raw)."""

    def __init__(self, dual=False):
        self.headers = {}  # ccxt merges session.headers into every request
        self.requests = []
        self.effects = {}  # (method, path) -> [exception | fn(http, params) -> None | (status, body) | exception]
        self.dual = dual
        self.margin_reply = None
        self.lev = {}
        self.pos, self.entry, self.orders, self.algos, self.trades = {}, {}, {}, {}, []
        self.px = {"BTCUSDT": "60012.30", "ETHUSDT": "3001.25", "BNBUSDT": "600.110"}
        self.fee_asset = "USDT"
        self.open_order_margin = "10.0"
        self._ids, self._algo_ids = itertools.count(4_000_000_001), itertools.count(1_000_001)

    def get(self, url, data=None, headers=None, **_):
        return self._call("GET", url, data, headers)

    def post(self, url, data=None, headers=None, **_):
        return self._call("POST", url, data, headers)

    def delete(self, url, data=None, headers=None, **_):
        return self._call("DELETE", url, data, headers)

    async def close(self):
        pass

    def _call(self, method, url, data, headers):
        parts = urlsplit(str(url))
        raw = parts.query or (data.decode() if data else "")
        params = dict(parse_qsl(raw, keep_blank_values=True))
        self.requests.append({"method": method, "host": parts.netloc, "path": parts.path, "params": params,
                              "headers": dict(headers or {}), "raw": raw})
        queue = self.effects.get((method, parts.path))
        if queue:
            effect = queue.pop(0)
            out = effect(self, params) if callable(effect) else effect
            if isinstance(out, BaseException):
                raise out
            if out is not None:
                return _HTTPResponse(*out)
        return _HTTPResponse(*self.route(method, parts.path, params))

    def sent(self, method, path):
        drop = {"timestamp", "signature", "recvWindow"}
        return [{k: v for k, v in r["params"].items() if k not in drop}
                for r in self.requests if r["method"] == method and r["path"] == path]

    def landed_then(self, exc):
        """Effect: the request reaches the matching engine, then the response is lost."""
        return lambda http, params: (http.route("POST", "/fapi/v1/" + ("algoOrder" if "algoType" in params
                                                                        else "order"), params), exc)[1]

    def open_order(self, symbol, cid):
        oid = next(self._ids)
        self.orders[oid] = {"orderId": oid, "symbol": symbol, "status": "NEW", "clientOrderId": cid, "price": "50000",
                            "avgPrice": "0", "origQty": "0.002", "executedQty": "0", "cumQuote": "0",
                            "type": "LIMIT", "side": "BUY", "positionSide": "BOTH", "updateTime": 1}
        return oid

    def route(self, method, path, p):
        now = int(time.time() * 1000)
        if path == "/fapi/v1/exchangeInfo":
            return 200, EXCHANGE_INFO
        if path == "/fapi/v1/time":
            return 200, {"serverTime": now}
        if path == "/fapi/v1/positionSide/dual":
            return 200, {"dualSidePosition": self.dual}
        if path == "/fapi/v1/marginType":
            return self.margin_reply or (200, {"code": 200, "msg": "success"})
        if path == "/fapi/v1/leverage":
            self.lev[p["symbol"]] = int(p["leverage"])
            return 200, {"leverage": int(p["leverage"]), "maxNotionalValue": "480000000", "symbol": p["symbol"]}
        if path == "/fapi/v1/leverageBracket":
            return 200, [{"symbol": s, "brackets": [{"bracket": 1, "initialLeverage": 125, "notionalCap": 50000,
                                                      "notionalFloor": 0, "maintMarginRatio": 0.004, "cum": 0.0}]}
                         for s in self.px]
        if path == "/fapi/v3/account":
            return 200, {"totalWalletBalance": "1000.0", "availableBalance": "800.0", "positions": [],
                         "assets": [{"asset": "USDT", "walletBalance": "1000.00000000",
                                     "unrealizedProfit": "-12.50000000", "marginBalance": "987.50000000",
                                     "availableBalance": "800.00000000", "maxWithdrawAmount": "800.0",
                                     "updateTime": 1_700_000_000_000},
                                    {"asset": "USDC", "walletBalance": "50.0", "unrealizedProfit": "0.0",
                                     "marginBalance": "50.0", "availableBalance": "50.0", "updateTime": 1}]}
        if path == "/api/v3/account":
            return 200, {"canTrade": True, "accountType": "SPOT", "updateTime": now, "permissions": ["SPOT"],
                         "balances": [{"asset": "BTC", "free": "0.001", "locked": "0.0"},
                                      {"asset": "USDT", "free": "5.00000000", "locked": "1.00000000"}]}
        if path == "/fapi/v3/positionRisk":
            rows = []
            for sym, amt in self.pos.items():
                if amt:
                    entry, mark = float(self.entry[sym]), float(self.px[sym])
                    notional, im = amt * mark, abs(amt * mark) / self.lev.get(sym, 1)
                    rows.append({"symbol": sym, "positionSide": "BOTH", "positionAmt": f"{amt:.3f}",
                                 "entryPrice": f"{entry}", "breakEvenPrice": f"{entry}", "markPrice": f"{mark}",
                                 "unRealizedProfit": f"{(mark - entry) * amt:.8f}", "liquidationPrice": "0",
                                 "isolatedMargin": f"{im:.8f}", "notional": f"{notional:.8f}", "marginAsset": "USDT",
                                 "isolatedWallet": f"{im:.8f}", "positionInitialMargin": f"{im:.8f}",
                                 "openOrderInitialMargin": self.open_order_margin,
                                 "initialMargin": f"{im + float(self.open_order_margin):.8f}", "maintMargin": "0.48",
                                 "adl": 1, "bidNotional": "0", "askNotional": "0", "updateTime": now})
            rows.append({"symbol": "BNBUSDT", "positionSide": "BOTH", "positionAmt": "0.00", "entryPrice": "0.0",
                         "markPrice": "600.11", "unRealizedProfit": "0", "notional": "0", "isolatedMargin": "0",
                         "initialMargin": "0", "updateTime": 0})  # flat rows are filtered out by ccxt
            return 200, rows
        if path == "/fapi/v1/order":
            return self._order(method, p, now)
        if path == "/fapi/v1/algoOrder":
            return self._algo(method, p, now)
        if path == "/fapi/v1/algoOpenOrders" and method == "DELETE":
            for a in self.algos.values():
                if a["symbol"] == p["symbol"] and a["algoStatus"] == "NEW":
                    a["algoStatus"] = "CANCELED"
            return 200, {"code": 200, "msg": "The operation of cancel all open order is done."}
        if path == "/fapi/v1/openOrders":
            return 200, [o for o in self.orders.values() if o["symbol"] == p["symbol"] and o["status"] == "NEW"]
        if path == "/fapi/v1/userTrades":
            return 200, [t for t in self.trades if t["symbol"] == p["symbol"]
                         and str(t["orderId"]) == p.get("orderId", str(t["orderId"]))
                         and t["time"] >= int(p.get("startTime", 0))]
        if path == "/fapi/v1/ticker/24hr":
            last = self.px[p["symbol"]]
            return 200, {"symbol": p["symbol"], "lastPrice": last, "openPrice": last, "highPrice": last,
                         "lowPrice": last, "volume": "1", "quoteVolume": last, "closeTime": now}
        raise AssertionError(f"unexpected Binance request {method} {path} {p}")

    def _order(self, method, p, now):
        if method == "GET":
            key = "clientOrderId" if "origClientOrderId" in p else "orderId"
            hits = [o for o in self.orders.values() if str(o[key]) == p.get("origClientOrderId", p.get("orderId"))]
            return (200, hits[0]) if hits else _binance_err(400, -2013, "Order does not exist.")
        if method == "DELETE":
            o = self.orders.get(int(p["orderId"]))
            if o is None or o["status"] != "NEW":
                return _binance_err(400, -2011, "Unknown order sent.")
            o["status"] = "CANCELED"
            return 200, o
        sym, side, qty = p["symbol"], p["side"], float(p["quantity"])
        sign, cur = (1 if side == "BUY" else -1), self.pos.get(sym, 0.0)
        if p.get("reduceOnly") == "true" and (cur == 0 or cur * sign > 0 or qty > abs(cur) + 1e-12):
            return _binance_err(400, -2022, "ReduceOnly Order is rejected.")
        oid, px = next(self._ids), self.px[sym]
        o = {"orderId": oid, "symbol": sym, "status": "FILLED", "clientOrderId": p["newClientOrderId"],
             "price": "0.00", "avgPrice": px, "origQty": p["quantity"], "executedQty": p["quantity"],
             "cumQty": p["quantity"], "cumQuote": f"{qty * float(px):.5f}", "timeInForce": "GTC", "type": "MARKET",
             "reduceOnly": p.get("reduceOnly") == "true", "closePosition": False, "side": side,
             "positionSide": "BOTH", "stopPrice": "0.00", "workingType": "CONTRACT_PRICE", "priceProtect": False,
             "origType": "MARKET", "priceMatch": "NONE", "selfTradePreventionMode": "EXPIRE_MAKER",
             "goodTillDate": 0, "updateTime": now}
        self.orders[oid] = o
        self.pos[sym] = round(cur + sign * qty, 9)
        self.entry.setdefault(sym, px)
        fee = qty * float(px) * 0.0005 / (float(self.px["BNBUSDT"]) if self.fee_asset == "BNB" else 1.0)
        self.trades.append({"symbol": sym, "id": len(self.trades) + 1, "orderId": oid, "side": side, "price": px,
                            "qty": p["quantity"], "realizedPnl": "0", "quoteQty": o["cumQuote"],
                            "commission": f"{fee:.8f}", "commissionAsset": self.fee_asset, "time": now,
                            "buyer": side == "BUY", "maker": False, "positionSide": "BOTH"})
        return 200, o

    def _algo(self, method, p, now):
        if method == "GET":
            key = "clientAlgoId" if "clientAlgoId" in p else "algoId"
            hits = [a for a in self.algos.values() if str(a[key]) == p[key]]
            return (200, hits[0]) if hits else _binance_err(400, -2013, "Order does not exist.")
        if method == "DELETE":
            a = self.algos.get(int(p["algoId"]))
            if a is None or a["algoStatus"] != "NEW":
                return _binance_err(400, -20124, "Invalid algo id or it has been completed.")
            a["algoStatus"] = "CANCELED"
            return 200, {"algoId": a["algoId"], "clientAlgoId": a["clientAlgoId"], "code": "200", "msg": "success"}
        aid = next(self._algo_ids)
        self.algos[aid] = {"algoId": aid, "clientAlgoId": p["clientAlgoId"], "algoType": p["algoType"],
                           "orderType": p["type"], "symbol": p["symbol"], "side": p["side"], "positionSide": "BOTH",
                           "timeInForce": "GTC", "quantity": p["quantity"], "algoStatus": "NEW",
                           "triggerPrice": p["triggerPrice"], "price": "0.00", "workingType": p["workingType"],
                           "priceMatch": "NONE", "closePosition": False, "priceProtect": False,
                           "reduceOnly": p.get("reduceOnly") == "true", "createTime": now, "updateTime": now,
                           "triggerTime": 0, "goodTillDate": 0}
        return 200, self.algos[aid]


def _real_client(cfg, http):
    ex = bf.build_exchange(cfg)
    ex.session, ex.own_session, ex.enableRateLimit = http, False, False
    ex.timeout_on_exit = 0  # close() waits for aiohttp's sockets; there are none here
    return ex


async def _real_broker(cfg, http):
    broker = BinanceFuturesBroker(cfg, _real_client(cfg, http))
    try:
        await broker.connect()
    except BaseException:
        await broker.close()
        raise
    return broker


def _signed_ok(http):
    signed = [r for r in http.requests if r["path"] not in UNSIGNED]
    assert signed and all({"timestamp", "signature"} <= set(r["params"]) and r["params"]["recvWindow"] == "5000"
                          and r["headers"].get("X-MBX-APIKEY") == API_KEY for r in signed)
    assert all(SECRET not in r["raw"] and SECRET not in str(r["headers"]) for r in http.requests)


@pytest.mark.parametrize("env", ["demo", "mainnet"])
def test_real_ccxt_requests_go_to_the_env_hosts_with_binance_params(env):
    cfg = live_cfg(env, symbols=("BTCUSDT", "ETHUSDT", "BNBUSDT", "OLDUSDT"), leverage=3, margin_mode="isolated")
    http = BinanceHTTP()
    since = int(time.time() * 1000) - 60_000

    async def scenario():
        broker = await _real_broker(cfg, http)
        try:
            out = {"bal": await broker.balance(), "spot": await broker.spot_usdt_balance()}
            out["open"] = await broker.market_order("BTCUSDT", LONG, 0.0029999, 60_000.0)
            out["ids"] = await broker.place_protection("BTCUSDT", LONG, out["open"].qty, 58_950.04, 62_000.06)
            await broker.cancel_protection("BTCUSDT", out["ids"])
            http.fee_asset = "BNB"
            out["short"] = await broker.market_order("ETHUSDT", SHORT, 0.05, 3_000.0)
            http.px["ETHUSDT"] = "2950.00"
            out["pos"] = await broker.positions()
            out["short_ids"] = await broker.place_protection("ETHUSDT", SHORT, 0.05, 3_100.123, 2_900.456,
                                                             "CONTRACT_PRICE")
            await broker.place_protection("BTCUSDT", LONG, out["open"].qty, 58_950.04)
            out["bot_oid"] = http.open_order("BTCUSDT", CLIENT_ID_PREFIX + "o-left")
            out["manual_oid"] = http.open_order("BTCUSDT", "web_manual_order")
            await broker.cancel_protection("BTCUSDT")
            out["close"] = await broker.close_position("BTCUSDT", LONG, out["open"].qty, 60_100.0)
            out["again"] = await broker.close_position("BTCUSDT", LONG, out["open"].qty, 60_100.0)
            out["exit"] = await broker.last_close_fill("BTCUSDT", since, LONG)
            return broker, out
        finally:
            await broker.close()

    broker, out = run(scenario())
    fapi_host, spot_host = HOSTS[env]
    # (1) every futures call on the env's fapi host, the spot balance on the env's spot host
    assert all(r["host"] == fapi_host for r in http.requests if r["path"].startswith("/fapi/"))
    assert [(r["host"], r["path"]) for r in http.requests if not r["path"].startswith("/fapi/")] == \
        [(spot_host, "/api/v3/account")]
    _signed_ok(http)
    # connect: markets, clock, one-way check, margin type + leverage per tradeable perpetual only
    assert [(r["method"], r["path"]) for r in http.requests[:3]] == [
        ("GET", "/fapi/v1/exchangeInfo"), ("GET", "/fapi/v1/time"), ("GET", "/fapi/v1/positionSide/dual")]
    assert http.sent("POST", "/fapi/v1/marginType") == [{"symbol": s, "marginType": "ISOLATED"}
                                                        for s in ("BTCUSDT", "ETHUSDT", "BNBUSDT")]
    assert http.sent("POST", "/fapi/v1/leverage") == [{"symbol": s, "leverage": "3"}
                                                      for s in ("BTCUSDT", "ETHUSDT", "BNBUSDT")]
    assert set(broker.unavailable) == {"OLDUSDT"}
    # (7) balances
    assert out["bal"] == FuturesBalance(wallet=1000.0, available=800.0, unrealized=-12.5, equity=987.5)
    assert http.sent("GET", "/fapi/v3/account") == [{}] and out["spot"] == pytest.approx(6.0)
    # (2) MARKET open and (3) reduce-only close on /fapi/v1/order, quantity truncated to the step
    orders = http.sent("POST", "/fapi/v1/order")
    opens = [o for o in orders if o["newClientOrderId"].startswith(CLIENT_ID_PREFIX + "o-")]
    assert opens[0] == {"symbol": "BTCUSDT", "side": "BUY", "type": "MARKET", "quantity": "0.002",
                        "newOrderRespType": "RESULT", "newClientOrderId": opens[0]["newClientOrderId"]}
    assert opens[1] == {**opens[0], "symbol": "ETHUSDT", "side": "SELL", "quantity": "0.05",
                        "newClientOrderId": opens[1]["newClientOrderId"]}
    closes = [o for o in orders if o["newClientOrderId"].startswith(CLIENT_ID_PREFIX + "c-")]
    assert len(closes) == 2 and closes[0] == {**opens[0], "side": "SELL", "reduceOnly": "true",
                                              "newClientOrderId": closes[0]["newClientOrderId"]}
    assert out["close"].qty == pytest.approx(0.002) and out["close"].side == SHORT
    assert out["again"] is None  # -2022 through ccxt's error mapping, then positionRisk shows nothing left
    assert http.sent("GET", "/fapi/v1/userTrades")[0] == {"symbol": "BTCUSDT", "orderId": "4000000001"}
    assert out["open"].fee == pytest.approx(0.002 * 60_012.3 * 0.0005)
    assert out["short"].fee == pytest.approx(0.05 * 3_001.25 * 0.0005, rel=1e-5)  # BNB fee at BNB/USDT:USDT
    assert {"symbol": "BNBUSDT"} in http.sent("GET", "/fapi/v1/ticker/24hr")
    # (7) positionRisk v3: one-way positionAmt sign, unRealizedProfit, leverage without open-order margin
    assert {"symbol": "BTCUSDT"} not in http.sent("GET", "/fapi/v3/positionRisk")
    assert out["pos"] == {
        "BTCUSDT": ExchangePosition("BTCUSDT", LONG, 0.002, 60_012.3, 0.0, 3.0),
        "ETHUSDT": ExchangePosition("ETHUSDT", SHORT, 0.05, 3_001.25, pytest.approx(2.5625), 3.0)}
    # (4) protection on the algo-order API: STOP_MARKET / TAKE_PROFIT_MARKET, reduce-only qty, tick-rounded trigger
    algo = http.sent("POST", "/fapi/v1/algoOrder")
    base = {"algoType": "CONDITIONAL", "symbol": "BTCUSDT", "side": "SELL", "quantity": "0.002", "reduceOnly": "true",
            "workingType": "MARK_PRICE", "newOrderRespType": "RESULT"}
    assert algo[0] == {**base, "type": "STOP_MARKET", "triggerPrice": "58950", "clientAlgoId": algo[0]["clientAlgoId"]}
    assert algo[1] == {**base, "type": "TAKE_PROFIT_MARKET", "triggerPrice": "62000.1",
                       "clientAlgoId": algo[1]["clientAlgoId"]}
    short_base = {**base, "symbol": "ETHUSDT", "side": "BUY", "quantity": "0.05", "workingType": "CONTRACT_PRICE"}
    assert algo[2] == {**short_base, "type": "STOP_MARKET", "triggerPrice": "3100.12",  # short stop above entry
                       "clientAlgoId": algo[2]["clientAlgoId"]}
    assert algo[3] == {**short_base, "type": "TAKE_PROFIT_MARKET", "triggerPrice": "2900.46",
                       "clientAlgoId": algo[3]["clientAlgoId"]}
    assert [a["clientAlgoId"][:9] for a in algo] == ["cappa-sl-", "cappa-tp-", "cappa-sl-", "cappa-tp-", "cappa-sl-"]
    assert all(o["type"] == "MARKET" for o in orders)  # no conditional order on the legacy endpoint
    assert out["ids"] == ["1000001", "1000002"] and out["short_ids"] == ["1000003", "1000004"]
    # (5) cancel: by algoId, or all algo orders of the symbol plus only the bot's regular open orders
    assert http.sent("DELETE", "/fapi/v1/algoOrder") == [{"symbol": "BTCUSDT", "algoId": i} for i in out["ids"]]
    assert http.sent("DELETE", "/fapi/v1/algoOpenOrders") == [{"symbol": "BTCUSDT"}]
    assert http.sent("GET", "/fapi/v1/openOrders") == [{"symbol": "BTCUSDT"}]
    assert http.sent("DELETE", "/fapi/v1/order") == [{"symbol": "BTCUSDT", "orderId": str(out["bot_oid"])}]
    assert http.orders[out["manual_oid"]]["status"] == "NEW" and http.algos[1_000_005]["algoStatus"] == "CANCELED"
    # last_close_fill: userTrades from the entry time, at most 1000 rows, no endTime (inside the 7-day window)
    assert http.sent("GET", "/fapi/v1/userTrades")[-1] == {"symbol": "BTCUSDT", "startTime": str(since),
                                                           "limit": "1000"}
    assert out["exit"].side == SHORT and out["exit"].qty == pytest.approx(0.002)


@pytest.mark.parametrize("env", ["demo", "mainnet"])
def test_real_ccxt_ambiguous_sends_are_looked_up_by_client_id(env):
    cfg = live_cfg(env, symbols=("BTCUSDT",))
    http = BinanceHTTP()
    order, algo = ("POST", "/fapi/v1/order"), ("POST", "/fapi/v1/algoOrder")

    async def scenario():
        broker = await _real_broker(cfg, http)
        try:
            http.effects[order] = [http.landed_then(asyncio.TimeoutError())]  # -> ccxt RequestTimeout
            first = await broker.market_order("BTCUSDT", LONG, 0.002, 60_000.0)
            http.effects[order] = [lambda h, p: (h.route(*order, p), _binance_err(
                408, -1007, "Timeout waiting for response from backend server. Send status unknown; "
                            "execution status unknown."))[1]]
            second = await broker.market_order("BTCUSDT", LONG, 0.002, 60_000.0)
            http.effects[algo] = [http.landed_then(asyncio.TimeoutError())]
            landed = await broker.place_protection("BTCUSDT", LONG, 0.004, 58_000.0)
            http.effects[algo] = [asyncio.TimeoutError()]  # lost before reaching Binance
            resent = await broker.place_protection("BTCUSDT", LONG, 0.004, 58_000.0)
            http.effects[order] = [_binance_err(400, -1021, "Timestamp for this request is outside of the "
                                                            "recvWindow.")]
            third = await broker.market_order("BTCUSDT", SHORT, 0.004, 60_000.0, reduce_only=True)
            return first, second, landed, resent, third
        finally:
            await broker.close()

    first, second, landed, resent, third = run(scenario())
    assert all(r["host"] == HOSTS[env][0] for r in http.requests)
    sent = http.sent(*order)
    cids = [o["newClientOrderId"] for o in sent]
    assert cids[:2] == [first_cid := cids[0], cids[1]] and len(set(cids[:2])) == 2  # each landed order sent once
    assert http.sent("GET", "/fapi/v1/order") == [{"symbol": "BTCUSDT", "origClientOrderId": cids[0]},
                                                  {"symbol": "BTCUSDT", "origClientOrderId": cids[1]},
                                                  {"symbol": "BTCUSDT", "origClientOrderId": cids[2]}]
    assert first_cid.startswith(CLIENT_ID_PREFIX) and first.qty == second.qty == pytest.approx(0.002)
    algo_sent = [a["clientAlgoId"] for a in http.sent(*algo)]
    assert algo_sent == [algo_sent[0], algo_sent[1], algo_sent[1]]  # landed once; lost one re-sent with its id
    assert http.sent("GET", "/fapi/v1/algoOrder") == [{"symbol": "BTCUSDT", "clientAlgoId": i}
                                                      for i in algo_sent[:2]]
    assert landed == ["1000001"] and resent == ["1000002"] and len(http.algos) == 2
    # -1021 (rejected, clock drift): resync from the fapi clock, then the same client id is sent again
    assert cids[2] == cids[3] and ("GET", "/fapi/v1/time") in [(r["method"], r["path"]) for r in http.requests[3:]]
    assert third.qty == pytest.approx(0.004) and http.pos["BTCUSDT"] == 0


@pytest.mark.parametrize("env", ["demo", "mainnet"])
def test_real_ccxt_binance_error_bodies_map_to_broker_outcomes(env):
    def cfg():
        return live_cfg(env, symbols=("BTCUSDT", "ETHUSDT"), margin_mode="isolated")

    async def connect_with(**attrs):
        http = BinanceHTTP()
        for key, value in attrs.items():
            setattr(http, key, value)
        try:
            broker = await _real_broker(cfg(), http)
        except FuturesError as e:
            return http, e
        await broker.close()
        return http, broker

    async def order_with(path, reply, call):
        http = BinanceHTTP()
        broker = await _real_broker(cfg(), http)
        try:
            await broker.market_order("BTCUSDT", LONG, 0.002, 60_000.0)
            if reply is not None:
                http.effects[("POST", path)] = [reply]
            n = len(http.requests)
            try:
                return http, await call(broker), http.requests[n:]
            except FuturesError as e:
                return http, e, http.requests[n:]
        finally:
            await broker.close()

    async def scenario():
        out = {}
        for name, reply in {"exact": _binance_err(400, -4046, "No need to change margin type."),
                            "variant": _binance_err(400, -4046, "No need to change margin type"),
                            "has_position": _binance_err(400, -4048, "Margin type cannot be changed if there "
                                                                     "exists position."),
                            "multi_assets": _binance_err(400, -4168, "Unable to adjust to isolated-margin mode "
                                                                     "under the Multi-Assets mode.")}.items():
            out[name] = await connect_with(margin_reply=reply)
        out["hedge"] = await connect_with(dual=True)
        out["auth"] = await connect_with(effects={("GET", "/fapi/v1/positionSide/dual"): [
            _binance_err(401, -2015, "Invalid API-key, IP, or permissions for action.")]})
        out["margin"] = await order_with("/fapi/v1/order", _binance_err(400, -2019, "Margin is insufficient."),
                                         lambda b: b.market_order("BTCUSDT", LONG, 0.002, 60_000.0))
        for code, text in ((-2021, "Order would immediately trigger."),
                           (-4142, "REJECT: take profit or stop order will be triggered immediately.")):
            out[code] = await order_with("/fapi/v1/algoOrder", _binance_err(400, code, text),
                                         lambda b: b.place_protection("BTCUSDT", LONG, 0.002, 61_000.0))
        out["gone"] = await order_with("/fapi/v1/algoOrder", None,  # DELETE of an unknown/finished algo order
                                       lambda b: b.cancel_protection("BTCUSDT", ["1000999"]))
        return out

    out = run(scenario())
    for name in ("exact", "variant", "has_position"):
        http, broker = out[name]
        assert broker.symbol_map == {"BTCUSDT": BTC, "ETHUSDT": ETH}
        assert len(http.sent("POST", "/fapi/v1/leverage")) == 2
    assert out["exact"][1].warnings == out["variant"][1].warnings == []
    assert any("marjin modu" in w for w in out["has_position"][1].warnings)
    assert isinstance(out["multi_assets"][1], LiveTradingRefused) and "Multi-Assets" in str(out["multi_assets"][1])
    assert isinstance(out["hedge"][1], LiveTradingRefused) and str(out["hedge"][1]) == bf.HEDGE_MODE_MESSAGE
    http, refused = out["auth"]
    assert isinstance(refused, LiveTradingRefused) and len(http.sent("GET", "/fapi/v1/positionSide/dual")) == 1
    http, rejected, reqs = out["margin"]
    assert isinstance(rejected, OrderRejected) and rejected.code == "insufficient_funds"
    assert [(r["method"], r["path"]) for r in reqs] == [("POST", "/fapi/v1/order")]  # never retried or looked up
    for code in (-2021, -4142):
        http, rejected, reqs = out[code]
        assert isinstance(rejected, OrderRejected) and rejected.code == "would_trigger"
        assert [(r["method"], r["path"]) for r in reqs] == [("POST", "/fapi/v1/algoOrder")]
    http, result, reqs = out["gone"]
    assert result is None and [(r["method"], r["path"], r["params"]["algoId"]) for r in reqs] == [
        ("DELETE", "/fapi/v1/algoOrder", "1000999")]  # -20124 (already triggered or cancelled) is ignored
    assert all(r["host"] == HOSTS[env][0] for _, v in out.items() for r in v[0].requests)


def test_real_ccxt_paper_mode_balance_reads_the_mainnet_account_read_only():
    cfg = Config()  # paper mode; live.env "demo" must not matter here
    cfg.account.balance_source = "binance"
    cfg.secrets = Secrets(binance_api_key=API_KEY, binance_secret_key=SECRET)
    out = {}
    for wallet in ("futures", "spot"):
        cfg.account.binance_wallet = wallet
        http = BinanceHTTP()
        out[wallet] = (run(fetch_binance_balance(cfg, _real_client(cfg, http))), http)
    assert out["futures"][0] == pytest.approx(987.5) and out["spot"][0] == pytest.approx(6.0)
    for wallet, account in (("futures", ("fapi.binance.com", "/fapi/v3/account")),
                            ("spot", ("api.binance.com", "/api/v3/account"))):
        http = out[wallet][1]
        assert [(r["host"], r["path"]) for r in http.requests] == [
            ("fapi.binance.com", "/fapi/v1/exchangeInfo"), ("fapi.binance.com", "/fapi/v1/time"), account]
        assert all(r["method"] == "GET" for r in http.requests)
        _signed_ok(http)
