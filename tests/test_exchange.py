"""Binance REST/WebSocket clients against a local fake server (no internet)."""

import asyncio
import hashlib
import hmac
import json
from urllib.parse import parse_qsl

from aiohttp import web
from aiohttp.test_utils import TestServer

from cappatrader.config import ExchangeConfig, Secrets
from cappatrader.data.binance_rest import BinanceREST
from cappatrader.data.binance_ws import BinanceStream, market_streams

H = 3_600_000


def _kline(t: int) -> list:
    return [t, "1", "2", "0.5", "1.5", "10", t + H - 1, "15", 3, "6", "9", "0"]


def _app(store: dict) -> web.Application:
    async def klines(request: web.Request):
        start = int(request.query["startTime"])
        end = int(request.query["endTime"])
        limit = int(request.query["limit"])
        rows = [_kline(t) for t in range(start - start % H, end, H)][:limit]
        store.setdefault("kline_calls", 0)
        store["kline_calls"] += 1
        return web.json_response(rows, headers={"X-MBX-USED-WEIGHT-1M": "42"})

    async def restrictions(request: web.Request):
        store["query"] = request.query_string
        store["key"] = request.headers.get("X-MBX-APIKEY")
        return web.json_response({"enableWithdrawals": False, "ipRestrict": True})

    async def exinfo(request: web.Request):
        return web.json_response({"symbols": [{"symbol": "BTCUSDT", "status": "TRADING", "filters": [
            {"filterType": "LOT_SIZE", "stepSize": "0.00001", "minQty": "0.00001"},
            {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
            {"filterType": "NOTIONAL", "minNotional": "5"}]}]})

    async def ws(request: web.Request):
        sock = web.WebSocketResponse()
        await sock.prepare(request)
        store["ws_query"] = request.query.get("streams")
        await sock.send_str(json.dumps({"stream": "btcusdt@aggTrade", "data": {"p": "100", "q": "1", "m": False}}))
        await sock.close()
        return sock

    app = web.Application()
    app.router.add_get("/api/v3/klines", klines)
    app.router.add_get("/api/v3/exchangeInfo", exinfo)
    app.router.add_get("/sapi/v1/account/apiRestrictions", restrictions)
    app.router.add_get("/stream", ws)
    return app


def test_rest_pagination_signing_and_filters():
    async def scenario():
        store: dict = {}
        server = TestServer(_app(store))
        await server.start_server()
        base = str(server.make_url("")).rstrip("/")
        rest = BinanceREST(ExchangeConfig(rest_base=base), Secrets("my-key", "my-secret", "", ""))
        start = 1_735_689_600_000
        df = await rest.klines_range("BTCUSDT", "1h", start, start + 2500 * H, drop_open_bar=False)
        r = await rest.api_restrictions()
        filters = await rest.exchange_filters(["BTCUSDT"])
        used = rest.limiter.used
        await rest.close()
        await server.close()
        return store, df, r, filters, used

    store, df, r, filters, used = asyncio.run(scenario())
    assert len(df) == 2500 and df.index.is_monotonic_increasing and store["kline_calls"] == 3
    assert used >= 42
    assert r["enableWithdrawals"] is False and store["key"] == "my-key"
    params = dict(parse_qsl(store["query"]))
    unsigned = store["query"].rsplit("&signature=", 1)[0]
    expected = hmac.new(b"my-secret", unsigned.encode(), hashlib.sha256).hexdigest()
    assert params["signature"] == expected and "timestamp" in params
    assert filters["BTCUSDT"]["step_size"] == 1e-5 and filters["BTCUSDT"]["min_notional"] == 5


def test_websocket_stream_dispatch_and_reconnect():
    async def scenario():
        store: dict = {}
        server = TestServer(_app(store))
        await server.start_server()
        ws_base = str(server.make_url("")).rstrip("/").replace("http://", "ws://")
        got, states = [], []
        stop = asyncio.Event()

        async def on_msg(stream, data):
            got.append((stream, data))
            if len(got) >= 2:  # second message arrives only after an automatic reconnect
                stop.set()

        async def on_state(connected):
            states.append(connected)

        stream = BinanceStream(ws_base, market_streams(["BTCUSDT"], "1h"), on_msg, on_state)
        await asyncio.wait_for(stream.run(stop), timeout=15)
        await server.close()
        return store, got, states, stream

    store, got, states, stream = asyncio.run(scenario())
    assert store["ws_query"] == "btcusdt@kline_1h/btcusdt@aggTrade/btcusdt@depth20@100ms"
    assert got[0] == ("btcusdt@aggTrade", {"p": "100", "q": "1", "m": False})
    assert stream.reconnects >= 1 and states[:2] == [True, False]
