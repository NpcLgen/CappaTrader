import asyncio

from aiohttp import web
from aiohttp.test_utils import TestServer

from cappatrader.models import LONG, Decision, Fill, Position, Trade
from cappatrader.notify import messages as msg
from cappatrader.notify.telegram import TelegramNotifier, split_message


def _pos() -> Position:
    return Position("ETHUSDT", LONG, 0.0852, 2345.12, "2025-01-01T10:00:00+00:00", 2290.40, 2290.40, 54.72,
                    2345.12, "TREND_UP", 71.0, 0.2, 0.05, ["Rejim: Trend ↑", "▲ EMA20 > EMA50 (+1.40 ATR)"])


def test_entry_message_contains_required_fields():
    pos = _pos()
    fill = Fill("ETHUSDT", 1, pos.qty, pos.entry_price, 2344.5, 0.2, 2.6, 251, None)
    d = Decision("ETHUSDT", None, "ENTER_LONG", 0.42, 71.0, "TREND_UP", "NORMAL", 2344.5, 27.36)
    text = msg.entry_message(pos, fill, d, ["Kelly notu", "Stop notu"], 1002.31, 800.2, 1000, 1)
    for needle in ("ALIŞ", "ETH/USDT", "Neden alındı", "EMA20 &gt; EMA50", "Pozisyon boyutu", "Güncel bakiye",
                   "1,002.31 USDT", "slippage 2.6 bps", "Stop"):
        assert needle in text, needle


def test_exit_message_contains_pnl_balance_and_reason():
    t = Trade("ETHUSDT", 1, 0.0852, 2345.12, 2401.30, "2025-01-01T10:00:00+00:00", "2025-01-02T00:15:00+00:00",
              4.79, 0.4, 0.1, 4.39, 0.95, "İzleyen stop (trailing) tetiklendi", "TREND_UP", ["Rejim: Trend ↑"])
    fill = Fill("ETHUSDT", -1, t.qty, t.exit_price, 2401.9, 0.2, 2.5, 250, None)
    text = msg.exit_message(t, fill, 1004.39, 1004.39, 1000, 0, [t])
    for needle in ("SATIŞ", "P&amp;L: +4.39 USDT", "+0.95R", "Neden satıldı", "İzleyen stop", "Güncel bakiye",
                   "14 sa 15 dk", "kazanma %100"):
        assert needle in text, needle


def test_messages_escape_html():
    assert "&lt;script&gt;" in msg.alert_message("x", "<script>")


def test_split_long_message():
    parts = split_message("\n".join(["a" * 100] * 100), limit=1000)
    assert all(len(p) <= 1000 for p in parts) and len(parts) > 1


def test_notifier_posts_and_retries_on_429():
    async def scenario():
        received, calls = [], {"n": 0}

        async def handler(request: web.Request):
            calls["n"] += 1
            if calls["n"] == 1:
                return web.json_response({"ok": False, "parameters": {"retry_after": 0}}, status=429)
            received.append((request.match_info["token"], await request.json()))
            return web.json_response({"ok": True})

        app = web.Application()
        app.router.add_post("/bot{token}/sendMessage", handler)
        server = TestServer(app)
        await server.start_server()
        tg = TelegramNotifier("123:ABC", "42", api_base=str(server.make_url("")).rstrip("/"), min_interval_s=0)
        await tg.start()
        tg.notify("<b>merhaba</b>")
        await tg.stop()
        await server.close()
        return received, tg

    received, tg = asyncio.run(scenario())
    assert received[0][0] == "123:ABC"
    body = received[0][1]
    assert body["chat_id"] == "42" and body["parse_mode"] == "HTML" and body["text"] == "<b>merhaba</b>"
    assert tg.sent == 1 and tg.failed == 0


def test_disabled_notifier_does_not_send():
    tg = TelegramNotifier("", "", enabled=True)
    assert not tg.enabled
    assert asyncio.run(tg.send_now("x")) is False


def test_html_error_page_does_not_kill_notifier():
    """Telegram's front end can answer 502 with an HTML page; the worker must survive and retry."""

    async def scenario():
        calls = {"n": 0}
        received = []

        async def handler(request: web.Request):
            calls["n"] += 1
            if calls["n"] == 1:
                return web.Response(status=502, text="<html><body>Bad Gateway</body></html>", content_type="text/html")
            received.append(await request.json())
            return web.json_response({"ok": True})

        app = web.Application()
        app.router.add_post("/bot{token}/sendMessage", handler)
        server = TestServer(app)
        await server.start_server()
        tg = TelegramNotifier("123:ABC", "42", api_base=str(server.make_url("")).rstrip("/"), min_interval_s=0)
        await tg.start()
        tg.notify("birinci")
        tg.notify("ikinci")
        await tg.stop(flush_timeout=20)
        await server.close()
        return received, tg

    received, tg = asyncio.run(scenario())
    assert [r["text"] for r in received] == ["birinci", "ikinci"]
    assert tg.sent == 2
