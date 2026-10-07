"""Binance combined-stream WebSocket client with automatic reconnect.

Streams used per symbol: <s>@kline_<tf> (bars), <s>@aggTrade (tick data),
<s>@depth20@100ms (Level-2 top-20 snapshots every 100 ms).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Awaitable, Callable

import aiohttp

log = logging.getLogger(__name__)

Handler = Callable[[str, dict], Awaitable[None]]
StateHandler = Callable[[bool], Awaitable[None]]


def market_streams(symbols: list[str], timeframe: str) -> list[str]:
    out = []
    for s in symbols:
        s = s.lower()
        out += [f"{s}@kline_{timeframe}", f"{s}@aggTrade", f"{s}@depth20@100ms"]
    return out


class BinanceStream:
    def __init__(self, ws_base: str, streams: list[str], on_message: Handler,
                 on_state: StateHandler | None = None, session: aiohttp.ClientSession | None = None,
                 receive_timeout: float = 60.0):
        self.url = f"{ws_base}/stream?streams={'/'.join(streams)}"
        self.n_streams = len(streams)
        self.on_message = on_message
        self.on_state = on_state
        self._session = session
        self.receive_timeout = receive_timeout
        self.connected = False
        self.reconnects = 0

    async def _set_state(self, connected: bool) -> None:
        if connected != self.connected:
            self.connected = connected
            if self.on_state:
                await self.on_state(connected)

    async def run(self, stop: asyncio.Event) -> None:
        own = self._session is None
        session = self._session or aiohttp.ClientSession(trust_env=True)
        backoff = 1.0
        try:
            while not stop.is_set():
                try:
                    timeout = aiohttp.ClientWSTimeout(ws_receive=self.receive_timeout, ws_close=10.0)
                    async with session.ws_connect(self.url, autoping=True, heartbeat=None,
                                                  timeout=timeout, max_msg_size=0) as ws:
                        log.info("WebSocket bağlandı (%d akış)", self.n_streams)
                        await self._set_state(True)
                        backoff = 1.0
                        while not stop.is_set():
                            msg = await ws.receive()
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                payload = json.loads(msg.data)
                                stream, data = payload.get("stream", ""), payload.get("data", payload)
                                try:
                                    await self.on_message(stream, data)
                                except Exception:  # a bad message must not kill the feed
                                    log.exception("Akış mesajı işlenemedi: %s", stream)
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING,
                                              aiohttp.WSMsgType.ERROR):
                                break
                except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
                    log.warning("WebSocket hatası: %s", type(exc).__name__)
                await self._set_state(False)
                if stop.is_set():
                    break
                self.reconnects += 1
                log.info("WebSocket %.0f sn sonra yeniden bağlanacak", backoff)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, 60.0)
        finally:
            if own:
                await session.close()
