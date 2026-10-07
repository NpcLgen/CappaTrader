"""Non-blocking Telegram notifier: trading code enqueues, a worker sends.

Sending never raises into the trading loop; failures are logged with the bot
token masked.
"""

from __future__ import annotations

import asyncio
import logging

import aiohttp

log = logging.getLogger(__name__)

MAX_LEN = 4096


def split_message(text: str, limit: int = MAX_LEN) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts, current = [], ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > limit:
            if current:
                parts.append(current)
            current = line[:limit]
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        parts.append(current)
    return parts


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, enabled: bool = True, min_interval_s: float = 1.1,
                 api_base: str = "https://api.telegram.org", session: aiohttp.ClientSession | None = None):
        self.token = token
        self.chat_id = chat_id
        self.enabled = enabled and bool(token and chat_id)
        self.min_interval_s = min_interval_s
        self.api_base = api_base
        self._session = session
        self._own_session = session is None
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=500)
        self._worker: asyncio.Task | None = None
        self.sent = 0
        self.failed = 0

    def _mask(self, text: str) -> str:
        return text.replace(self.token, "***") if self.token else text

    async def _sess(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15), trust_env=True)
            self._own_session = True
        return self._session

    async def send_now(self, text: str, retries: int = 3) -> bool:
        if not self.enabled:
            log.info("[telegram kapalı] %s", text.split("\n")[0])
            return False
        ok = True
        for chunk in split_message(text):
            ok = await self._send_chunk(chunk, retries) and ok
        return ok

    async def _send_chunk(self, text: str, retries: int) -> bool:
        url = f"{self.api_base}/bot{self.token}/sendMessage"
        body = {"chat_id": self.chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
        for attempt in range(retries + 1):
            try:
                s = await self._sess()
                async with s.post(url, json=body) as resp:
                    if resp.status == 200:
                        self.sent += 1
                        return True
                    data = await resp.json(content_type=None)
                    if resp.status == 429:
                        wait = float(data.get("parameters", {}).get("retry_after", 5))
                        await asyncio.sleep(wait)
                        continue
                    if resp.status == 400 and "parse" in str(data.get("description", "")).lower():
                        body.pop("parse_mode", None)  # malformed HTML: retry as plain text
                        continue
                    log.error("Telegram HTTP %s: %s", resp.status, self._mask(str(data.get("description", ""))))
                    if resp.status < 500:
                        break
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                log.warning("Telegram gönderilemedi (%s): %s", type(exc).__name__, self._mask(str(exc)))
            await asyncio.sleep(2**attempt)
        self.failed += 1
        return False

    def notify(self, text: str) -> None:
        """Enqueue without blocking; drops the oldest message if the queue is full."""
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        self.queue.put_nowait(text)

    async def _run(self) -> None:
        while True:
            text = await self.queue.get()
            try:
                await self.send_now(text)
            finally:
                self.queue.task_done()
            await asyncio.sleep(self.min_interval_s)

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="telegram")

    async def stop(self, flush_timeout: float = 15.0) -> None:
        if self._worker is not None:
            try:
                await asyncio.wait_for(self.queue.join(), timeout=flush_timeout)
            except asyncio.TimeoutError:
                log.warning("Telegram kuyruğu boşaltılamadı (%d mesaj)", self.queue.qsize())
            self._worker.cancel()
            self._worker = None
        if self._own_session and self._session is not None and not self._session.closed:
            await self._session.close()
