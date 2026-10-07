"""Alternative data: funding rates, Crypto Fear & Greed index, news headline sentiment.

All sources are free and keyless. News sentiment is a transparent finance
lexicon over RSS headlines: crude, but explainable in every Telegram message.
On-chain metrics need paid providers (Glassnode, CryptoQuant) and are not wired in.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime

import aiohttp
import numpy as np

from cappatrader.data.binance_rest import BinanceREST

log = logging.getLogger(__name__)

FNG_URL = "https://api.alternative.me/fng/?limit=1"

SYMBOL_KEYWORDS = {
    "BTCUSDT": ("bitcoin", "btc"),
    "ETHUSDT": ("ethereum", "ether", "eth"),
    "SOLUSDT": ("solana", "sol"),
    "BNBUSDT": ("bnb", "binance coin"),
    "XRPUSDT": ("xrp", "ripple"),
    "PAXGUSDT": ("gold", "paxg", "xau", "tokenized gold"),
    "DOGEUSDT": ("dogecoin", "doge"),
    "ADAUSDT": ("cardano", "ada"),
}
MARKET_KEYWORDS = ("crypto", "cryptocurrency", "digital asset", "altcoin", "stablecoin")

POSITIVE = (
    "surge", "surges", "soar", "soars", "rally", "rallies", "jump", "jumps", "gain", "gains", "bullish",
    "record high", "all-time high", "approval", "approved", "approves", "inflow", "inflows", "adoption",
    "partnership", "upgrade", "breakout", "rebound", "rebounds", "recovers", "recovery", "accumulate",
    "accumulation", "launches", "outperform", "buyback", "rate cut",
)
NEGATIVE = (
    "crash", "crashes", "plunge", "plunges", "tumble", "tumbles", "drop", "drops", "fall", "falls", "bearish",
    "hack", "hacked", "exploit", "ban", "bans", "lawsuit", "sues", "sued", "fraud", "outflow", "outflows",
    "liquidation", "liquidations", "bankrupt", "bankruptcy", "sell-off", "selloff", "dump", "delist",
    "investigation", "probe", "fine", "fined", "collapse", "slump", "fear", "fears", "warning", "rate hike",
)


def _count(text: str, words: tuple[str, ...]) -> int:
    return sum(1 for w in words if re.search(rf"\b{re.escape(w)}\b", text))


def score_headline(title: str) -> float | None:
    t = title.lower()
    pos, neg = _count(t, POSITIVE), _count(t, NEGATIVE)
    return (pos - neg) / (pos + neg) if pos + neg else None


def parse_rss(xml_text: str) -> list[tuple[str, float]]:
    """(title, unix time) for every <item> of an RSS 2.0 feed."""
    out = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        pub = item.findtext("pubDate")
        try:
            ts = parsedate_to_datetime(pub).timestamp() if pub else time.time()
        except (TypeError, ValueError):
            ts = time.time()
        if title:
            out.append((title, ts))
    return out


def news_sentiment(headlines: list[tuple[str, float]], symbol: str, now: float,
                   half_life_h: float = 6.0, max_age_h: float = 24.0) -> tuple[float | None, int]:
    keys = SYMBOL_KEYWORDS.get(symbol, (symbol.lower().replace("usdt", ""),))
    total_w, total, n = 0.0, 0.0, 0
    for title, ts in headlines:
        age_h = (now - ts) / 3600.0
        if age_h < 0 or age_h > max_age_h:
            continue
        t = title.lower()
        specific = _count(t, keys) > 0
        # General market news counts at half weight for crypto, not for gold.
        general = not specific and symbol != "PAXGUSDT" and _count(t, MARKET_KEYWORDS) > 0
        if not (specific or general):
            continue
        s = score_headline(title)
        if s is None:
            continue
        w = 0.5 ** (age_h / half_life_h) * (1.0 if specific else 0.5)
        total += w * s
        total_w += w
        n += 1
    return (total / total_w if total_w > 0 else None), n


def funding_score(rate: float) -> float:
    # Neutral around the default +0.01%/8h; crowded longs (high funding) -> contrarian bearish.
    return float(-np.clip((rate - 0.0001) / 0.0004, -1.0, 1.0))


def fng_score(value: float) -> float:
    # Extreme fear -> contrarian bullish, extreme greed -> bearish.
    return float(np.clip((50.0 - value) / 40.0, -1.0, 1.0))


@dataclass
class AltDataHub:
    rest: BinanceREST
    feeds: list[str]
    symbols: list[str]
    funding: dict[str, float] = field(default_factory=dict)
    fng: tuple[int, str] | None = None
    headlines: list[tuple[str, float]] = field(default_factory=list)
    last_refresh: float = 0.0

    async def refresh(self, session: aiohttp.ClientSession) -> None:
        async def get_text(url: str) -> str | None:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
                    return await r.text() if r.status == 200 else None
            except (aiohttp.ClientError, asyncio.TimeoutError):
                return None

        for s in self.symbols:
            rate = await self.rest.funding_rate(s)
            if rate is not None:
                self.funding[s] = rate
        try:
            async with session.get(FNG_URL, timeout=aiohttp.ClientTimeout(total=10)) as r:
                if r.status == 200:
                    d = (await r.json(content_type=None))["data"][0]
                    self.fng = (int(d["value"]), d.get("value_classification", ""))
        except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, IndexError, ValueError):
            pass
        heads: list[tuple[str, float]] = []
        for url in self.feeds:
            text = await get_text(url)
            if text:
                heads.extend(parse_rss(text))
        if heads:
            self.headlines = heads
        self.last_refresh = time.time()
        log.info("Alternatif veri güncellendi: funding %d sembol, F&G %s, %d başlık",
                 len(self.funding), self.fng, len(self.headlines))

    def sentiment(self, symbol: str, now: float | None = None) -> tuple[float | None, str]:
        now = time.time() if now is None else now
        parts, scores = [], []
        rate = self.funding.get(symbol)
        if rate is not None:
            scores.append(funding_score(rate))
            parts.append(f"fonlama oranı %{rate * 100:.4f}/8s")
        if self.fng is not None and symbol != "PAXGUSDT":
            scores.append(fng_score(self.fng[0]))
            parts.append(f"Korku&Açgözlülük {self.fng[0]} ({self.fng[1]})")
        news, n = news_sentiment(self.headlines, symbol, now)
        if news is not None:
            scores.append(news)
            parts.append(f"haber duyarlılığı {news:+.2f} ({n} başlık)")
        if not scores:
            return None, ""
        score = float(np.mean(scores))
        return (score if math.isfinite(score) else None), "Duyarlılık: " + " · ".join(parts)
