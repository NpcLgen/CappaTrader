"""Number formatting for reports and Telegram messages."""

from __future__ import annotations

import math


def price(p: float) -> str:
    if p is None or not math.isfinite(p):
        return "-"
    if abs(p) >= 1000:
        return f"{p:,.2f}"
    if abs(p) >= 1:
        return f"{p:,.4f}"
    return f"{p:.6f}"


def qty(q: float) -> str:
    if abs(q) >= 100:
        return f"{q:,.2f}"
    if abs(q) >= 1:
        return f"{q:,.4f}"
    return f"{q:.6f}".rstrip("0").rstrip(".")


def usd(x: float, signed: bool = False) -> str:
    return f"{x:+,.2f} USDT" if signed else f"{x:,.2f} USDT"


def pct(x: float, signed: bool = False, digits: int = 2) -> str:
    return f"{x * 100:+.{digits}f}%" if signed else f"{x * 100:.{digits}f}%"


def base_asset(symbol: str, quote: str = "USDT") -> str:
    return symbol[: -len(quote)] if symbol.endswith(quote) else symbol


def pair(symbol: str, quote: str = "USDT") -> str:
    return f"{base_asset(symbol, quote)}/{quote}" if symbol.endswith(quote) else symbol
