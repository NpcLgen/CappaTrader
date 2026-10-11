"""Turkish Telegram message templates (HTML parse mode)."""

from __future__ import annotations

from html import escape

import pandas as pd

from cappatrader import fmt
from cappatrader.models import LONG, Decision, Fill, Position, Trade
from cappatrader.regime.detector import REGIME_TR, regime_name


def _bullets(lines: list[str]) -> str:
    return "\n".join(f"• {escape(line)}" for line in lines)


def _duration(start: str, end: str) -> str:
    sec = max((pd.Timestamp(end) - pd.Timestamp(start)).total_seconds(), 0)
    days, rem = divmod(int(sec), 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    return (f"{days} gün " if days else "") + f"{hours} sa {minutes} dk"


def balance_block(equity: float, cash: float, start: float, open_positions: int) -> str:
    total = equity / start - 1 if start else 0.0
    return (f"💼 <b>Güncel bakiye: {fmt.usd(equity)}</b> ({fmt.pct(total, signed=True)} başlangıç {fmt.usd(start)})\n"
            f"Nakit {fmt.usd(cash)} · açık pozisyon {open_positions}")


def entry_message(pos: Position, fill: Fill, d: Decision, sizing_notes: list[str], equity: float, cash: float,
                  start_balance: float, open_positions: int, mode: str = "KAĞIT HESAP") -> str:
    long = pos.side == LONG
    head = "🟢 <b>ALIŞ (LONG)" if long else "🟠 <b>AÇIĞA SATIŞ (SHORT)"
    asset = fmt.base_asset(pos.symbol)
    stop_pct = (pos.stop_price / pos.entry_price - 1)
    style = {"MR": "Ortalamaya dönüş", "SQZ": "Squeeze momentum + VWAP"}.get(pos.style, "Trend takibi / momentum")
    risk_amt = pos.risk_per_unit * pos.qty
    lines = [
        f"{head} — {escape(fmt.pair(pos.symbol))}</b> · {mode}",
        f"Fiyat: <code>{fmt.price(fill.price)}</code> (sinyal {fmt.price(fill.ref_price)} · slippage "
        f"{fill.slippage_bps:.1f} bps · gecikme {fill.latency_ms:.0f} ms)",
        f"Miktar: {fmt.qty(pos.qty)} {escape(asset)} ≈ {fmt.usd(fill.notional)} · komisyon {fill.fee:.2f} USDT",
        f"Stop: <code>{fmt.price(pos.stop_price)}</code> ({fmt.pct(stop_pct, signed=True)}) · risk {fmt.usd(risk_amt)} "
        f"(%{risk_amt / equity * 100:.2f} kasa)" + (f" · hedef {fmt.price(pos.take_profit)}" if pos.take_profit else ""),
        (f"Rejim: {escape(regime_name(d.regime))} · HMM olasılığı %{d.confidence:.0f} · strateji: {style}"
         if pos.style == "SQZ" else
         f"Rejim: {escape(REGIME_TR.get(d.regime, d.regime))} · güven %{d.confidence:.0f} · strateji: {style}"),
        "",
        "<b>Neden alındı?</b>" if long else "<b>Neden açığa satıldı?</b>",
        _bullets(pos.reasons),
        "",
        "<b>Pozisyon boyutu</b>",
        _bullets(sizing_notes),
        "",
        balance_block(equity, cash, start_balance, open_positions),
    ]
    return "\n".join(lines)


def exit_message(t: Trade, fill: Fill, equity: float, cash: float, start_balance: float, open_positions: int,
                 all_trades: list[Trade], mode: str = "KAĞIT HESAP") -> str:
    long = t.side == LONG
    head = "🔴 <b>SATIŞ — pozisyon kapandı" if long else "🟢 <b>ALIŞ — short kapandı"
    win = t.pnl > 0
    n = len(all_trades)
    wins = sum(1 for x in all_trades if x.pnl > 0)
    realized = sum(x.pnl for x in all_trades)
    lines = [
        f"{head} — {escape(fmt.pair(t.symbol))}</b> · {mode}",
        f"Giriş <code>{fmt.price(t.entry_price)}</code> → Çıkış <code>{fmt.price(t.exit_price)}</code> "
        f"(slippage {fill.slippage_bps:.1f} bps) · süre {_duration(t.entry_time, t.exit_time)}",
        f"Miktar: {fmt.qty(t.qty)} {escape(fmt.base_asset(t.symbol))} ≈ {fmt.usd(t.qty * t.exit_price)}",
        f"{'✅' if win else '❌'} <b>P&amp;L: {fmt.usd(t.pnl, signed=True)} ({fmt.pct(t.return_pct, signed=True)} · "
        f"{t.r_multiple:+.2f}R)</b> · komisyon {t.fees:.2f} USDT dahil",
        "",
        "<b>Neden satıldı?</b>" if long else "<b>Neden kapatıldı?</b>",
        f"• {escape(t.exit_reason)}",
    ]
    if t.entry_reasons:
        lines += ["", "<b>Giriş gerekçesi (özet)</b>", _bullets(t.entry_reasons[:3])]
    lines += [
        "",
        balance_block(equity, cash, start_balance, open_positions),
        f"Realize P&amp;L toplam {fmt.usd(realized, signed=True)} · {n} işlem · kazanma %{wins / n * 100:.0f}" if n else "",
    ]
    return "\n".join(x for x in lines if x is not None)


def startup_message(symbols: list[str], timeframe: str, equity: float, cash: float, start_balance: float,
                    positions: dict[str, Position], warnings: list[str], mode: str = "KAĞIT HESAP",
                    extra: str = "") -> str:
    note = " (gerçek emir yok)" if mode == "KAĞIT HESAP" else ""
    lines = [
        f"🤖 <b>CappaTrader başlatıldı</b> · {escape(mode)}{note}",
        f"Pariteler: {escape(', '.join(fmt.pair(s) for s in symbols))} · zaman dilimi {escape(timeframe)}",
    ]
    if extra:
        lines.append(escape(extra))
    lines.append(balance_block(equity, cash, start_balance, len(positions)))
    for p in positions.values():
        lines.append(f"• Açık: {escape(fmt.pair(p.symbol))} {fmt.qty(p.qty)} @ {fmt.price(p.entry_price)}, "
                     f"stop {fmt.price(p.stop_price)}")
    if warnings:
        lines += ["", "<b>Uyarılar</b>", _bullets(warnings)]
    return "\n".join(lines)


def daily_summary(day: str, equity: float, cash: float, start_balance: float, day_start_equity: float,
                  positions: dict[str, Position], prices: dict[str, float], day_trades: list[Trade]) -> str:
    day_ret = equity / day_start_equity - 1 if day_start_equity else 0.0
    lines = [f"📊 <b>Günlük özet — {escape(day)}</b>",
             f"Gün içi: {fmt.pct(day_ret, signed=True)} · kapanan işlem {len(day_trades)} · "
             f"realize {fmt.usd(sum(t.pnl for t in day_trades), signed=True)}"]
    for p in positions.values():
        px = prices.get(p.symbol, p.entry_price)
        lines.append(f"• {escape(fmt.pair(p.symbol))}: {fmt.usd(p.unrealized(px), signed=True)} açık P&amp;L, "
                     f"stop {fmt.price(p.stop_price)}")
    lines.append(balance_block(equity, cash, start_balance, len(positions)))
    return "\n".join(lines)


def alert_message(title: str, text: str) -> str:
    return f"⚠️ <b>{escape(title)}</b>\n{escape(text)}"
