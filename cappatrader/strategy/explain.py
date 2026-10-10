"""Human-readable (Turkish) explanation of why the ensemble wants to trade."""

from __future__ import annotations

import math
import numbers

import pandas as pd

from cappatrader import fmt
from cappatrader.config import Config
from cappatrader.regime.detector import REGIME_TR, VOL_TR


def _ok(x) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, bool) and math.isfinite(x)


def regime_line(row: pd.Series) -> str:
    """Regime label with its source (HMM probability or ADX) and the volatility view (GARCH or ATR)."""
    label = REGIME_TR.get(row["regime"], row["regime"])
    prob = row.get("hmm_prob")
    src = f"HMM olasılık %{prob * 100:.0f}" if _ok(prob) else f"ADX {row['adx']:.1f}"
    ratio = row.get("garch_ratio")
    if _ok(ratio):
        vol = (f"GARCH tahmini vol %{row['garch_vol'] * 100:.2f} (uzun dönem %{row['garch_longrun'] * 100:.2f}, "
               f"×{ratio:.2f})")
    else:
        vol_pct = row.get("vol_pct")
        vol = f"ATR {row['atr_pct'] * 100:.2f}%" + (f", {vol_pct * 100:.0f}. persentil" if _ok(vol_pct) else "")
    return f"Rejim: {label} · {src} · volatilite {VOL_TR.get(row['vol_state'], row['vol_state'])} ({vol})"


def _component_text(name: str, row: pd.Series, cfg: Config) -> str | None:
    s = cfg.signals
    if name == "ema_trend":
        d = (row["ema_fast"] - row["ema_slow"]) / row["atr"]
        rel = ">" if d > 0 else "<"
        text = f"EMA{s.ema_fast} {rel} EMA{s.ema_slow} ({d:+.2f} ATR)"
        age = row.get("ema_cross_age")
        if _ok(age) and age <= 5:
            text += f", {'yukarı' if d > 0 else 'aşağı'} kesişim {int(age)} bar önce"
        return text
    if name == "macd":
        side = "üstünde" if row["macd"] > 0 else "altında"
        text = f"MACD sıfır çizgisinin {side} ({row['macd']:+.4g}, histogram {row['macd_hist']:+.4g})"
        age = row.get("macd_cross_age")
        if _ok(age) and age <= 5:
            text += f", sıfırı {'yukarı' if row['macd'] > 0 else 'aşağı'} kesti {int(age)} bar önce"
        return text
    if name == "breakout":
        n = s.breakout_period
        if row["close"] > row["dc_high"]:
            return f"{n} barlık kanal yukarı kırıldı (kapanış {fmt.price(row['close'])} > zirve {fmt.price(row['dc_high'])})"
        if row["close"] < row["dc_low"]:
            return f"{n} barlık kanal aşağı kırıldı (kapanış {fmt.price(row['close'])} < dip {fmt.price(row['dc_low'])})"
        pos = (row["close"] - row["dc_low"]) / max(row["dc_high"] - row["dc_low"], 1e-12)
        return f"Fiyat {n} barlık kanalın %{pos * 100:.0f} seviyesinde"
    if name == "rsi_mr":
        r = row["rsi"]
        if r <= s.rsi_oversold:
            state = "aşırı satım bölgesi, tepki alımı beklentisi"
        elif r >= s.rsi_overbought:
            state = "aşırı alım bölgesi, geri çekilme beklentisi"
        else:
            state = "ortalamanın altında" if r < 50 else "ortalamanın üstünde"
        return f"RSI {r:.1f} — {state}"
    if name == "bb_mr":
        b = row["bb_pctb"]
        where = "alt banda yakın" if b < 0.2 else "üst banda yakın" if b > 0.8 else "bant ortasında"
        return f"Bollinger %b {b:.2f} — {where} (bant içi sekme / ortalamaya dönüş)"
    if name == "statarb":
        parts = []
        for col in row.index:
            if col.startswith("pair_z:") and _ok(row[col]):
                a, b = col[len("pair_z:"):].split("/")
                z = row[col]
                cheap = fmt.base_asset(a) if z < 0 else fmt.base_asset(b)
                parts.append(f"{fmt.base_asset(a)}/{fmt.base_asset(b)} makası z={z:+.2f} → {cheap} görece ucuz")
        return "Pairs trading: " + "; ".join(parts) if parts else None
    if name == "ml":
        p = row.get("ml_proba")
        if not _ok(p):
            return None
        return f"ML (Random Forest) {s.ml.horizon} bar sonrası yükseliş olasılığı %{p * 100:.0f}"
    return None


def explain(row: pd.Series, side: int, cfg: Config, live_notes: dict[str, str] | None = None,
            min_contrib: float = 0.02) -> list[str]:
    """Regime line, ensemble line, then each contributing signal ranked by weight.
    ▲ supports the trade direction, ▼ argues against it."""
    lines = [regime_line(row)]
    conf = row["conf_long"] if side > 0 else row["conf_short"]
    lines.append(
        f"Ensemble skoru {row['composite']:+.2f} → güven %{conf:.0f} (eşik %{cfg.ensemble.entry_confidence:.0f})"
    )
    live_notes = live_notes or {}
    ranked = []
    for col in row.index:
        if not col.startswith("contrib_"):
            continue
        name = col[len("contrib_"):]
        c = row[col]
        if not _ok(c) or abs(c) < min_contrib:
            continue
        text = live_notes.get(name) or _component_text(name, row, cfg)
        if text:
            ranked.append((abs(c), "▲" if c * side > 0 else "▼", text))
    ranked.sort(key=lambda t: -t[0])
    lines.extend(f"{mark} {text}" for _, mark, text in ranked)
    return lines
