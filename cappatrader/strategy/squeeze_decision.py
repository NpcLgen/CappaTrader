"""Squeeze Momentum + VWAP strategy: entries and momentum early exits at the bar close.

LONG when the close is above VWAP and momentum shows buyer pressure (value above the previous
bar's); SHORT when the close is below VWAP and momentum turns from buyers to sellers. The stop
(atr_stop_mult x ATR of the signal bar) and the take-profit (take_profit_r x stop distance) are
placed by the sizing layer and triggered intrabar; this engine only adds the early exit when
momentum turns against the trade before the target is reached.
"""

from __future__ import annotations

import copy
import math
import numbers

import numpy as np
import pandas as pd

from cappatrader import fmt
from cappatrader.config import Config
from cappatrader.models import LONG, SHORT, Decision, Position
from cappatrader.regime.detector import UNKNOWN, VOL_EXTREME, VOL_TR, regime_name
from cappatrader.strategy.decision import FrameView

STYLE = "SQZ"


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _flag(x) -> bool:
    """True only for a real True (NaN / None / pd.NA from a reindexed bool column count as False)."""
    try:
        return bool(x) and not pd.isna(x)
    except (TypeError, ValueError):
        return False


def _num(x: float) -> str:
    """Signed value with ~4 significant digits and never an exponent: momentum is in price units,
    so it spans 1e4+ (BTC) down to 1e-6 (low-priced coins)."""
    if not math.isfinite(x):
        return "-"
    if x == 0:
        return "0"
    s = f"{x:+,.{min(max(3 - math.floor(math.log10(abs(x))), 0), 12)}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def _held(flags: np.ndarray, i: int, n: int) -> bool:
    return i - n + 1 >= 0 and all(_flag(flags[k]) for k in range(i - n + 1, i + 1))


class SqueezeVwapEngine:
    """Drop-in for DecisionEngine when strategy.name == "squeeze_vwap"."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.sq = cfg.strategy.squeeze

    @property
    def n(self) -> int:
        return max(int(self.sq.confirm_bars), 1)

    def long_signal(self, v: FrameView, i: int) -> bool:
        """Momentum rising on each of the last N bars; "turn" also needs bar i-N not rising
        (the N-th bar of a new up-leg, Pine "ardışık 3 yeşil" for N=3)."""
        up = v["sqz_mom_up"]
        if not _held(up, i, self.n):
            return False
        return self.sq.long_trigger == "state" or (i >= self.n and not _flag(up[i - self.n]))

    def short_signal(self, v: FrameView, i: int) -> bool:
        """Momentum falling on each of the last N bars; "turn" also needs bar i-N rising
        (buyers -> sellers, then N falling bars)."""
        if not _held(v["sqz_mom_down"], i, self.n):
            return False
        return self.sq.short_trigger == "state" or (i >= self.n and _flag(v["sqz_mom_up"][i - self.n]))

    def evaluate(self, symbol: str, v: FrameView, i: int, position: Position | None,
                 bars_since_exit: int | None = None) -> Decision:
        """Same contract as DecisionEngine.evaluate. bars_since_exit is accepted for parity only:
        the strategy has no re-entry cooldown, a fresh momentum signal is required instead."""
        close, atr = _f(v["close"][i]), _f(v["atr"][i])
        val = _f(v["sqz_val"][i])
        prev = _f(v["sqz_val"][i - 1]) if i > 0 else float("nan")
        prev = prev if math.isfinite(prev) else 0.0  # Pine nz(val[1])
        up, down = _flag(v["sqz_mom_up"][i]), _flag(v["sqz_mom_down"][i])
        direction = 1.0 if up else -1.0 if down else 0.0
        composite = (direction * min(1.0, abs(val - prev) / atr)
                     if math.isfinite(val) and math.isfinite(atr) and atr > 0 else float("nan"))
        prob = _f(v["hmm_prob"][i]) if "hmm_prob" in v.cols else float("nan")
        d = Decision(
            symbol=symbol, timestamp=v.index[i], action="HOLD", composite=composite,
            confidence=100.0 * prob if math.isfinite(prob) else 50.0,
            regime=v["regime"][i], vol_state=v["vol_state"][i], close=close, atr=atr, style=STYLE,
        )

        if position is not None:
            if position.side == LONG and down:
                d.action = "EXIT"
                d.exit_reason = (f"Momentum satıcıya döndü — hedefe ulaşmadan erken çıkış "
                                 f"(momentum {_num(val)} < önceki {_num(prev)})")
            elif position.side == SHORT and up:
                d.action = "EXIT"
                d.exit_reason = (f"Momentum alıcıya döndü — hedefe ulaşmadan erken çıkış "
                                 f"(momentum {_num(val)} > önceki {_num(prev)})")
            if d.action == "EXIT":
                # the turn that closes the trade can be a valid opposite entry on the same bar
                # (LONG closes on buyer->seller below VWAP = the SHORT rule): reverse after the close
                entry = self._entry_action(v, i, d)
                if entry is not None and (entry == "ENTER_SHORT") == (position.side == LONG):
                    rev = copy.copy(d)
                    rev.action, rev.exit_reason, rev.reverse = entry, "", None
                    d.reverse = rev
            return d

        entry = self._entry_action(v, i, d)
        if entry is not None:
            d.action = entry
        return d

    def _entry_action(self, v: FrameView, i: int, d: Decision) -> str | None:
        """ENTER_LONG / ENTER_SHORT when bar i passes every entry gate, else None."""
        close, atr, val = d.close, d.atr, _f(v["sqz_val"][i])
        if not (math.isfinite(close) and math.isfinite(atr) and atr > 0 and math.isfinite(val)):
            return None
        if d.regime == UNKNOWN or d.regime not in self.sq.tradeable_regimes or d.vol_state == VOL_EXTREME:
            return None
        vw = _f(v["vwap"][i])
        if not math.isfinite(vw):
            return None
        if close > vw and self.long_signal(v, i):
            return "ENTER_LONG"
        if self.cfg.account.allow_short and close < vw and self.short_signal(v, i):
            return "ENTER_SHORT"
        return None


def _ok(x) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, (bool, np.bool_)) and math.isfinite(x)


def _mark(sign: float, side: int) -> str:
    s = sign * side
    return "▲ " if s > 0 else "▼ " if s < 0 else ""


def _regime_line(row: pd.Series) -> str:
    regime = row.get("regime")
    parts = [f"Rejim: {regime_name(regime)}"]
    prob = row.get("hmm_prob")
    if _ok(prob):
        parts.append(f"HMM olasılığı %{prob * 100:.0f}")
    elif _ok(row.get("adx")):
        parts.append(f"ADX {row['adx']:.1f}")
    vol = row.get("vol_state")
    parts.append(f"volatilite {VOL_TR.get(vol, vol)}")
    gv, lr = row.get("garch_vol"), row.get("garch_longrun")
    if _ok(gv) and _ok(lr) and lr > 0:
        ratio = row.get("garch_ratio")
        ratio = ratio if _ok(ratio) else gv / lr
        garch = f"GARCH sonraki bar σ {fmt.pct(gv)} (uzun dönem {fmt.pct(lr)}, {ratio:.2f}×)"
        mult = row.get("garch_size_mult")
        parts.append(garch + (f" → boyut çarpanı ×{mult:.2f}" if _ok(mult) else ""))
    elif _ok(row.get("atr_pct")):
        parts.append(f"ATR {fmt.pct(row['atr_pct'])}")
    return " · ".join(parts)


def explain_squeeze(row: pd.Series, side: int, cfg: Config) -> list[str]:
    """Turkish reason lines for a squeeze_vwap entry: ▲ supports the trade direction, ▼ argues against it."""
    sq = cfg.strategy.squeeze
    lines = [_regime_line(row)]

    close, vw = row["close"], row.get("vwap")
    anchor = {"D": "günlük", "W": "haftalık"}.get(sq.vwap_anchor, f"{sq.vwap_rolling_bars} barlık")
    if _ok(vw) and vw > 0:
        rel = "üstünde" if close > vw else "altında" if close < vw else "üzerinde"
        lines.append(f"{_mark(np.sign(close - vw), side)}Fiyat {anchor} VWAP'ın {rel}: kapanış "
                     f"{fmt.price(close)} · VWAP {fmt.price(vw)} ({fmt.pct(close / vw - 1, signed=True)})")
    else:
        lines.append(f"{anchor.capitalize()} VWAP henüz hesaplanamadı (yetersiz veri)")

    val = row.get("sqz_val")
    up, down = _flag(row.get("sqz_mom_up")), _flag(row.get("sqz_mom_down"))
    if _ok(val):
        sign = 1 if up else -1 if down else 0
        streak = row.get("sqz_mom_streak")
        bars = f"{abs(int(streak))} bardır " if _ok(streak) and streak != 0 else ""
        trend = (f"{bars}yükseliyor — alıcı baskısı" if up else
                 f"{bars}düşüyor — satıcı baskısı" if down else "önceki barla aynı")
        zero = "sıfırın üstünde" if val > 0 else "sıfırın altında" if val < 0 else "sıfırda"
        text = f"Squeeze momentum {_num(val)} ({zero}), {trend}"
        if _flag(row.get("sqz_up3")) or _flag(row.get("sqz_down3")):
            text += " · ardışık 3. bar"
        lines.append(f"{_mark(sign, side)}{text}")

    if _flag(row.get("sqz_up_trend")):
        lines.append(f"{_mark(1, side)}Squeeze yukarı çözüldü: sıkışma bitti, önceki mum yeşil")
    elif _flag(row.get("sqz_down_trend")):
        lines.append(f"{_mark(-1, side)}Squeeze aşağı çözüldü: sıkışma bitti, önceki mum kırmızı")
    elif _flag(row.get("sqz_released")):
        lines.append("Squeeze çözüldü (önceki mum nötr, yön belirsiz)")
    elif _flag(row.get("sqz_on")):
        lines.append("Squeeze aktif: Bollinger bantları Keltner kanalı içinde (sıkışma, kırılım bekleniyor)")
    elif "sqz_on" in row.index:
        lines.append("Squeeze yok: Bollinger bantları Keltner kanalının dışında")

    if _flag(row.get("sqz_bull_pivot")):
        lines.append(f"{_mark(1, side)}Boğa pivotu: momentum alt eşiğin ({_num(_f(row.get('sqz_lower')))}) "
                     f"altında yukarı döndü")
    if _flag(row.get("sqz_bear_pivot")):
        lines.append(f"{_mark(-1, side)}Ayı pivotu: momentum üst eşiğin ({_num(_f(row.get('sqz_upper')))}) "
                     f"üstünde aşağı döndü")

    atr = _f(row.get("atr"))
    rule = f"Stop {sq.atr_stop_mult:g}×ATR, hedef {sq.take_profit_r:g}R"
    if math.isfinite(atr) and atr > 0:
        dist = sq.atr_stop_mult * atr
        rule += (f" (sinyal kapanışına göre ≈ stop {fmt.price(close - side * dist)}, "
                 f"hedef {fmt.price(close + side * sq.take_profit_r * dist)})")
    turn = "satıcıya" if side == LONG else "alıcıya"
    lines.append(f"{rule} · hedeften önce momentum {turn} dönerse erken çıkış")
    return lines
