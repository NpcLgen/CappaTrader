"""Turns a prepared frame row plus position state into ENTER / EXIT / HOLD."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from cappatrader.config import Config
from cappatrader.models import LONG, SHORT, Decision, Position
from cappatrader.regime.detector import UNKNOWN, VOL_EXTREME

_MR = ("contrib_rsi_mr", "contrib_bb_mr")
_MOM = ("contrib_ema_trend", "contrib_macd", "contrib_breakout")


class FrameView:
    """Column arrays of a prepared frame for fast per-bar access."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.index = frame.index
        self.cols = {c: frame[c].to_numpy() for c in frame.columns}

    def __getitem__(self, col: str) -> np.ndarray:
        return self.cols[col]

    def __len__(self) -> int:
        return len(self.index)

    def row(self, i: int) -> pd.Series:
        return self.frame.iloc[i]


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


class DecisionEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.e = cfg.ensemble

    def style(self, v: FrameView, i: int, side: int) -> str:
        mr = sum(np.nan_to_num(v[c][i]) for c in _MR)
        mom = sum(np.nan_to_num(v[c][i]) for c in _MOM)
        return "MR" if side * mr > side * mom else "TREND"

    def evaluate(self, symbol: str, v: FrameView, i: int, position: Position | None,
                 bars_since_exit: int | None = None) -> Decision:
        comp = _f(v["composite"][i])
        regime = v["regime"][i]
        vol = v["vol_state"][i]
        d = Decision(
            symbol=symbol, timestamp=v.index[i], action="HOLD", composite=comp,
            confidence=50.0 * (1 + comp) if math.isfinite(comp) else float("nan"),
            regime=regime, vol_state=vol, close=_f(v["close"][i]), atr=_f(v["atr"][i]),
        )
        if not math.isfinite(comp) or not math.isfinite(d.atr) or d.atr <= 0:
            return d

        if position is not None:
            s = position.side * comp
            if position.style == "MR":
                if s <= self.e.mr_exit_score:
                    d.action, d.exit_reason = "EXIT", f"Ortalamaya dönüş tamamlandı / sinyal nötrlendi (skor {comp:+.2f})"
                elif position.bars_held >= self.e.mr_max_hold_bars:
                    d.action, d.exit_reason = "EXIT", f"Süre aşımı ({position.bars_held} bar, ortalamaya dönüş gerçekleşmedi)"
            elif s <= self.e.trend_exit_score:
                d.action, d.exit_reason = "EXIT", f"Sinyal dönüşü: ensemble skoru {comp:+.2f}"
            return d

        if regime == UNKNOWN or vol == VOL_EXTREME:
            return d
        if bars_since_exit is not None and bars_since_exit < self.e.reentry_cooldown_bars:
            return d
        th = self.e.entry_confidence
        if 50.0 * (1 + comp) >= th:
            d.action, d.confidence = "ENTER_LONG", 50.0 * (1 + comp)
            d.style = self.style(v, i, LONG)
        elif self.cfg.account.allow_short and 50.0 * (1 - comp) >= th:
            d.action, d.confidence = "ENTER_SHORT", 50.0 * (1 - comp)
            d.style = self.style(v, i, SHORT)
        return d
