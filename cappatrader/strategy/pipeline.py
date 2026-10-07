"""The shared analysis pipeline used by backtest, walk-forward and live trading.

OHLCV -> indicators -> regime -> signal scores -> regime-weighted ensemble.
Every column is causal, so the backtest sees exactly what live trading would.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cappatrader.config import COMPONENTS, Config
from cappatrader.features import indicators as ind
from cappatrader.regime.detector import classify
from cappatrader.signals import technical as tech
from cappatrader.signals.ensemble import combine
from cappatrader.signals.ml import ml_score
from cappatrader.signals.statarb import statarb_scores


def compute_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    s, r = cfg.signals, cfg.regime
    c, h, l = df["close"], df["high"], df["low"]
    parts = [
        df,
        pd.DataFrame({"ema_fast": ind.ema(c, s.ema_fast), "ema_slow": ind.ema(c, s.ema_slow),
                      "rsi": ind.rsi(c, s.rsi_period), "atr": ind.atr(h, l, c, r.atr_period)}),
        ind.macd(c, s.macd_fast, s.macd_slow, s.macd_signal),
        ind.bollinger(c, s.bb_period, s.bb_std),
        ind.donchian_prev(h, l, s.breakout_period),
        ind.adx(h, l, c, r.adx_period),
    ]
    return pd.concat(parts, axis=1)


def _last_only(index: pd.Index, value: float | None) -> pd.Series:
    out = pd.Series(np.nan, index=index)
    if value is not None and len(index) and np.isfinite(value):
        out.iloc[-1] = float(value)
    return out


def prepare_frame(
    df: pd.DataFrame,
    cfg: Config,
    statarb: pd.Series | None = None,
    pair_z: dict[str, pd.Series] | None = None,
    ml_proba: pd.Series | None = None,
    live: dict[str, Any] | None = None,
) -> pd.DataFrame:
    f = compute_indicators(df, cfg)
    reg = classify(f, cfg.regime)
    nan = pd.Series(np.nan, index=f.index)
    proba = ml_proba.reindex(f.index) if ml_proba is not None else nan
    live = live or {}
    scores = pd.DataFrame({
        "ema_trend": tech.ema_trend_score(f),
        "macd": tech.macd_score(f),
        "breakout": tech.breakout_score(f),
        "rsi_mr": tech.rsi_mr_score(f, cfg.signals),
        "bb_mr": tech.bb_mr_score(f),
        "statarb": statarb.reindex(f.index) if statarb is not None else nan,
        "ml": ml_score(proba, cfg.signals.ml.prob_threshold) if cfg.signals.ml.enabled else nan,
        # Order flow and sentiment exist only live (no historical L2 / news archive).
        "orderflow": _last_only(f.index, live.get("orderflow")),
        "sentiment": _last_only(f.index, live.get("sentiment")),
    }, index=f.index)
    comb = combine(scores, reg["regime"], cfg.ensemble.weights)
    extra = pd.DataFrame({
        "ml_proba": proba,
        "ema_cross_age": ind.bars_since_sign_change(f["ema_fast"] - f["ema_slow"]),
        "macd_cross_age": ind.bars_since_sign_change(f["macd"]),
    }, index=f.index)
    pz = pd.DataFrame({f"pair_z:{k}": v.reindex(f.index) for k, v in (pair_z or {}).items()}, index=f.index)
    return pd.concat([f, reg, scores.add_prefix("score_"), comb, extra, pz], axis=1)


def prepare_universe(
    data: dict[str, pd.DataFrame],
    cfg: Config,
    ml_probas: dict[str, pd.Series] | None = None,
    live: dict[str, dict[str, Any]] | None = None,
) -> dict[str, pd.DataFrame]:
    closes = {s: df["close"] for s, df in data.items()}
    sa, spreads = statarb_scores(closes, cfg.pairs, cfg.signals.statarb)
    frames = {}
    for sym, df in data.items():
        pair_z = {name: sp["z"] for name, sp in spreads.items() if sym in name.split("/")}
        frames[sym] = prepare_frame(
            df, cfg,
            statarb=sa.get(sym),
            pair_z=pair_z,
            ml_proba=(ml_probas or {}).get(sym),
            live=(live or {}).get(sym),
        )
    return frames


SCORE_COLUMNS = [f"score_{c}" for c in COMPONENTS]
