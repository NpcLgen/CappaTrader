"""The shared analysis pipeline used by backtest, walk-forward and live trading.

OHLCV -> indicators -> regime (HMM or ADX, GARCH or ATR volatility) -> squeeze/VWAP
layer + ensemble signal scores. Every column is causal, so the backtest sees exactly
what live trading would.

Regime layers (walk-forward HMM / GARCH) are the expensive part; research commands
compute them once per symbol with regime_layers() and pass them in, live trading
passes single-bar layers from its online models.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd

from cappatrader.config import COMPONENTS, Config
from cappatrader.features import indicators as ind
from cappatrader.regime.hybrid import classify_regime
from cappatrader.signals import technical as tech
from cappatrader.signals.ensemble import combine
from cappatrader.signals.ml import ml_score
from cappatrader.signals.squeeze import squeeze_layer
from cappatrader.signals.statarb import statarb_scores

Layers = dict[str, pd.DataFrame | None]  # {"hmm": walk_forward_hmm frame, "garch": walk_forward_garch frame}


_LAYER_CACHE: OrderedDict[tuple, Layers] = OrderedDict()
_LAYER_CACHE_MAX = 64


def _layer_key(df: pd.DataFrame, cfg: Config) -> tuple:
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(df.index.asi8 if isinstance(df.index, pd.DatetimeIndex) else np.arange(len(df))))
    h.update(np.ascontiguousarray(df["close"].to_numpy(dtype=float)))
    r = cfg.regime
    tz = str(getattr(df.index, "tz", None))  # a tz-naive copy must not reuse tz-aware layers (reindex mismatch)
    return (h.hexdigest(), tz, str(df.index.dtype), r.method, r.vol_method, repr(r.hmm), repr(r.garch))


def compute_layers(df: pd.DataFrame, cfg: Config) -> Layers:
    """Walk-forward HMM and GARCH outputs for one symbol (only those the config uses).

    Both models only read closes, so results are memoised on the close series and the
    model settings; callers must treat the returned frames as read-only."""
    key = _layer_key(df, cfg)
    if key in _LAYER_CACHE:
        _LAYER_CACHE.move_to_end(key)
        return _LAYER_CACHE[key]
    layers: Layers = {"hmm": None, "garch": None}
    if cfg.regime.method == "hmm":
        from cappatrader.regime.hmm import walk_forward_hmm

        layers["hmm"] = walk_forward_hmm(df, cfg.regime.hmm)
    if cfg.regime.vol_method == "garch":
        from cappatrader.regime.garch import walk_forward_garch

        layers["garch"] = walk_forward_garch(df, cfg.regime.garch)
    _LAYER_CACHE[key] = layers
    if len(_LAYER_CACHE) > _LAYER_CACHE_MAX:
        _LAYER_CACHE.popitem(last=False)
    return layers


def regime_layers(data: dict[str, pd.DataFrame], cfg: Config) -> dict[str, Layers]:
    return {sym: compute_layers(df, cfg) for sym, df in data.items()}


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
    layers: Layers | None = None,
) -> pd.DataFrame:
    """layers=None computes the walk-forward HMM/GARCH layers here; pass {} to skip them
    (the regime then falls back to ADX/ATR wherever HMM/GARCH values are missing)."""
    f = compute_indicators(df, cfg)
    if layers is None:
        layers = compute_layers(df, cfg)
    reg = classify_regime(f, cfg.regime, layers.get("hmm"), layers.get("garch"))
    sqz = squeeze_layer(df, cfg.strategy.squeeze)
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
    reg = reg.drop(columns=[c for c in reg.columns if c in f.columns])
    sqz = sqz.drop(columns=[c for c in sqz.columns if c in f.columns or c in reg.columns])
    return pd.concat([f, reg, sqz, scores.add_prefix("score_"), comb, extra, pz], axis=1)


def prepare_universe(
    data: dict[str, pd.DataFrame],
    cfg: Config,
    ml_probas: dict[str, pd.Series] | None = None,
    live: dict[str, dict[str, Any]] | None = None,
    layers: dict[str, Layers] | None = None,
) -> dict[str, pd.DataFrame]:
    """layers: per-symbol regime layers; a symbol missing from the dict gets them computed."""
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
            layers=(layers or {}).get(sym),
        )
    return frames


SCORE_COLUMNS = [f"score_{c}" for c in COMPONENTS]
