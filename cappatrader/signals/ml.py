"""Machine-learning direction model (Random Forest) trained strictly on the past.

Features use fixed indicator periods so the model does not change when the
strategy grid is optimised. Probabilities are produced walk-forward: the model
that scores bar t was trained only on labels whose horizon ended before t.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from cappatrader.config import MLConfig
from cappatrader.features import indicators as ind

FEATURES = [
    "ret_1", "ret_2", "ret_4", "ret_8", "ret_16", "rv_20", "rsi", "macd_n", "atr_pct",
    "adx", "di_diff", "bb_pctb", "bb_width", "vol_z", "ema_dist", "close_ema", "hour_sin", "hour_cos",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l = df["close"], df["high"], df["low"]
    logc = np.log(c)
    a = ind.atr(h, l, c, 14)
    m = ind.macd(c)
    dx = ind.adx(h, l, c, 14)
    bb = ind.bollinger(c, 20, 2.0)
    e20, e50 = ind.ema(c, 20), ind.ema(c, 50)
    out = pd.DataFrame(index=df.index)
    for k in (1, 2, 4, 8, 16):
        out[f"ret_{k}"] = logc.diff(k)
    out["rv_20"] = logc.diff().rolling(20).std()
    out["rsi"] = ind.rsi(c, 14) / 100.0
    out["macd_n"] = m["macd_hist"] / a
    out["atr_pct"] = a / c
    out["adx"] = dx["adx"] / 100.0
    out["di_diff"] = (dx["plus_di"] - dx["minus_di"]) / 100.0
    out["bb_pctb"] = bb["bb_pctb"]
    out["bb_width"] = bb["bb_width"]
    out["vol_z"] = ind.zscore(np.log1p(df["volume"]), 50)
    out["ema_dist"] = (e20 - e50) / a
    out["close_ema"] = (c - e50) / a
    if isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour + df.index.minute / 60.0
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    else:
        out["hour_sin"] = 0.0
        out["hour_cos"] = 0.0
    return out.replace([np.inf, -np.inf], np.nan)


def build_labels(close: pd.Series, horizon: int) -> pd.Series:
    fwd = np.log(close.shift(-horizon) / close)
    return (fwd > 0).astype(float).where(fwd.notna())


def make_model(cfg: MLConfig) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        random_state=cfg.random_state,
        n_jobs=-1,
    )


def _proba_up(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    classes = list(model.classes_)
    p = model.predict_proba(X)
    return p[:, classes.index(1.0)] if 1.0 in classes else np.zeros(len(X))


@dataclass
class MLStats:
    predictions: int = 0
    accuracy: float = float("nan")
    confident_share: float = float("nan")
    confident_accuracy: float = float("nan")
    fits: int = 0


def walk_forward_proba(df: pd.DataFrame, cfg: MLConfig) -> tuple[pd.Series, MLStats]:
    """Out-of-sample up-probabilities for every bar (NaN until enough history)."""
    X = build_features(df)
    y = build_labels(df["close"], cfg.horizon)
    valid = X.notna().all(axis=1).to_numpy()
    Xv = X.to_numpy()
    yv = y.to_numpy()
    n = len(df)
    proba = np.full(n, np.nan)
    stats = MLStats()
    first_valid = int(np.argmax(valid)) if valid.any() else n
    min_train = max(200, cfg.train_bars // 2)
    start = first_valid + min_train + cfg.horizon
    while start < n:
        train_end = start - cfg.horizon  # labels of rows < train_end only use closes < start
        train_start = max(0, train_end - cfg.train_bars)
        rows = np.arange(train_start, train_end)
        mask = valid[rows] & ~np.isnan(yv[rows])
        stop = min(start + cfg.retrain_every, n)
        if mask.sum() >= min_train and len(np.unique(yv[rows][mask])) == 2:
            model = make_model(cfg)
            model.fit(Xv[rows][mask], yv[rows][mask])
            stats.fits += 1
            pred_rows = np.arange(start, stop)
            pred_rows = pred_rows[valid[pred_rows]]
            if len(pred_rows):
                proba[pred_rows] = _proba_up(model, Xv[pred_rows])
        start = stop
    s = pd.Series(proba, index=df.index, name="ml_proba")
    scored = s.notna() & y.notna()
    if scored.any():
        p, t = s[scored], y[scored]
        stats.predictions = int(scored.sum())
        stats.accuracy = float(((p > 0.5).astype(float) == t).mean())
        conf = (p >= cfg.prob_threshold) | (p <= 1 - cfg.prob_threshold)
        stats.confident_share = float(conf.mean())
        if conf.any():
            stats.confident_accuracy = float(((p[conf] > 0.5).astype(float) == t[conf]).mean())
    return s, stats


def ml_score(proba: pd.Series, threshold: float) -> pd.Series:
    """Map probability to [-1, 1]; zero unless the model clears the confidence threshold."""
    score = 2.0 * (proba - 0.5)
    confident = (proba >= threshold) | (proba <= 1.0 - threshold)
    return score.where(confident, 0.0).where(proba.notna())


class OnlineModel:
    """Live counterpart of walk_forward_proba: refit every `retrain_every` closed bars."""

    def __init__(self, cfg: MLConfig):
        self.cfg = cfg
        self.model: RandomForestClassifier | None = None
        self.bars_since_fit = 0

    def needs_fit(self) -> bool:
        return self.model is None or self.bars_since_fit >= self.cfg.retrain_every

    def fit(self, df: pd.DataFrame) -> bool:
        X = build_features(df)
        y = build_labels(df["close"], self.cfg.horizon)
        mask = X.notna().all(axis=1) & y.notna()
        Xt, yt = X[mask].tail(self.cfg.train_bars), y[mask].tail(self.cfg.train_bars)
        if len(Xt) < max(200, self.cfg.train_bars // 2) or yt.nunique() < 2:
            return False
        model = make_model(self.cfg)
        model.fit(Xt.to_numpy(), yt.to_numpy())
        self.model = model
        self.bars_since_fit = 0
        return True

    def predict_last(self, df: pd.DataFrame) -> float:
        if self.model is None:
            return float("nan")
        row = build_features(df).iloc[[-1]]
        if row.isna().any(axis=None):
            return float("nan")
        return float(_proba_up(self.model, row.to_numpy())[0])
