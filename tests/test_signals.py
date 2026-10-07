import numpy as np
import pandas as pd

from cappatrader.config import COMPONENTS
from cappatrader.regime.detector import RANGE, TREND_DOWN, TREND_UP, UNKNOWN, classify
from cappatrader.signals.ensemble import combine
from cappatrader.signals.ml import ml_score, walk_forward_proba
from cappatrader.signals.statarb import engle_granger, rolling_spread, statarb_scores
from cappatrader.strategy.pipeline import compute_indicators, prepare_frame, prepare_universe


def _trend(n=400, slope=0.004):
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(slope + 0.002 * rng.standard_normal(n)))
    return pd.DataFrame({"open": close, "high": close * 1.003, "low": close * 0.997, "close": close,
                         "volume": 1.0}, index=idx)


def test_regime_detects_trend_up(cfg):
    f = compute_indicators(_trend(), cfg)
    reg = classify(f, cfg.regime)
    assert (reg["regime"].iloc[-100:] == TREND_UP).mean() > 0.8


def test_regime_detects_trend_down(cfg):
    f = compute_indicators(_trend(slope=-0.004), cfg)
    assert (classify(f, cfg.regime)["regime"].iloc[-100:] == TREND_DOWN).mean() > 0.8


def test_regime_detects_range(cfg):
    idx = pd.date_range("2025-01-01", periods=600, freq="h", tz="UTC")
    rng = np.random.default_rng(1)
    close = 100 + np.sin(np.arange(600) / 4) + 0.2 * rng.standard_normal(600)
    df = pd.DataFrame({"open": close, "high": close + 0.3, "low": close - 0.3, "close": close, "volume": 1.0}, index=idx)
    reg = classify(compute_indicators(df, cfg), cfg.regime)
    assert (reg["regime"].iloc[-200:] == RANGE).mean() > 0.6
    assert reg["regime"].iloc[0] == UNKNOWN


def test_scores_are_bounded(cfg, btc):
    f = prepare_frame(btc, cfg)
    for c in COMPONENTS:
        s = f[f"score_{c}"].dropna()
        assert s.between(-1, 1).all(), c
    assert f["conf_long"].dropna().between(0, 100).all()


def test_ensemble_regime_weights():
    idx = pd.RangeIndex(2)
    scores = pd.DataFrame({c: [np.nan, np.nan] for c in COMPONENTS}, index=idx)
    scores["ema_trend"] = [1.0, 1.0]
    scores["rsi_mr"] = [-1.0, -1.0]
    weights = {"TREND": {"ema_trend": 1.0, "rsi_mr": 0.0}, "RANGE": {"ema_trend": 0.0, "rsi_mr": 1.0}}
    out = combine(scores, pd.Series(["TREND_UP", "RANGE"]), weights)
    assert out["composite"].tolist() == [1.0, -1.0]
    assert out["conf_long"].tolist() == [100.0, 0.0]


def test_ensemble_unknown_regime_is_nan():
    scores = pd.DataFrame({c: [0.5] for c in COMPONENTS})
    out = combine(scores, pd.Series(["UNKNOWN"]), {"TREND": {"ema_trend": 1.0}})
    assert np.isnan(out["composite"].iloc[0])


def test_statarb_sign():
    idx = pd.date_range("2025-01-01", periods=400, freq="h", tz="UTC")
    rng = np.random.default_rng(2)
    b = pd.Series(100 * np.exp(np.cumsum(0.01 * rng.standard_normal(400))), index=idx)
    a = b * 1.0
    a.iloc[-1] = a.iloc[-1] * 0.95  # A suddenly cheap vs B
    sp = rolling_spread(a, b, 100)
    assert sp["z"].iloc[-1] < -2
    from cappatrader.config import StatArbConfig

    scores, _ = statarb_scores({"A": a, "B": b}, [["A", "B"]], StatArbConfig(window=100, min_corr=0.0))
    assert scores["A"].iloc[-1] > 0 > scores["B"].iloc[-1]


def test_engle_granger_detects_cointegration():
    rng = np.random.default_rng(4)
    x = np.cumsum(rng.standard_normal(2000)) + 500
    noise = np.zeros(2000)
    for t in range(1, 2000):
        noise[t] = 0.8 * noise[t - 1] + rng.standard_normal()
    a = pd.Series(np.exp((x + noise) / 100))
    b = pd.Series(np.exp(x / 100))
    assert engle_granger(a, b).cointegrated
    walk = pd.Series(np.exp((np.cumsum(rng.standard_normal(2000)) + 500) / 100))
    assert not engle_granger(walk, b).cointegrated


def test_ml_walk_forward_has_no_lookahead(cfg, btc):
    """Predictions for the past must not change when future bars are added or altered."""
    p_short, _ = walk_forward_proba(btc.iloc[:1800], cfg.signals.ml)
    future = btc.copy()
    future.iloc[1800:, future.columns.get_loc("close")] *= 1.5
    p_long, _ = walk_forward_proba(future, cfg.signals.ml)
    common = p_short.dropna().index
    assert len(common) > 100
    assert np.allclose(p_short[common], p_long[common])


def test_ml_score_threshold():
    s = ml_score(pd.Series([0.5, 0.6, 0.7, 0.3, np.nan]), 0.65)
    assert s.iloc[0] == 0 and s.iloc[1] == 0
    assert np.isclose(s.iloc[2], 0.4) and np.isclose(s.iloc[3], -0.4)
    assert np.isnan(s.iloc[4])


def test_pipeline_is_causal(cfg, universe):
    """A prepared row must be identical whether or not later bars exist."""
    cut = 1500
    head = {s: df.iloc[:cut] for s, df in universe.items()}
    full = prepare_universe(universe, cfg)["ETHUSDT"]
    part = prepare_universe(head, cfg)["ETHUSDT"]
    cols = ["composite", "regime", "score_statarb", "score_ema_trend", "atr", "vol_pct"]
    a = part[cols].iloc[-200:]
    b = full[cols].loc[a.index]
    pd.testing.assert_frame_equal(a, b)
