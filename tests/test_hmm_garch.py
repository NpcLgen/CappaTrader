import numpy as np
import pandas as pd
import pytest
from arch import arch_model
from hmmlearn.stats import log_multivariate_normal_density

from cappatrader.config import GARCHConfig, HMMConfig, RegimeConfig
from cappatrader.data.synthetic import generate_universe
from cappatrader.features import indicators as ind
from cappatrader.regime import hmm as hmm_module
from cappatrader.regime import hybrid
from cappatrader.regime.detector import (
    HIGH_VOL, RANGE, TREND_DOWN, TREND_UP, UNKNOWN, VOL_EXTREME, VOL_HIGH, VOL_NORMAL, classify,
)
from cappatrader.regime.garch import (
    GARCHParams, OnlineGARCH, arch_backcast, fit_garch, garch_filter, pct_log_returns, unconditional_variance,
    walk_forward_garch,
)
from cappatrader.regime.hmm import (
    OnlineHMM, chunk_start, fit_hmm, forward_filter, hmm_features, label_states, refit_buckets, refit_schedule,
    starts_chunk, walk_forward_hmm,
)
from cappatrader.regime.hybrid import classify_regime, compute_regime_layer

N = 1400
CUTS = (700, 1000)  # mid-chunk for both schedules (epoch-anchored refits at 264, 364, ... with chunks of 100)


def _hmm_cfg(**kw) -> HMMConfig:
    return HMMConfig(**{"train_bars": 400, "retrain_every": 100, "min_train": 200, "n_iter": 25, **kw})


def _garch_cfg(**kw) -> GARCHConfig:
    return GARCHConfig(**{"train_bars": 400, "retrain_every": 100, "min_train": 200, **kw})


def _regime_cfg(method="hmm", vol_method="garch") -> RegimeConfig:
    return RegimeConfig(method=method, vol_method=vol_method, hmm=_hmm_cfg(), garch=_garch_cfg(), vol_lookback=200)


def _with_indicators(df: pd.DataFrame, cfg: RegimeConfig) -> pd.DataFrame:
    h, l, c = df["high"], df["low"], df["close"]
    return pd.concat([df, ind.atr(h, l, c, cfg.atr_period).rename("atr"), ind.adx(h, l, c, cfg.adx_period)], axis=1)


def _perturb_after(df: pd.DataFrame, cut: int, seed: int = 0) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(seed)
    shock = np.exp(np.cumsum(0.02 * rng.standard_normal(len(df) - cut)))
    for col in ("open", "high", "low", "close"):
        out.iloc[cut:, out.columns.get_loc(col)] *= shock
    return out


def _assert_same(a: pd.DataFrame, b: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-10, atol=1e-12)


@pytest.fixture
def df(btc):
    return btc.iloc[:N]


# -- causality -------------------------------------------------------------------

def test_walk_forward_hmm_is_causal(df):
    cfg = _hmm_cfg()
    full = walk_forward_hmm(df, cfg)
    assert full["hmm_label"].iloc[:200].eq(UNKNOWN).all() and full["hmm_state"].iloc[:200].isna().all()
    assert full["hmm_label"].iloc[300:].ne(UNKNOWN).all()
    assert full["hmm_prob"].dropna().between(0, 1 + 1e-12).all()
    for cut in CUTS:
        _assert_same(walk_forward_hmm(df.iloc[:cut], cfg), full.iloc[:cut])
        moved = walk_forward_hmm(_perturb_after(df, cut), cfg)
        _assert_same(moved.iloc[:cut], full.iloc[:cut])
        assert not np.allclose(moved["hmm_prob"].iloc[cut:], full["hmm_prob"].iloc[cut:])


def test_walk_forward_garch_is_causal(df):
    cfg = _garch_cfg()
    full = walk_forward_garch(df, cfg)
    assert full.iloc[:201].isna().all().all() and full.iloc[201:].notna().all().all()
    for cut in CUTS:
        _assert_same(walk_forward_garch(df.iloc[:cut], cfg), full.iloc[:cut])
        moved = walk_forward_garch(_perturb_after(df, cut), cfg)
        _assert_same(moved.iloc[:cut], full.iloc[:cut])
        assert not np.allclose(moved["garch_vol"].iloc[cut:], full["garch_vol"].iloc[cut:])


def test_compute_regime_layer_is_causal(df):
    cfg = _regime_cfg()
    full = compute_regime_layer(_with_indicators(df, cfg), cfg)
    assert list(full.columns) == hybrid.REGIME_COLUMNS
    for cut in CUTS:
        _assert_same(compute_regime_layer(_with_indicators(df.iloc[:cut], cfg), cfg), full.iloc[:cut])
        moved = compute_regime_layer(_with_indicators(_perturb_after(df, cut), cfg), cfg)
        _assert_same(moved.iloc[:cut], full.iloc[:cut])


# -- HMM ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # hmmlearn's own predict_proba on numpy >= 2.5
@pytest.mark.parametrize("covariance_type", ["full", "diag", "spherical", "tied"])
def test_forward_filter_equals_smoothing_at_the_last_point(df, covariance_type):
    x = hmm_features(df, 12).dropna().to_numpy()[:600]
    fitted = fit_hmm(x[:400], _hmm_cfg(covariance_type=covariance_type))
    post = forward_filter(fitted, x)
    np.testing.assert_allclose(post.sum(axis=1), 1.0)
    z = fitted.standardise(x)
    m = fitted.model  # its emission densities are ours; check them against hmmlearn's own
    np.testing.assert_allclose(m._compute_log_likelihood(z),
                               log_multivariate_normal_density(z, m.means_, m._covars_, covariance_type), rtol=1e-10)
    for t in (0, 1, 10, 150, 399, 599):
        np.testing.assert_allclose(post[t], fitted.model.predict_proba(z[: t + 1])[-1], atol=1e-9)
        np.testing.assert_allclose(forward_filter(fitted, x[: t + 1])[-1], post[t], atol=1e-12)


def test_forward_filter_missing_row_is_prediction_only(df, monkeypatch):
    x = hmm_features(df, 12).dropna().to_numpy()[:300].copy()
    fitted = fit_hmm(x, _hmm_cfg())
    x[150] = np.nan
    post = forward_filter(fitted, x)
    np.testing.assert_allclose(post[150], post[149] @ fitted.model.transmat_, atol=1e-12)
    # The numpy fallback (hmmlearn's compiled forward pass is private) gives the same posteriors.
    monkeypatch.setattr(hmm_module, "_forward_log_c", None)
    np.testing.assert_allclose(forward_filter(fitted, x), post, atol=1e-12)


def test_fit_is_deterministic_and_data_driven(df, monkeypatch):
    # The EM start is a pure function of the training rows: no k-means++ restarts, no random_state,
    # no warm start from an earlier fit (live refits on a different schedule than the backtest).
    from sklearn.cluster import KMeans

    monkeypatch.setattr(KMeans, "fit", lambda *a, **k: (_ for _ in ()).throw(AssertionError("random init")))
    x = hmm_features(df, 12).dropna().to_numpy()
    a = fit_hmm(x[:400], _hmm_cfg())
    fit_hmm(x[400:800], _hmm_cfg())
    b = fit_hmm(x[:400], _hmm_cfg(random_state=7))
    assert a.model.monitor_.converged
    for name in ("startprob_", "transmat_", "means_", "covars_"):
        np.testing.assert_array_equal(getattr(a.model, name), getattr(b.model, name))
    assert a.labels == b.labels


def test_fit_survives_a_state_collapsing_on_an_outlier(df, caplog):
    # A one-bar x20 print gives one extreme feature row; a state that takes only that row used to get
    # a singular covariance (hmmlearn's scalar prior is added to every entry) and every fit whose
    # window held it failed, freezing the regime model for train_bars bars.
    spiked = df.copy()
    spiked.iloc[800, spiked.columns.get_loc("close")] *= 20
    feats = hmm_features(spiked, 12)
    for lo in (412, 500, 700):
        fitted = fit_hmm(feats.iloc[lo:lo + 400], _hmm_cfg())
        assert fitted is not None
        assert np.linalg.eigvalsh(hmm_module._full_covars(fitted.model)).min() > 0
    with caplog.at_level("WARNING", logger="cappatrader.regime.hmm"):
        wf = walk_forward_hmm(spiked, _hmm_cfg())
    assert "HMM eğitimi başarısız" not in caplog.text
    assert np.isfinite(wf["hmm_prob"].iloc[212:]).all()


def test_fit_falls_back_to_log_space_when_scaling_underflows(df, monkeypatch):
    x = hmm_features(df, 12).dropna().to_numpy()[:400]
    want = fit_hmm(x, _hmm_cfg())
    assert want.model.implementation == "scaling"

    def underflow(self, X):
        raise ValueError("forward pass failed with underflow; consider using implementation='log' instead")

    monkeypatch.setattr(hmm_module._GaussianHMM, "_fit_scaling", underflow)
    got = fit_hmm(x, _hmm_cfg())
    assert got is not None and got.model.implementation == "log"
    np.testing.assert_allclose(got.model.means_, want.model.means_, atol=1e-8)
    assert got.labels == want.labels


def _segments(seed: int = 3) -> tuple[pd.DataFrame, np.ndarray]:
    spec = {"up": (0.002, 0.004, 800), "down": (-0.002, 0.004, 800), "flat": (0.0, 0.004, 800),
            "shock": (0.0, 0.012, 300)}
    order = ("up", "flat", "shock", "down", "flat", "up", "shock", "down")
    rng = np.random.default_rng(seed)
    r = np.concatenate([spec[s][0] + spec[s][1] * rng.standard_normal(spec[s][2]) for s in order])
    truth = np.concatenate([[s] * spec[s][2] for s in order])
    close = 100 * np.exp(np.cumsum(r))
    idx = pd.date_range("2025-01-01", periods=len(r), freq="h", tz="UTC")
    frame = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1.0}, index=idx)
    return frame, truth


def _interior(truth: np.ndarray, name: str, warmup: int) -> np.ndarray:
    first = np.r_[True, truth[1:] != truth[:-1]]
    age = pd.Series(first.cumsum()).groupby(first.cumsum()).cumcount().to_numpy()
    return (truth == name) & (age >= warmup)


def test_hmm_labels_synthetic_regimes():
    frame, truth = _segments()
    cfg = HMMConfig(n_states=4, feature_window=24, n_iter=50)
    feats = hmm_features(frame, cfg.feature_window)
    fitted = fit_hmm(feats, cfg)
    assert sorted(fitted.labels) == sorted([TREND_UP, TREND_DOWN, RANGE, HIGH_VOL])
    labels = np.asarray(fitted.labels, dtype=object)[forward_filter(fitted, feats).argmax(axis=1)]
    for name, want in (("up", TREND_UP), ("down", TREND_DOWN), ("flat", RANGE), ("shock", HIGH_VOL)):
        assert (labels[_interior(truth, name, 48)] == want).mean() > 0.6, name


def test_hmm_labels_stay_semantic_across_refits():
    frame, truth = _segments()
    cfg = HMMConfig(n_states=4, feature_window=24, n_iter=50, train_bars=6000, retrain_every=1000, min_train=2700)
    labels = walk_forward_hmm(frame, cfg)["hmm_label"].to_numpy()
    oos = np.arange(len(frame)) >= 2700 + cfg.feature_window
    for name, want in (("up", TREND_UP), ("down", TREND_DOWN)):
        assert (labels[_interior(truth, name, 48) & oos] == want).mean() > 0.6, name


def test_label_states_rule():
    cfg = HMMConfig(trend_threshold=0.15, high_vol_ratio=1.6)
    labels = label_states(np.array([0.002, -0.002, 0.0, 0.01]), np.array([0.004, 0.004, 0.004, 0.02]), cfg)
    assert labels == [TREND_UP, TREND_DOWN, RANGE, HIGH_VOL]
    # Not volatile enough relative to the calmest state: judged by the trend rule instead.
    assert label_states(np.array([0.0, 0.0]), np.array([0.004, 0.006]), cfg) == [RANGE, RANGE]


def test_online_hmm_matches_walk_forward(df):
    cfg = _hmm_cfg()
    wf = walk_forward_hmm(df, cfg)
    start, stop = 564, 664  # an anchored chunk whose training window [164, 564) is fully valid
    online = OnlineHMM(cfg)
    assert online.due(df) and online.infer_last(df)[0] == UNKNOWN
    assert online.fit_chunk(df.iloc[: start + 1])  # fits on the bars before the chunk: rows [164, 564)
    assert online.train_start == df.index[164]
    for t in (start, start + 37, stop - 1):
        label, prob, state = online.infer_last(df.iloc[: t + 1])
        assert label == wf["hmm_label"].iloc[t] and state == wf["hmm_state"].iloc[t]
        assert prob == pytest.approx(wf["hmm_prob"].iloc[t], abs=1e-10)
        assert not online.due(df.iloc[: t + 1])
    assert online.due(df.iloc[: stop + 1])


def test_refits_are_anchored_on_time_not_on_the_data_start(df):
    cfg = _hmm_cfg()
    b = refit_buckets(df.index, cfg.retrain_every)
    feats = hmm_features(df, cfg.feature_window).to_numpy()
    starts = [s for _, s, _ in refit_schedule(np.isfinite(feats).all(axis=1), 200, 400, 100, b)]
    assert starts[1:] == [t for t in range(starts[0] + 1, len(df)) if b[t] != b[t - 1]]
    assert all((df.index[t].value // 3_600_000_000_000) % 100 == 0 for t in starts[1:])  # epoch hour % 100
    # dropping the first 37 bars moves the first fit but not the later refit bars or their outputs
    late = df.iloc[37:]
    w = cfg.feature_window
    pd.testing.assert_frame_equal(hmm_features(late, w).iloc[w:], hmm_features(df, w).loc[late.index].iloc[w:],
                                  check_exact=True)  # bit-identical: EM is chaotic, rolling-sum drift is not ok
    a, z = walk_forward_hmm(df, cfg), walk_forward_hmm(late, cfg)
    pd.testing.assert_frame_equal(a.loc[df.index[864]:], z.loc[df.index[864]:])
    assert chunk_start(df.index[:700], 100) == 664 and starts_chunk(df.index[:665], 100)
    assert refit_buckets(pd.RangeIndex(10), 5) is None


def _live_replay(df: pd.DataFrame, model, t0: int, t1: int, max_bars: int, score, skip=()) -> tuple[list, list[int]]:
    """Bar-by-bar loop of LiveEngine: start-up fit on the bars before the current anchored chunk, then
    for every closed bar refit when due on the bars before it, then score the buffer's last row."""
    out, refits = [], []
    assert model.fit_chunk(df.iloc[max(0, t0 - max_bars): t0])
    for t in range(t0, t1):
        if t in skip:
            continue  # bars missed during an outage: the next bar is scored on the backfilled buffer
        bars = df.iloc[max(0, t + 1 - max_bars): t + 1]
        model.bars_since_fit += 1
        if model.due(bars):
            assert model.fit_chunk(bars)
            refits.append(t)
        out.append((t, score(model, bars)))
    return out, refits


def test_live_loop_matches_walk_forward_on_every_bar(df):
    # Live refits on the same epoch-anchored bars as the walk-forward and fits on the bars before the
    # one being scored, so a run started mid-chunk reproduces the backtest exactly, refit bars included.
    hcfg, gcfg = _hmm_cfg(), _garch_cfg()
    wf_h, wf_g = walk_forward_hmm(df, hcfg), walk_forward_garch(df, gcfg)
    max_bars = hcfg.train_bars + hcfg.feature_window + 100  # LiveEngine.max_bars lower bound

    for skip, expect in (((), [664, 764, 864]), (range(760, 770), [664, 770, 864])):  # second run: outage
        got, refits = _live_replay(df, OnlineHMM(hcfg), 611, 911, max_bars, lambda m, b: m.infer_last(b), skip)
        assert refits == expect
        for t, (label, prob, state) in got:
            assert label == wf_h["hmm_label"].iloc[t] and state == wf_h["hmm_state"].iloc[t], t
            assert prob == pytest.approx(wf_h["hmm_prob"].iloc[t], abs=1e-6), t

        got, refits = _live_replay(df, OnlineGARCH(gcfg), 600, 900, gcfg.train_bars + 100,
                                   lambda m, b: m.forecast_last(b), skip)
        assert refits == expect
        for t, vals in got:
            for col, value in vals.items():
                assert value == pytest.approx(wf_g[col].iloc[t], rel=1e-10), (t, col)


def test_min_train_above_train_bars_still_fits(df):
    # First fit once min_train valid rows exist, on the last train_bars of them.
    hcfg, gcfg = _hmm_cfg(train_bars=150), _garch_cfg(train_bars=150)
    h, g = walk_forward_hmm(df, hcfg), walk_forward_garch(df, gcfg)
    assert h["hmm_label"].iloc[:212].eq(UNKNOWN).all() and h["hmm_label"].iloc[212:].ne(UNKNOWN).all()
    assert g.iloc[:201].isna().all().all() and g.iloc[201:].notna().all().all()
    online_h, online_g = OnlineHMM(hcfg), OnlineGARCH(gcfg)
    assert not online_h.fit(df.iloc[:211]) and not online_g.fit(df.iloc[:200])
    assert online_h.fit(df.iloc[:212]) and online_g.fit(df.iloc[:201])
    t = 250
    label, prob, _ = online_h.infer_last(df.iloc[: t + 1])
    assert label == h["hmm_label"].iloc[t] and prob == pytest.approx(h["hmm_prob"].iloc[t], abs=1e-10)
    assert online_g.forecast_last(df.iloc[: t + 1])["garch_vol"] == pytest.approx(g["garch_vol"].iloc[t], rel=1e-12)


# -- GARCH -------------------------------------------------------------------------

@pytest.mark.parametrize("p,q", [(1, 1), (2, 1)])
def test_garch_recursion_matches_arch(df, p, q):
    y = pct_log_returns(df["close"])[1:801]
    params = fit_garch(y, GARCHConfig(p=p, q=q))
    assert params is not None and 0 < params.persistence < 1
    model = arch_model(y, mean="Constant", vol="GARCH", p=p, q=q, dist="normal", rescale=False)
    res = model.fit(disp="off", backcast=params.backcast)
    np.testing.assert_allclose(res.params.to_numpy(), np.r_[params.mu, params.omega, params.alphas, params.betas])
    var_next = garch_filter(params, y, params.backcast)
    np.testing.assert_allclose(np.sqrt(var_next[:-1]), res.conditional_volatility[1:], rtol=1e-10)
    first = params.omega + (params.alphas.sum() + params.betas.sum()) * params.backcast
    assert np.sqrt(first) == pytest.approx(res.conditional_volatility[0], rel=1e-10)
    # arch's forecast restarts the recursion from a backcast at the fitted mean, otherwise identical.
    fc = res.forecast(horizon=1, reindex=False).variance.to_numpy()[-1, 0]
    assert garch_filter(params, y, arch_backcast(y - params.mu))[-1] == pytest.approx(fc, rel=1e-10)
    assert var_next[-1] == pytest.approx(fc, rel=1e-3)
    default = arch_model(y, mean="Constant", vol="GARCH", p=p, q=q, dist="normal", rescale=False).fit(disp="off")
    np.testing.assert_allclose(default.conditional_volatility, res.conditional_volatility, rtol=1e-6)


def test_garch_filter_rolls_through_missing_returns():
    params = GARCHParams(mu=0.0, omega=0.1, alphas=np.array([0.1]), betas=np.array([0.8]), backcast=1.0,
                         uncond_var=1.0)
    out = garch_filter(params, np.array([1.0, np.nan, 2.0]), init_var=1.0)
    s0 = 0.1 + 0.9 * 1.0
    s1 = 0.1 + 0.1 * 1.0 + 0.8 * s0
    s2 = 0.1 + 0.9 * s1  # missing return replaced by its expectation
    s3 = 0.1 + 0.1 * 4.0 + 0.8 * s2
    np.testing.assert_allclose(out, [s1, s2, s3])
    np.testing.assert_allclose(garch_filter(params, np.array([1.0, np.inf, 2.0]), init_var=1.0), out)


def test_bad_print_does_not_poison_later_garch_forecasts(df):
    bad = df.copy()
    bad.iloc[700, bad.columns.get_loc("close")] = 0.0  # log return -inf then +inf
    with np.errstate(divide="ignore", invalid="ignore"):
        g = walk_forward_garch(bad, _garch_cfg())
    assert np.isfinite(g.iloc[201:].to_numpy()).all()


def test_longrun_variance_must_be_credible():
    # omega at its lower bound with persistence 0.9957 (< MAX_PERSISTENCE): omega / (1 - p) is 2e-6
    # percent^2 although the window's variance was 0.83; garch_ratio sat at ~250 (EXTREME) for 100 bars.
    a, b = np.array([0.05092222]), np.array([0.94480835])
    assert unconditional_variance(8.319e-9, a, b, sample_var=0.832) == 0.832
    assert unconditional_variance(0.004, a, b, sample_var=0.832) == pytest.approx(0.004 / (1 - a[0] - b[0]))
    assert unconditional_variance(0.05, a, b, sample_var=0.832) == 0.832  # 11.7 > 4 x sample
    df = generate_universe(symbols=["BTCUSDT"], n_bars=1400, seed=3)["BTCUSDT"]
    g = walk_forward_garch(df, _garch_cfg())
    assert g["garch_ratio"].max() < 5


def test_garch_columns(df):
    cfg = _garch_cfg()
    g = walk_forward_garch(df, cfg).dropna()
    np.testing.assert_allclose(g["garch_ratio"], g["garch_vol"] / g["garch_longrun"])
    np.testing.assert_allclose(g["garch_size_mult"], np.clip(1 / g["garch_ratio"], cfg.min_size_mult, 1.0))
    assert (g.loc[g["garch_ratio"] <= 1, "garch_size_mult"] == 1.0).all()
    assert g["garch_vol"].between(1e-4, 0.2).all()  # fractions, not percent


def test_online_garch_matches_walk_forward(df):
    cfg = _garch_cfg()
    wf = walk_forward_garch(df, cfg)
    start, stop = 564, 664  # anchored chunk; training returns are rows [164, 564)
    online = OnlineGARCH(cfg)
    assert online.due(df) and np.isnan(online.forecast_last(df)["garch_vol"])
    assert online.fit_chunk(df.iloc[: start + 1])
    # The forecast for bar `start` made at the end of the fit window is arch's one-step forecast.
    y = pct_log_returns(df["close"])[164:start]
    res = arch_model(y, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False).fit(
        disp="off", backcast=online.params.backcast)
    fc = res.forecast(horizon=1, reindex=False).variance.to_numpy()[-1, 0]
    assert online.forecast_last(df.iloc[:start])["garch_vol"] == pytest.approx(np.sqrt(fc) / 100, rel=1e-3)
    for t in (start, start + 41, stop - 1):
        got = online.forecast_last(df.iloc[: t + 1])
        for col, value in got.items():
            assert value == pytest.approx(wf[col].iloc[t], rel=1e-12), col
    assert online.due(df.iloc[: stop + 1]) and not online.due(df.iloc[: stop])
    # an outage that swallowed the chunk's first bars still refits on the walk-forward window
    late = OnlineGARCH(cfg)
    late.fit_chunk(df.iloc[: start + 1])
    assert late.due(df.iloc[: stop + 7]) and late.fit_chunk(df.iloc[: stop + 7])
    got = late.forecast_last(df.iloc[: stop + 7])
    assert got["garch_vol"] == pytest.approx(wf["garch_vol"].iloc[stop + 6], rel=1e-12)


# -- hybrid classifier -------------------------------------------------------------

def _fake_layers(index: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(index)
    cycle = [TREND_UP, RANGE, HIGH_VOL, TREND_DOWN, HIGH_VOL]
    labels = np.array([cycle[i % len(cycle)] for i in range(n)], dtype=object)
    labels[:50] = UNKNOWN
    hmm = pd.DataFrame({"hmm_state": np.where(np.arange(n) < 50, np.nan, np.arange(n) % 5 % 4),
                        "hmm_label": labels, "hmm_prob": 0.8}, index=index)
    ratio = np.array([[0.8, 1.6, 3.0][i % 3] for i in range(n)])
    ratio[:150] = np.nan
    garch = pd.DataFrame({"garch_vol": ratio * 0.01, "garch_longrun": 0.01, "garch_ratio": ratio,
                          "garch_size_mult": np.clip(1 / ratio, 0.25, 1)}, index=index)
    return hmm, garch


@pytest.mark.parametrize("method", ["hmm", "adx"])
@pytest.mark.parametrize("vol_method", ["garch", "atr"])
def test_classify_regime_dispatch(btc, method, vol_method):
    cfg = _regime_cfg(method, vol_method)
    frame = _with_indicators(btc.iloc[:600], cfg)
    hmm, garch = _fake_layers(frame.index)
    out = classify_regime(frame, cfg, hmm, garch)
    base = classify(frame, cfg)
    assert list(out.columns) == hybrid.REGIME_COLUMNS
    pd.testing.assert_series_equal(out["vol_pct"], base["vol_pct"])
    pd.testing.assert_series_equal(out["atr_pct"], base["atr_pct"])
    pd.testing.assert_frame_equal(out[hybrid.HMM_COLUMNS], hmm)
    pd.testing.assert_frame_equal(out[hybrid.GARCH_COLUMNS], garch)

    want_regime = hmm["hmm_label"] if method == "hmm" else base["regime"]
    assert (out["regime"] == want_regime).all()

    ratio = garch["garch_ratio"]
    want_vol = base["vol_state"].copy()
    if vol_method == "garch":
        g = np.where(ratio >= 2.5, VOL_EXTREME, np.where(ratio >= 1.5, VOL_HIGH, VOL_NORMAL))
        want_vol[ratio.notna()] = g[ratio.notna().to_numpy()]
    want_vol[(want_regime == HIGH_VOL) & (want_vol == VOL_NORMAL)] = VOL_HIGH
    assert (out["vol_state"] == want_vol).all()

    high_vol = out["regime"] == HIGH_VOL
    assert (high_vol.any() if method == "hmm" else not high_vol.any())
    assert out.loc[high_vol, "vol_state"].isin([VOL_HIGH, VOL_EXTREME]).all()
    if vol_method == "garch":
        assert (out.loc[ratio >= 2.5, "vol_state"] == VOL_EXTREME).all()
        assert (out.loc[ratio.isna() & ~high_vol, "vol_state"] == base.loc[ratio.isna() & ~high_vol, "vol_state"]).all()


def test_classify_regime_without_models(btc):
    cfg = _regime_cfg("hmm", "garch")
    frame = _with_indicators(btc.iloc[:400], cfg)
    out = classify_regime(frame, cfg, None, None)
    assert out["regime"].eq(UNKNOWN).all() and out["hmm_label"].eq(UNKNOWN).all()
    assert out[["hmm_state", "hmm_prob", *hybrid.GARCH_COLUMNS]].isna().all().all()
    assert (out["vol_state"] == classify(frame, cfg)["vol_state"]).all()


def test_compute_regime_layer_skips_unused_models(btc, monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("must not run")

    monkeypatch.setattr(hybrid, "walk_forward_hmm", boom)
    monkeypatch.setattr(hybrid, "walk_forward_garch", boom)
    cfg = _regime_cfg("adx", "atr")
    frame = _with_indicators(btc.iloc[:400], cfg)
    out = compute_regime_layer(frame, cfg)
    base = classify(frame, cfg)
    assert (out["regime"] == base["regime"]).all() and (out["vol_state"] == base["vol_state"]).all()
    assert out[hybrid.GARCH_COLUMNS].isna().all().all()
