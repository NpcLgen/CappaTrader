"""Squeeze Momentum port vs a bar-by-bar Pine reference, VWAP anchoring, and the squeeze_vwap engine."""

import dataclasses
import math
import re

import numpy as np
import pandas as pd
import pytest

from cappatrader.config import SqueezeConfig
from cappatrader.models import LONG, SHORT, Position
from cappatrader.regime.detector import HIGH_VOL, RANGE, TRANSITION, TREND_DOWN, TREND_UP, UNKNOWN
from cappatrader.signals.squeeze import SQUEEZE_COLUMNS, pine_linreg, squeeze_layer, squeeze_momentum, vwap
from cappatrader.strategy.decision import FrameView
from cappatrader.strategy.squeeze_decision import SqueezeVwapEngine, explain_squeeze

BOOL_COLUMNS = [c for c in SQUEEZE_COLUMNS if c not in ("sqz_val", "sqz_upper", "sqz_lower")]


# ---------------------------------------------------------------- Pine reference (plain loops, None = na)

def _pine_reference(df: pd.DataFrame, sq: SqueezeConfig) -> dict[str, list]:
    o, h, l, c = (df[k].tolist() for k in ("open", "high", "low", "close"))
    n, L = len(c), sq.length

    def window(series, t, length):
        if t - length + 1 < 0:
            return None
        w = series[t - length + 1:t + 1]
        return None if any(x is None for x in w) else w

    def sma(series, t, length):
        w = window(series, t, length)
        return None if w is None else sum(w) / length

    def stdev(series, t, length):  # ta.stdev: biased
        w = window(series, t, length)
        if w is None:
            return None
        m = sum(w) / length
        return math.sqrt(sum((x - m) ** 2 for x in w) / length)

    def highest(series, t, length):
        w = window(series, t, length)
        return None if w is None else max(w)

    def lowest(series, t, length):
        w = window(series, t, length)
        return None if w is None else min(w)

    def linreg(series, t, length):  # ta.linreg(src, len, 0) = intercept + slope * (len - 1)
        w = window(series, t, length)
        if w is None:
            return None
        sx, sy = sum(range(length)), sum(w)
        sxx = sum(k * k for k in range(length))
        sxy = sum(k * y for k, y in enumerate(w))
        slope = (length * sxy - sx * sy) / (length * sxx - sx * sx)
        intercept = (sy - slope * sx) / length
        return intercept + slope * (length - 1)

    def nz(x):
        return 0.0 if x is None else x

    def gt(a, b):
        return a is not None and b is not None and a > b

    def lt(a, b):
        return a is not None and b is not None and a < b

    def hist(series, t, k):
        return series[t - k] if t - k >= 0 else None

    def add(a, b):
        return None if a is None or b is None else a + b

    def mul(a, k):
        return None if a is None else a * k

    def neg(a):
        return None if a is None else -a

    src = []
    for t in range(n):
        hh, ll, m = highest(h, t, L), lowest(l, t, L), sma(c, t, L)
        src.append(None if None in (hh, ll, m) else c[t] - ((hh + ll) / 2 + m) / 2)
    val = [linreg(src, t, L) for t in range(n)]
    abs_val = [None if x is None else abs(x) for x in val]
    tr = [None if t == 0 else max(h[t] - l[t], abs(h[t] - c[t - 1]), abs(l[t] - c[t - 1])) for t in range(n)]

    out = {k: [] for k in SQUEEZE_COLUMNS}
    for t in range(n):
        v0, v1, v2 = val[t], hist(val, t, 1), hist(val, t, 2)
        if sq.calc_mode == "fixed":
            upper, lower = sq.fixed_val, -sq.fixed_val
        elif sq.calc_mode == "dynamic":
            upper = mul(stdev(val, t, L), sq.dyn_mult)
            lower = neg(upper)
        else:
            upper = mul(highest(abs_val, t, L * 2), sq.perc_val / 100)
            lower = neg(upper)
        crossover = gt(v0, nz(v1)) and (v1 is not None and v1 <= nz(v2))
        crossunder = lt(v0, nz(v1)) and (v1 is not None and v1 >= nz(v2))
        basis, sd = sma(c, t, L), stdev(c, t, L)
        dev = mul(sd, sq.bb_mult)
        upper_bb, lower_bb = add(basis, dev), add(basis, neg(dev))
        ma, range_ma = sma(c, t, L), sma(tr, t, L)
        upper_kc, lower_kc = add(ma, mul(range_ma, sq.kc_mult)), add(ma, neg(mul(range_ma, sq.kc_mult)))
        out["sqz_val"].append(np.nan if v0 is None else v0)
        out["sqz_mom_up"].append(gt(v0, nz(v1)))
        out["sqz_mom_down"].append(lt(v0, nz(v1)))
        out["sqz_upper"].append(np.nan if upper is None else upper)
        out["sqz_lower"].append(np.nan if lower is None else lower)
        out["sqz_bull_pivot"].append(lt(v0, 0) and lt(v0, lower) and crossover)
        out["sqz_bear_pivot"].append(gt(v0, 0) and gt(v0, upper) and crossunder)
        out["sqz_on"].append(gt(lower_bb, lower_kc) and lt(upper_bb, upper_kc))

    on, up, down = out["sqz_on"], out["sqz_mom_up"], out["sqz_mom_down"]

    def at(flags, t, k):  # bool history before the first bar is false
        return flags[t - k] if t - k >= 0 else False

    for t in range(n):
        released = t >= 2 and on[t - 1] is False and on[t - 2] is True
        out["sqz_released"].append(released)
        out["sqz_up_trend"].append(released and c[t - 1] > o[t - 1])
        out["sqz_down_trend"].append(released and c[t - 1] < o[t - 1])
        out["sqz_up3"].append(up[t] and at(up, t, 1) and at(up, t, 2) and not at(up, t, 3))
        out["sqz_down3"].append(down[t] and at(down, t, 1) and at(down, t, 2) and not at(down, t, 3))
    return out


def _coil_frame(n=1100, seed=11) -> pd.DataFrame:
    """Random-walk legs alternating with tight mean-reverting coils that have wide candles:
    Bollinger bands fall inside the Keltner channel during the coils (squeeze on / release)."""
    rng = np.random.default_rng(seed)
    coil = (np.arange(n) // 150) % 2 == 1
    close = np.empty(n)
    x, anchor = 100.0, 100.0
    for t in range(n):
        if t > 0 and coil[t] and not coil[t - 1]:
            anchor = x
        x = x + 0.08 * (anchor - x) + 0.05 * rng.standard_normal() if coil[t] else x + 0.4 * rng.standard_normal()
        close[t] = x
    open_ = np.r_[close[0], close[:-1]] + 0.02 * rng.standard_normal(n)
    wick = np.where(coil, 1.0, 0.2) * np.abs(rng.standard_normal(n)) + 0.05
    idx = pd.date_range("2025-01-06", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + wick,
                         "low": np.minimum(open_, close) - wick, "close": close,
                         "volume": rng.lognormal(0, 0.5, n)}, index=idx)


_CASES = [
    ("BTCUSDT", {}), ("BTCUSDT", {"calc_mode": "dynamic"}), ("BTCUSDT", {"calc_mode": "percent"}),
    ("PAXGUSDT", {"length": 20}), ("PAXGUSDT", {"length": 20, "calc_mode": "dynamic", "dyn_mult": 1.0}),
    ("PAXGUSDT", {"length": 20, "calc_mode": "percent"}), ("XRPUSDT", {"length": 20, "fixed_val": 0.002}),
    ("coil", {}), ("coil", {"calc_mode": "dynamic", "dyn_mult": 1.5}), ("coil", {"calc_mode": "percent"}),
]


@pytest.fixture(scope="module")
def reference_runs(universe):
    runs = []
    for source, overrides in _CASES:
        df = _coil_frame() if source == "coil" else universe[source].iloc[:1100]
        sq = SqueezeConfig(**overrides)
        runs.append((source, overrides, squeeze_momentum(df, sq), _pine_reference(df, sq)))
    return runs


def test_squeeze_matches_pine_reference(reference_runs):
    for source, overrides, got, ref in reference_runs:
        for col in ("sqz_val", "sqz_upper", "sqz_lower"):
            exp = np.asarray(ref[col], dtype=float)
            scale = np.nanmax(np.abs(exp)) if np.isfinite(exp).any() else 1.0
            np.testing.assert_allclose(got[col].to_numpy(), exp, rtol=1e-9, atol=1e-10 * scale, equal_nan=True,
                                       err_msg=f"{source} {overrides} {col}")
        for col in BOOL_COLUMNS:
            assert got[col].dtype == bool, col
            mismatch = np.flatnonzero(got[col].to_numpy() != np.asarray(ref[col], dtype=bool))
            assert mismatch.size == 0, f"{source} {overrides} {col} differs at bars {mismatch[:10]}"


def test_reference_cases_exercise_every_flag(reference_runs):
    for col in BOOL_COLUMNS:
        assert sum(int(np.sum(ref[col])) for *_, ref in reference_runs) > 0, col
    for source, overrides, got, _ in reference_runs:
        if source == "coil" and not overrides:
            assert got["sqz_on"].sum() > 100 and got["sqz_released"].sum() >= 2
            assert got["sqz_up_trend"].sum() + got["sqz_down_trend"].sum() == got["sqz_released"].sum()


def test_squeeze_warmup_follows_pine():
    df = _coil_frame(400)
    L = 60
    fixed = squeeze_momentum(df, SqueezeConfig(length=L))
    val = fixed["sqz_val"]
    assert val.iloc[:2 * L - 2].isna().all() and val.iloc[2 * L - 2:].notna().all()
    # nz(val[1]) = 0 on the first valid bar
    assert fixed["sqz_mom_up"].iloc[2 * L - 2] == (val.iloc[2 * L - 2] > 0)
    assert not fixed[["sqz_mom_up", "sqz_mom_down"]].iloc[:2 * L - 2].any().any()
    assert (fixed["sqz_upper"] == 0.0).all()
    dyn = squeeze_momentum(df, SqueezeConfig(length=L, calc_mode="dynamic"))
    assert dyn["sqz_upper"].iloc[:3 * L - 3].isna().all() and dyn["sqz_upper"].iloc[3 * L - 3:].notna().all()
    pct = squeeze_momentum(df, SqueezeConfig(length=L, calc_mode="percent"))
    assert pct["sqz_upper"].iloc[:4 * L - 3].isna().all() and pct["sqz_upper"].iloc[4 * L - 3:].notna().all()
    np.testing.assert_allclose(dyn["sqz_lower"], -dyn["sqz_upper"], equal_nan=True)
    # ta.tr is na on bar 0, so the Keltner channel (and the squeeze) starts one bar after the Bollinger bands
    t = np.arange(200)
    close = 100 + 0.01 * np.sin(t)
    tight = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1.0},
                         index=pd.date_range("2025-01-06", periods=200, freq="h", tz="UTC"))
    on = squeeze_momentum(tight, SqueezeConfig(length=L))["sqz_on"]
    assert not on.iloc[:L].any() and on.iloc[L:].all()


def test_squeeze_momentum_streak(universe):
    out = squeeze_momentum(universe["BTCUSDT"].iloc[:600], SqueezeConfig(length=20))
    up, down, streak = out["sqz_mom_up"].to_numpy(), out["sqz_mom_down"].to_numpy(), out["sqz_mom_streak"].to_numpy()
    for t in range(1, len(out)):
        if up[t]:
            assert streak[t] == (streak[t - 1] + 1 if up[t - 1] else 1)
        elif down[t]:
            assert streak[t] == (streak[t - 1] - 1 if down[t - 1] else -1)
        else:
            assert streak[t] == 0
    assert (out["sqz_up3"] == (out["sqz_mom_streak"] == 3)).all()
    assert (out["sqz_down3"] == (out["sqz_mom_streak"] == -3)).all()


def test_pine_linreg_matches_polyfit():
    rng = np.random.default_rng(5)
    s = pd.Series(np.cumsum(rng.standard_normal(400)) + 50, index=pd.date_range("2025-01-01", periods=400, freq="h"))
    for length in (2, 5, 20, 60):
        for offset in (0, 3):
            out = pine_linreg(s, length, offset)
            assert out.index.equals(s.index)
            assert out.iloc[:length - 1].isna().all() and out.iloc[length - 1:].notna().all()
            for t in rng.integers(length - 1, len(s), size=25):
                window = s.to_numpy()[t - length + 1:t + 1]
                expected = np.polyval(np.polyfit(np.arange(length), window, 1), length - 1 - offset)
                assert out.iloc[t] == pytest.approx(expected, rel=1e-9, abs=1e-9)
    # a straight line is reproduced exactly; arrays are accepted too
    line = 3.0 + 0.5 * np.arange(30)
    np.testing.assert_allclose(pine_linreg(line, 10).to_numpy()[9:], line[9:])


def test_squeeze_layer_is_causal(universe):
    df = universe["BTCUSDT"].iloc[:700]
    sq = SqueezeConfig(length=20, calc_mode="percent")
    full = squeeze_layer(df, sq)
    assert list(full.columns) == [*SQUEEZE_COLUMNS, "sqz_mom_streak", "vwap"]
    k = 450
    pd.testing.assert_frame_equal(squeeze_layer(df.iloc[:k], sq), full.iloc[:k], check_exact=True)
    shocked = df.copy()
    rng = np.random.default_rng(9)
    factor = np.exp(0.05 * rng.standard_normal(len(df) - k))
    for col in ("open", "high", "low", "close"):
        shocked.iloc[k:, shocked.columns.get_loc(col)] *= factor
    shocked.iloc[k:, shocked.columns.get_loc("volume")] *= 10
    pd.testing.assert_frame_equal(squeeze_layer(shocked, sq).iloc[:k], full.iloc[:k], check_exact=True)


def test_squeeze_layer_does_not_depend_on_history_start(universe):
    """Live recomputes on bars.tail(max_bars): once warmed up (and past the first, partial VWAP period)
    a bar's values must be bit-identical to the backtest's whatever bar the history starts at."""
    df = universe["ETHUSDT"].iloc[:900]
    L = 20
    warm = 4 * L + 7 * 24  # slowest threshold warm-up plus one full UTC week of 1h bars
    for mode, anchor in (("dynamic", "D"), ("percent", "W"), ("fixed", "rolling")):
        sq = SqueezeConfig(length=L, calc_mode=mode, vwap_anchor=anchor)
        full = squeeze_layer(df, sq)
        for start in (5, 77, 200):
            part = squeeze_layer(df.iloc[start:], sq)
            pd.testing.assert_frame_equal(part.iloc[warm:], full.iloc[start + warm:], check_exact=True)


# ---------------------------------------------------------------- VWAP

def _ohlcv(index: pd.DatetimeIndex, seed=1, volume=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(index)
    close = 100 + np.cumsum(rng.standard_normal(n))
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    vol = rng.uniform(1, 10, n) if volume is None else np.asarray(volume, dtype=float)
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": vol}, index=index)


def _vwap_loop(df: pd.DataFrame, period_start) -> np.ndarray:
    tp = ((df["high"] + df["low"] + df["close"]) / 3).to_numpy()
    vol = df["volume"].to_numpy()
    keys = [period_start(ts) for ts in df.index.tz_convert("UTC")]
    out = []
    for t in range(len(df)):
        js = [j for j in range(t + 1) if keys[j] == keys[t]]
        out.append(sum(tp[j] * vol[j] for j in js) / sum(vol[j] for j in js))
    return np.array(out)


def test_vwap_resets_at_utc_midnight():
    idx = pd.date_range("2025-01-06", periods=72, freq="h", tz="UTC")
    df = _ohlcv(idx)
    got = vwap(df, "D")
    np.testing.assert_allclose(got.to_numpy(), _vwap_loop(df, lambda ts: ts.date()), rtol=1e-12)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    midnights = idx.hour == 0
    np.testing.assert_allclose(got[midnights], tp[midnights], rtol=1e-12)
    assert not np.allclose(got[~midnights], tp[~midnights])
    # the boundary is UTC midnight whatever timezone the index carries
    local = df.tz_convert("Europe/Istanbul")
    np.testing.assert_allclose(vwap(local, "D").to_numpy(), got.to_numpy(), rtol=1e-12)
    naive = df.tz_convert(None)
    np.testing.assert_allclose(vwap(naive, "D").to_numpy(), got.to_numpy(), rtol=1e-12)


def test_vwap_resets_at_utc_week_start():
    idx = pd.date_range("2025-01-06", periods=6 * 21, freq="4h", tz="UTC")  # Monday, three weeks
    df = _ohlcv(idx, seed=2)
    got = vwap(df, "W")
    def week(ts):
        return (ts.normalize() - pd.Timedelta(days=ts.dayofweek)).date()

    np.testing.assert_allclose(got.to_numpy(), _vwap_loop(df, week), rtol=1e-12)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mondays = (idx.dayofweek == 0) & (idx.hour == 0)
    assert mondays.sum() == 3
    np.testing.assert_allclose(got[mondays], tp[mondays], rtol=1e-12)
    tuesday = pd.Timestamp("2025-01-07", tz="UTC")
    assert got[tuesday] != pytest.approx(tp[tuesday])


def test_vwap_rolling():
    idx = pd.date_range("2025-01-06 05:00", periods=60, freq="h", tz="UTC")
    df = _ohlcv(idx, seed=3)
    got = vwap(df, "rolling", rolling_bars=5)
    assert got.iloc[:4].isna().all() and got.iloc[4:].notna().all()
    tp = ((df["high"] + df["low"] + df["close"]) / 3).to_numpy()
    vol = df["volume"].to_numpy()
    for t in range(4, len(df)):
        w = slice(t - 4, t + 1)
        assert got.iloc[t] == pytest.approx((tp[w] * vol[w]).sum() / vol[w].sum(), rel=1e-12)
    # rolling mode needs no datetime index
    assert np.allclose(vwap(df.reset_index(drop=True), "rolling", 5).to_numpy(), got.to_numpy(), equal_nan=True)
    with pytest.raises(TypeError):
        vwap(df.reset_index(drop=True), "D")


def test_vwap_zero_volume():
    idx = pd.date_range("2025-01-06", periods=30, freq="h", tz="UTC")
    volume = np.ones(30)
    volume[:3] = 0.0  # session opens with no trades
    volume[10:16] = 0.0
    df = _ohlcv(idx, seed=4, volume=volume)
    tp = ((df["high"] + df["low"] + df["close"]) / 3).to_numpy()
    got = vwap(df, "D").to_numpy()
    np.testing.assert_allclose(got[:3], np.cumsum(tp[:3]) / np.arange(1, 4))
    assert got[3] == pytest.approx(tp[3])  # zero-volume bars carry no weight
    assert np.isfinite(got).all()
    roll = vwap(df, "rolling", 4).to_numpy()
    np.testing.assert_allclose(roll[13:16], [tp[t - 3:t + 1].mean() for t in range(13, 16)])
    assert roll[12] == pytest.approx(tp[9])
    assert np.isfinite(roll[3:]).all()


def test_vwap_partial_first_period_is_nan():
    longer = _ohlcv(pd.date_range("2025-01-05", periods=80, freq="h", tz="UTC"), seed=5)
    df = longer.loc["2025-01-06 05:00":]  # history that starts mid-day
    got = vwap(df, "D")
    first_day = df.index < pd.Timestamp("2025-01-07", tz="UTC")
    assert got[first_day].isna().all() and got[~first_day].notna().all()
    # once a period is fully covered the value does not depend on where the history begins
    np.testing.assert_allclose(got[~first_day].to_numpy(), vwap(longer, "D").loc[got.index[~first_day]].to_numpy(),
                               rtol=1e-12)
    assert vwap(longer, "D").notna().all()


def test_vwap_is_causal():
    idx = pd.date_range("2025-01-06", periods=400, freq="h", tz="UTC")
    df = _ohlcv(idx, seed=6)
    for anchor in ("D", "W", "rolling"):
        full = vwap(df, anchor, 24)
        k = 250
        pd.testing.assert_series_equal(vwap(df.iloc[:k], anchor, 24), full.iloc[:k], check_exact=True)
        future = df.copy()
        future.iloc[k:] = future.iloc[k:].to_numpy() * 1.7
        pd.testing.assert_series_equal(vwap(future, anchor, 24).iloc[:k], full.iloc[:k], check_exact=True)


# ---------------------------------------------------------------- SqueezeVwapEngine

def _sqz_frame(vals, close=101.0, vwap_=100.0, regime=TREND_UP, vol_state="NORMAL", prob=0.8, atr=1.0):
    val = np.asarray(vals, dtype=float)
    prev = np.nan_to_num(np.r_[np.nan, val[:-1]], nan=0.0)
    idx = pd.date_range("2025-01-06", periods=len(val), freq="h", tz="UTC")
    return pd.DataFrame({
        "open": close, "high": close + 1.0, "low": close - 1.0, "close": close, "atr": atr,
        "regime": regime, "vol_state": vol_state, "hmm_prob": prob,
        "sqz_val": val, "sqz_mom_up": val > prev, "sqz_mom_down": val < prev, "vwap": vwap_,
    }, index=idx)


def _pos(side: int) -> Position:
    return Position(symbol="BTCUSDT", side=side, qty=1.0, entry_price=100.0, entry_time="2025-01-06T00:00:00+00:00",
                    stop_price=100.0 - side * 1.5, initial_stop=100.0 - side * 1.5, risk_per_unit=1.5,
                    best_price=100.0, entry_regime=TREND_UP, entry_confidence=80.0, entry_fee=0.0,
                    entry_slippage_cost=0.0, style="SQZ")


def _decide(cfg, frame, position=None, i=None):
    v = FrameView(frame)
    return SqueezeVwapEngine(cfg).evaluate("BTCUSDT", v, len(v) - 1 if i is None else i, position)


def test_long_state_entry(cfg):
    d = _decide(cfg, _sqz_frame([1.0, 2.0, 3.0]))
    assert d.action == "ENTER_LONG" and d.side == LONG and d.style == "SQZ"
    assert d.confidence == pytest.approx(80.0) and d.composite == pytest.approx(1.0)
    assert d.atr == 1.0 and d.close == 101.0 and d.regime == TREND_UP
    # "state": rising momentum is enough, below zero or after a dip
    assert _decide(cfg, _sqz_frame([-3.0, -5.0, -4.5])).action == "ENTER_LONG"
    assert _decide(cfg, _sqz_frame([1.0, 1.5])).composite == pytest.approx(0.5)
    # price must be above VWAP
    assert _decide(cfg, _sqz_frame([1.0, 2.0, 3.0], close=99.0)).action == "HOLD"
    assert _decide(cfg, _sqz_frame([1.0, 2.0, 3.0], close=100.0)).action == "HOLD"
    # falling momentum is no long
    assert _decide(cfg, _sqz_frame([3.0, 2.0])).action == "HOLD"


def test_long_turn_entry(cfg):
    cfg.strategy.squeeze.long_trigger = "turn"
    assert _decide(cfg, _sqz_frame([3.0, 2.0, 1.0, 2.0])).action == "ENTER_LONG"
    assert _decide(cfg, _sqz_frame([3.0, 3.0, 4.0])).action == "ENTER_LONG"  # flat bar is not "up"
    assert _decide(cfg, _sqz_frame([1.0, 2.0, 3.0])).action == "HOLD"  # already rising on the previous bar
    assert _decide(cfg, _sqz_frame([3.0])).action == "HOLD"  # no history to see the turn


def test_short_turn_entry(cfg):
    cfg.account.allow_short = True
    assert cfg.strategy.squeeze.short_trigger == "turn"
    d = _decide(cfg, _sqz_frame([1.0, 2.0, 3.0, 2.0], close=99.0))
    assert d.action == "ENTER_SHORT" and d.side == SHORT and d.style == "SQZ"
    assert d.composite == pytest.approx(-1.0)
    assert _decide(cfg, _sqz_frame([3.0, 2.0, 1.0], close=99.0)).action == "HOLD"  # already falling
    assert _decide(cfg, _sqz_frame([2.0, 2.0, 1.0], close=99.0)).action == "HOLD"  # previous bar not buyers
    assert _decide(cfg, _sqz_frame([1.0, 2.0, 3.0, 2.0], close=101.0)).action == "HOLD"  # above VWAP


def test_short_state_entry(cfg):
    cfg.account.allow_short = True
    cfg.strategy.squeeze.short_trigger = "state"
    assert _decide(cfg, _sqz_frame([3.0, 2.0, 1.0], close=99.0)).action == "ENTER_SHORT"
    assert _decide(cfg, _sqz_frame([1.0, 2.0, 1.5], close=99.0)).action == "ENTER_SHORT"
    assert _decide(cfg, _sqz_frame([1.0, 2.0, 3.0], close=99.0)).action == "HOLD"


def test_short_blocked_without_allow_short(cfg):
    assert cfg.account.allow_short is False
    for trigger in ("turn", "state"):
        cfg.strategy.squeeze.short_trigger = trigger
        assert _decide(cfg, _sqz_frame([1.0, 2.0, 3.0, 2.0], close=99.0)).action == "HOLD"


@pytest.mark.parametrize("regime,vol_state,allowed", [
    (TREND_UP, "NORMAL", True), (TREND_DOWN, "NORMAL", True), (RANGE, "HIGH", True),
    (TRANSITION, "NORMAL", False), (HIGH_VOL, "NORMAL", False), (UNKNOWN, "NORMAL", False),
    (TREND_UP, "EXTREME", False), (np.nan, "NORMAL", False),
])
def test_entry_regime_gate(cfg, regime, vol_state, allowed):
    cfg.account.allow_short = True
    long_ = _decide(cfg, _sqz_frame([1.0, 2.0], regime=regime, vol_state=vol_state))
    short = _decide(cfg, _sqz_frame([1.0, 2.0, 1.0], close=99.0, regime=regime, vol_state=vol_state))
    assert long_.action == ("ENTER_LONG" if allowed else "HOLD")
    assert short.action == ("ENTER_SHORT" if allowed else "HOLD")


def test_entry_regime_gate_follows_config(cfg):
    cfg.strategy.squeeze.tradeable_regimes = [TREND_UP, UNKNOWN]
    assert _decide(cfg, _sqz_frame([1.0, 2.0], regime=TREND_UP)).action == "ENTER_LONG"
    assert _decide(cfg, _sqz_frame([1.0, 2.0], regime=RANGE)).action == "HOLD"
    assert _decide(cfg, _sqz_frame([1.0, 2.0], regime=UNKNOWN)).action == "HOLD"  # never on UNKNOWN


def test_confirm_bars_three(cfg):
    cfg.account.allow_short = True
    sq = cfg.strategy.squeeze
    sq.confirm_bars = 3
    two_up, three_up, four_up = [5, 4, 3, 4, 5], [5, 4, 3, 4, 5, 6], [5, 4, 3, 4, 5, 6, 7]
    assert _decide(cfg, _sqz_frame(two_up)).action == "HOLD"
    assert _decide(cfg, _sqz_frame(three_up)).action == "ENTER_LONG"
    assert _decide(cfg, _sqz_frame(four_up)).action == "ENTER_LONG"  # state: at least N bars
    sq.long_trigger = "turn"
    assert _decide(cfg, _sqz_frame(two_up)).action == "HOLD"
    assert _decide(cfg, _sqz_frame(three_up)).action == "ENTER_LONG"  # exactly the 3rd bar of the up-leg
    assert _decide(cfg, _sqz_frame(four_up)).action == "HOLD"
    two_dn, three_dn, four_dn = [1, 2, 3, 2, 1], [1, 2, 3, 2, 1, 0], [1, 2, 3, 2, 1, 0, -1]
    assert _decide(cfg, _sqz_frame(two_dn, close=99.0)).action == "HOLD"
    assert _decide(cfg, _sqz_frame(three_dn, close=99.0)).action == "ENTER_SHORT"
    assert _decide(cfg, _sqz_frame(four_dn, close=99.0)).action == "HOLD"
    sq.short_trigger = "state"
    assert _decide(cfg, _sqz_frame(two_dn, close=99.0)).action == "HOLD"
    assert _decide(cfg, _sqz_frame(four_dn, close=99.0)).action == "ENTER_SHORT"


def test_confirm_bars_turn_matches_pine_ardisik3(cfg, universe):
    cfg.strategy.squeeze.confirm_bars = 3
    cfg.strategy.squeeze.long_trigger = "turn"
    out = squeeze_momentum(universe["ETHUSDT"].iloc[:600], SqueezeConfig(length=20))
    v = FrameView(out)
    eng = SqueezeVwapEngine(cfg)
    assert [eng.long_signal(v, i) for i in range(3, len(v))] == out["sqz_up3"].iloc[3:].tolist()
    assert out["sqz_up3"].sum() > 5


def test_early_exit_long_on_seller_pressure(cfg):
    d = _decide(cfg, _sqz_frame([1.0, 2.0, 1.5]), _pos(LONG))
    assert d.action == "EXIT" and "satıcıya" in d.exit_reason and "erken çıkış" in d.exit_reason
    # exits even when the entry filters would now block, and never flips into a short on the same call
    cfg.account.allow_short = True
    d = _decide(cfg, _sqz_frame([1.0, 2.0, 1.5], close=99.0, regime=UNKNOWN, vol_state="EXTREME"), _pos(LONG))
    assert d.action == "EXIT"


def test_early_exit_short_on_buyer_pressure(cfg):
    cfg.account.allow_short = True
    d = _decide(cfg, _sqz_frame([3.0, 2.0, 2.5], close=99.0), _pos(SHORT))
    assert d.action == "EXIT" and "alıcıya" in d.exit_reason
    d = _decide(cfg, _sqz_frame([3.0, 2.0, 2.5], close=101.0), _pos(SHORT))  # long setup present
    assert d.action == "EXIT"


def test_no_exit_while_momentum_holds(cfg):
    cfg.account.allow_short = True
    assert _decide(cfg, _sqz_frame([1.0, 2.0, 3.0]), _pos(LONG)).action == "HOLD"
    assert _decide(cfg, _sqz_frame([1.0, 2.0, 3.0], close=99.0), _pos(LONG)).action == "HOLD"  # below VWAP
    assert _decide(cfg, _sqz_frame([1.0, 2.0, 2.0]), _pos(LONG)).action == "HOLD"  # unchanged momentum
    assert _decide(cfg, _sqz_frame([3.0, 2.0, 1.0], close=99.0), _pos(SHORT)).action == "HOLD"
    assert _decide(cfg, _sqz_frame([3.0, 2.0, 2.0], close=99.0), _pos(SHORT)).action == "HOLD"


def test_engine_confidence_and_bad_inputs(cfg):
    frame = _sqz_frame([1.0, 2.0])
    assert _decide(cfg, frame.drop(columns="hmm_prob")).confidence == 50.0
    assert _decide(cfg, _sqz_frame([1.0, 2.0], prob=np.nan)).confidence == 50.0
    assert _decide(cfg, _sqz_frame([1.0, 2.0], atr=np.nan)).action == "HOLD"
    assert _decide(cfg, _sqz_frame([1.0, 2.0], vwap_=np.nan)).action == "HOLD"
    assert _decide(cfg, _sqz_frame([np.nan, np.nan])).action == "HOLD"
    # a reindexed bool column holds NaN for missing bars: never read as True
    gappy = frame.astype({"sqz_mom_up": object})
    gappy.iloc[-1, gappy.columns.get_loc("sqz_mom_up")] = np.nan
    assert _decide(cfg, gappy).action == "HOLD"


def _strategy_frame(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Synthetic prepared frame: squeeze layer plus deterministic stand-in regime-layer columns."""
    from cappatrader.features.indicators import atr

    layer = squeeze_layer(df, dataclasses.replace(cfg.strategy.squeeze, length=20))
    k = np.arange(len(df))
    labels = np.array([TREND_UP, RANGE, TREND_DOWN, TRANSITION], dtype=object)
    regime_cols = pd.DataFrame({
        "atr": atr(df["high"], df["low"], df["close"], cfg.regime.atr_period),
        "regime": np.where(k < 50, UNKNOWN, labels[(k // 37) % 4]),
        "vol_state": "NORMAL", "vol_pct": 0.5, "hmm_prob": 0.4 + 0.6 * ((k * 7919) % 100) / 100,
    }, index=df.index)
    regime_cols["atr_pct"] = regime_cols["atr"] / df["close"]
    return pd.concat([df, regime_cols, layer], axis=1)


def test_engine_decisions_are_causal(cfg, universe):
    cfg.account.allow_short = True
    df = universe["SOLUSDT"].iloc[:500]
    frame = _strategy_frame(df, cfg)
    eng = SqueezeVwapEngine(cfg)
    full = FrameView(frame)
    actions = []
    for i in range(0, len(frame), 3):
        part = FrameView(_strategy_frame(df.iloc[:i + 1], cfg))
        for pos in (None, _pos(LONG), _pos(SHORT)):
            a, b = eng.evaluate("SOLUSDT", full, i, pos), eng.evaluate("SOLUSDT", part, i, pos)
            assert (a.action, a.exit_reason) == (b.action, b.exit_reason), i
            assert a.composite == pytest.approx(b.composite, nan_ok=True)
            actions.append(a.action)
    assert {"ENTER_LONG", "ENTER_SHORT", "EXIT", "HOLD"} <= set(actions)


# ---------------------------------------------------------------- explain_squeeze

def _row(**over) -> pd.Series:
    base = {
        "close": 101.0, "atr": 2.0, "atr_pct": 2.0 / 101.0, "regime": TREND_UP, "vol_state": "NORMAL",
        "vol_pct": 0.4, "hmm_state": 1.0, "hmm_label": TREND_UP, "hmm_prob": 0.87,
        "garch_vol": 0.0062, "garch_longrun": 0.0055, "garch_ratio": 0.0062 / 0.0055, "garch_size_mult": 0.887,
        "sqz_val": 1.234, "sqz_mom_up": True, "sqz_mom_down": False, "sqz_upper": 0.0, "sqz_lower": 0.0,
        "sqz_bull_pivot": False, "sqz_bear_pivot": False, "sqz_on": False, "sqz_released": True,
        "sqz_up_trend": True, "sqz_down_trend": False, "sqz_up3": True, "sqz_down3": False,
        "sqz_mom_streak": 3, "vwap": 100.0,
    }
    base.update(over)
    return pd.Series(base, dtype=object)


def test_explain_squeeze_long(cfg):
    lines = explain_squeeze(_row(), LONG, cfg)
    assert all(isinstance(x, str) and x for x in lines)
    assert lines[0].startswith("Rejim: Trend ↑") and "HMM olasılığı %87" in lines[0]
    assert "GARCH" in lines[0] and "0.62%" in lines[0] and "×0.89" in lines[0]
    vwap_line = next(x for x in lines if "VWAP" in x)
    assert vwap_line.startswith("▲") and "üstünde" in vwap_line and "+1.00%" in vwap_line
    mom = next(x for x in lines if "momentum" in x.lower())
    assert mom.startswith("▲") and "alıcı baskısı" in mom and "3 bardır" in mom
    assert any(x.startswith("▲ Squeeze yukarı çözüldü") for x in lines)
    assert "Stop 1.5×ATR, hedef 1.1R" in lines[-1] and "satıcıya" in lines[-1]
    assert "≈ stop 98.0000" in lines[-1] and "hedef 104.3000" in lines[-1]


def test_explain_squeeze_short_and_fallbacks(cfg):
    row = _row(close=99.0, sqz_val=-0.5, sqz_mom_up=False, sqz_mom_down=True, sqz_mom_streak=-1,
               sqz_up3=False, sqz_released=False, sqz_up_trend=False, sqz_on=True, sqz_bear_pivot=True)
    for col in ("hmm_prob", "garch_vol", "garch_longrun", "garch_ratio", "garch_size_mult"):
        row = row.drop(col)
    row["adx"] = 31.0
    lines = explain_squeeze(row, SHORT, cfg)
    assert "ADX 31.0" in lines[0] and "ATR 1.98%" in lines[0] and "GARCH" not in lines[0]
    assert next(x for x in lines if "VWAP" in x).startswith("▲")  # below VWAP supports the short
    assert next(x for x in lines if "momentum" in x.lower()).startswith("▲")
    assert any(x.startswith("Squeeze aktif") for x in lines)
    assert any(x.startswith("▲ Ayı pivotu") for x in lines)
    assert "alıcıya" in lines[-1]
    # the same bar argues against a long
    longs = explain_squeeze(row, LONG, cfg)
    assert next(x for x in longs if "VWAP" in x).startswith("▼")


def test_momentum_values_never_use_an_exponent(cfg):
    # momentum is in price units: 1e4+ on BTC, 1e-6 on low-priced coins; "%.4g" printed "+1.2e+04"
    d = _decide(cfg, _sqz_frame([20000.0, 25000.0, 12000.0], atr=500.0), _pos(LONG))
    assert d.action == "EXIT" and "(momentum +12,000 < önceki +25,000)" in d.exit_reason
    lines = explain_squeeze(_row(sqz_val=3.4e-7, sqz_bull_pivot=True, sqz_lower=-1.25e-6), LONG, cfg)
    assert not any(re.search(r"\de[+-]\d", x) for x in lines)
    assert any("Squeeze momentum +0.00000034" in x for x in lines)
    assert any("alt eşiğin (-0.00000125)" in x for x in lines)
    assert any("Squeeze momentum +23,457 " in x for x in explain_squeeze(_row(sqz_val=23456.7), LONG, cfg))


def test_explain_squeeze_on_prepared_frame(cfg, universe):
    frame = _strategy_frame(universe["BNBUSDT"].iloc[:300], cfg)
    v = FrameView(frame)
    for i in (0, 10, 150, 299):
        for side in (LONG, SHORT):
            lines = explain_squeeze(v.row(i), side, cfg)
            assert lines and all(isinstance(x, str) for x in lines)
    lines = explain_squeeze(v.row(299), LONG, cfg)
    assert any("VWAP" in x for x in lines) and any("momentum" in x.lower() for x in lines)
