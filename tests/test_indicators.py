import numpy as np
import pandas as pd

from cappatrader.features import indicators as ind


def test_ema_matches_pandas(btc):
    out = ind.ema(btc["close"], 20)
    ref = btc["close"].ewm(span=20, adjust=False).mean()
    assert np.allclose(out.dropna(), ref[out.notna()])


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1, 60, dtype=float))
    assert ind.rsi(up, 14).dropna().iloc[-1] == 100.0
    down = pd.Series(np.arange(60, 1, -1, dtype=float))
    assert ind.rsi(down, 14).dropna().iloc[-1] < 1.0


def test_rsi_range(btc):
    r = ind.rsi(btc["close"]).dropna()
    assert r.between(0, 100).all()


def test_atr_positive_and_adx_bounded(btc):
    a = ind.atr(btc["high"], btc["low"], btc["close"]).dropna()
    assert (a > 0).all()
    dx = ind.adx(btc["high"], btc["low"], btc["close"]).dropna()
    assert dx["adx"].between(0, 100).all()


def test_bollinger_pctb(btc):
    bb = ind.bollinger(btc["close"]).dropna()
    inside = (btc["close"].loc[bb.index] <= bb["bb_upper"]) & (btc["close"].loc[bb.index] >= bb["bb_lower"])
    assert np.allclose(bb["bb_pctb"][inside].between(0, 1), True)


def test_donchian_excludes_current_bar(btc):
    dc = ind.donchian_prev(btc["high"], btc["low"], 10)
    assert dc["dc_high"].iloc[20] == btc["high"].iloc[10:20].max()


def test_bars_since_sign_change():
    s = pd.Series([1, 2, -1, -2, -3, 4])
    assert list(ind.bars_since_sign_change(s)) == [0, 1, 0, 1, 2, 0]
