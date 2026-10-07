import time

import numpy as np
import pandas as pd

from cappatrader.data.alt_data import AltDataHub, fng_score, funding_score, news_sentiment, parse_rss, score_headline
from cappatrader.data.cleaning import TickFilter, clean_klines, klines_to_frame
from cappatrader.data.orderbook import OrderBook, SpoofDetector, TradeFlow, orderflow_signal


def test_klines_to_frame():
    rows = [[1735689600000, "1", "2", "0.5", "1.5", "10", 1735693199999, "15", 3, "6", "9", "0"]]
    df = klines_to_frame(rows)
    assert df.index[0] == pd.Timestamp("2025-01-01", tz="UTC")
    assert df["close"].iloc[0] == 1.5 and df["quote_volume"].iloc[0] == 15.0


def test_clean_fills_gaps_fixes_ohlc_and_spikes(btc):
    df = btc.iloc[:300].copy()
    df = df.drop(df.index[[50, 51]])
    i = df.index[100]
    df.loc[i, "high"] = df.loc[i, "low"] * 0.9  # inconsistent OHLC
    j = df.index[200]
    df.loc[j, "close"] = df.loc[j, "close"] * 1.4  # absurd one-bar spike
    df.loc[j, "high"] = df.loc[j, "close"]
    out, rep = clean_klines(df, "1h")
    assert rep.gaps_filled == 2 and len(out) == 300
    assert rep.ohlc_fixed >= 1 and (out["high"] >= out["low"]).all()
    assert rep.spikes_fixed == 1
    assert abs(out.loc[j, "close"] / btc.loc[j, "close"] - 1) < 0.05


def test_tick_filter_rejects_spike_but_follows_real_move():
    f = TickFilter(max_dev=0.03, confirm=3)
    for p in [100, 100.1, 99.9, 100.2, 100.0, 100.1]:
        assert f.accept(p)
    assert not f.accept(150.0)  # single absurd print
    assert f.accept(100.1)
    assert not f.accept(90.0) and not f.accept(90.1)
    assert f.accept(89.9)  # third consecutive print confirms a real move


def _book():
    b = OrderBook("BTCUSDT")
    b.update([["100", "1"], ["99.9", "1"], ["99.8", "50"], ["99.7", "1"], ["99.6", "1"], ["99.5", "1"]],
             [["100.1", "1"], ["100.2", "1"], ["100.3", "1"], ["100.4", "1"], ["100.5", "1"]], now=0.0)
    return b


def test_orderbook_metrics():
    b = _book()
    assert b.mid == 100.05 and b.spread_bps > 0
    assert b.imbalance(10) > 0.8
    walls = b.walls(k=5)
    assert len(walls) == 1 and walls[0].side == "bid" and walls[0].price == 99.8
    avg, qty, levels = b.vwap_fill(1, 2.5)
    assert levels == 3 and avg == (100.1 + 100.2 + 0.5 * 100.3) / 2.5
    worse, _, _ = b.vwap_fill(1, 10)
    assert worse > avg  # walking past the visible book costs more


def test_spoof_detector():
    sd = SpoofDetector(k=5, max_life_s=10)
    b = _book()
    sd.on_book(b, now=0.0)
    gone = OrderBook("BTCUSDT")
    gone.update([["100", "1"], ["99.9", "1"], ["99.8", "1"], ["99.7", "1"], ["99.6", "1"], ["99.5", "1"]],
                [["100.1", "1"], ["100.2", "1"], ["100.3", "1"], ["100.4", "1"], ["100.5", "1"]], now=3.0)
    sd.on_book(gone, now=3.0)  # wall vanished after 3s, never traded at 99.8
    assert sd.recent("bid", 3.0) == 1
    sd2 = SpoofDetector(k=5, max_life_s=10)
    sd2.on_book(b, now=0.0)
    sd2.on_trade(99.8)  # wall genuinely hit
    sd2.on_book(gone, now=3.0)
    assert sd2.recent("bid", 3.0) == 0


def test_trade_flow_and_orderflow_signal():
    tf = TradeFlow(window_s=60)
    tf.add(0, 100, 3, buyer_is_maker=False)  # taker buy
    tf.add(1, 100, 1, buyer_is_maker=True)   # taker sell
    assert tf.imbalance()[0] == 0.5
    score, note = orderflow_signal(_book(), tf, SpoofDetector(), now=1.0, base_asset="BTC")
    assert 0 < score <= 1 and "alış duvarı" in note


RSS = """<?xml version="1.0"?><rss><channel>
<item><title>Bitcoin surges to record high as ETF inflows jump</title><pubDate>{now}</pubDate></item>
<item><title>Exchange hacked, Ethereum plunges</title><pubDate>{now}</pubDate></item>
<item><title>Gold rally continues</title><pubDate>{now}</pubDate></item>
</channel></rss>"""


def test_news_sentiment():
    from email.utils import formatdate

    heads = parse_rss(RSS.format(now=formatdate(time.time())))
    assert len(heads) == 3
    assert score_headline("Bitcoin surges to record high") == 1.0
    btc, n = news_sentiment(heads, "BTCUSDT", time.time())
    eth, _ = news_sentiment(heads, "ETHUSDT", time.time())
    gold, _ = news_sentiment(heads, "PAXGUSDT", time.time())
    assert btc > 0 and eth < 0 and gold > 0 and n == 1
    assert parse_rss("not xml") == []


def test_contrarian_scores():
    assert funding_score(0.0001) == 0.0 and funding_score(0.001) == -1.0 and funding_score(-0.0005) > 0
    assert fng_score(10) == 1.0 and fng_score(90) == -1.0 and np.isclose(fng_score(50), 0)


def test_alt_hub_sentiment_note():
    hub = AltDataHub(rest=None, feeds=[], symbols=["BTCUSDT"])
    hub.funding["BTCUSDT"] = 0.0003
    hub.fng = (20, "Extreme Fear")
    score, note = hub.sentiment("BTCUSDT")
    assert score is not None and "Korku" in note and "fonlama" in note
