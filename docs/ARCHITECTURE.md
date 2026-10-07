# Mimari: veri nasıl akar, modüller nasıl bağlanır

Etkileşimli diyagramdaki 6 modül (Veri Kaynağı → Ön İşleme → Rejim → Sinyal → Risk → Emir Yürütme) koda
birebir karşılık gelir.

```
             ┌──────────────── Binance ────────────────┐
             │ WS: kline_1h · aggTrade · depth20@100ms │   REST: geçmiş mum, exchangeInfo,
             └───────────────┬─────────────────────────┘         funding (fapi), apiRestrictions
                             │ WS                                   Alternatif: Fear&Greed, RSS
 1. Veri Kaynağı   data/binance_ws.py  data/binance_rest.py  data/alt_data.py
                             │
 2. Ön İşleme      data/cleaning.py      TickFilter (absürt print), clean_klines (eksik mum, iğne, OHLC)
                   data/orderbook.py     OrderBook (imbalance, duvarlar, VWAP dolum), SpoofDetector, TradeFlow
                   features/indicators.py  EMA, RSI, MACD, ATR, ADX, Bollinger, Donchian (hepsi nedensel)
                             │  DF
 3. Rejim          regime/detector.py    TREND_UP/DOWN · RANGE · TRANSITION  +  vol: NORMAL/HIGH/EXTREME
                             │  REG
 4. Sinyal         signals/technical.py  ema_trend, macd, breakout, rsi_mr, bb_mr     ─┐
                   signals/statarb.py    pairs trading z-skoru                          │ her biri [-1, 1]
                   signals/ml.py         Random Forest, walk-forward eğitim             │
                   data/orderbook.py     orderflow (yalnız canlı)                       │
                   data/alt_data.py      sentiment (yalnız canlı)                      ─┘
                   signals/ensemble.py   rejim ağırlıklı ortalama → composite → güven 0-100
                   strategy/pipeline.py  hepsini tek DataFrame'de birleştirir
                   strategy/decision.py  ENTER / EXIT / HOLD (+ stil: TREND veya MR çıkış kuralı)
                   strategy/explain.py   Türkçe gerekçe satırları (Telegram + işlem kaydı)
                             │  SIG
 5. Risk           risk/sizing.py        Kelly risk payı × vol çarpanı / (ATR stop) → miktar, tavanlar
                   risk/stops.py         stop, izleyen stop, take-profit
                   risk/manager.py       kill-switch (%15 DD), günlük limit, max pozisyon, flash koruması
                             │  ORD
 6. Emir Yürütme   execution/costs.py    komisyon + kayma + etki + gecikme (backtest)
                   execution/paper.py    canlı defterde dolum + gerçek bekleme (paper)
                   execution/account.py  nakit/pozisyon defteri
                             │
                   live/state.py (SQLite)   notify/telegram.py + notify/messages.py
```

## Bir bar kapandığında (canlı)

1. `kline` mesajı `x=true` ile gelir → `LiveEngine.on_kline` barı ekler, eksik bar varsa REST ile tamamlar, temizler.
2. ML modeli `retrain_every` barda bir yeniden eğitilir (thread'de), son bar için olasılık üretilir.
3. Emir defteri + işlem akışından `orderflow`, funding/F&G/haberden `sentiment` skoru alınır.
4. `prepare_universe` (sembol + pairs ortakları) → `DecisionEngine.evaluate`.
5. Giriş onaylanırsa `size_position` → `PaperBroker.market_order` (gecikme kadar bekler, defterde yürür) →
   `Account.open_position` → SQLite → Telegram.
6. Her tick'te (`aggTrade`): absürt print filtresi, flash kontrolü, stop/TP kontrolü, izleyen stop güncellemesi.
7. Veri 30 sn gelmezse veya WS koparsa **koruma modu**: yeni işlem yok, açık pozisyonların stopları REST fiyatıyla izlenir.

## Backtest ile canlı neden aynı sonucu üretir

- Göstergeler, rejim, statarb ve ensemble vektörel ama **nedensel**: t anındaki değer yalnızca ≤ t verisini kullanır.
  `tests/test_signals.py::test_pipeline_is_causal` gelecek barlar eklendiğinde geçmiş satırların değişmediğini doğrular.
- ML olasılıkları walk-forward üretilir: t barını puanlayan model, etiket ufku t'den önce biten verilerle eğitilmiştir.
  `test_ml_walk_forward_has_no_lookahead` geleceği bozup geçmiş tahminlerin aynı kaldığını doğrular.
- Backtest kararı t kapanışında verir, t+1 açılışında (+ isteğe bağlı gecikme barı) maliyetlerle doldurur. Stoplar barın
  high/low aralığına göre, açılışta boşluk varsa açılış fiyatından tetiklenir; izleyen stop ancak bir sonraki bardan geçerlidir.

## Maliyet modeli

```
slippage_bps = taban
             + vol_katsayısı × ATR(bps)
             + etki_katsayısı × sqrt(emir tutarı / bar hacmi)
             + gecikme_z × ATR(bps) × sqrt(gecikme / bar süresi)
             + ek_kayma (stres testi)
dolum = referans × (1 ± slippage_bps/1e4),   komisyon = dolum tutarı × taker_fee
```

Stop mesafesi, beklenen gidiş-dönüş maliyetin riskin (1R) en fazla `risk.max_cost_r` kadarı olacağı şekilde alttan sınırlanır.

## Walk-forward

```
|------ IS 4000 bar (ızgara optimize) ------|-- OOS 1000 bar (en iyi parametreyle işlem) --|
                    |------ IS ------|-- OOS --|   … OOS uzunluğu kadar kaydırılır
```

GEÇTİ şartları: birleştirilmiş OOS getirisi > 0, OOS pencerelerinin en az yarısı kârlı, walk-forward verimliliği
(OOS/IS yıllık getiri) ≥ 0.5, OOS Sharpe ≥ 0.5.
