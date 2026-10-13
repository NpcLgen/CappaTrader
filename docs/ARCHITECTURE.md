# Mimari: veri nasıl akar, modüller nasıl bağlanır

Mimari tarifindeki 6 modül (Veri Kaynağı → Ön İşleme → Rejim → Sinyal → Risk → Emir Yürütme) koda birebir
karşılık gelir. Rejim katmanı HMM + GARCH, varsayılan strateji Squeeze Momentum + VWAP, yürütme katmanı kağıt
hesap veya Binance USD-M futures'tur.

```
             ┌──────────────── Binance ────────────────┐
             │ WS: kline_1h · aggTrade · depth20@100ms │   REST: geçmiş mum, exchangeInfo,
             └───────────────┬─────────────────────────┘         funding (fapi), apiRestrictions
                             │ WS                                   Alternatif: Fear&Greed, RSS
 1. Veri Kaynağı   data/binance_ws.py  data/binance_rest.py  data/alt_data.py
                             │
 2. Ön İşleme      data/cleaning.py      TickFilter (absürt print), clean_klines (eksik mum, OHLC; iğne yalnızca raporlanır)
                   data/orderbook.py     OrderBook (imbalance, duvarlar, VWAP dolum), SpoofDetector, TradeFlow
                   features/indicators.py  EMA, RSI, MACD, ATR, ADX, Bollinger, Donchian (hepsi nedensel)
                             │  DF
 3. Rejim          regime/hmm.py         Gaussian HMM, walk-forward eğitim, İLERİ FİLTRELEME ile durum olasılığı
                                         → TREND_UP / TREND_DOWN / RANGE / HIGH_VOL (anlamsal etiketler)
                   regime/garch.py       GARCH(1,1): bir sonraki barın volatilitesi, uzun dönem oranı, boyut çarpanı
                   regime/hybrid.py      regime + vol_state (NORMAL/HIGH/EXTREME); method: hmm|adx, vol: garch|atr
                   regime/detector.py    eski ADX / ATR persentil sınıflandırıcı (yedek ve karşılaştırma)
                             │  REG
 4. Sinyal         signals/squeeze.py    Pine "Squeeze Momentum" portu (linreg momentum, eşikler, sıkışma, pivotlar) + VWAP
                   strategy/squeeze_decision.py  VWAP üstü + alıcı baskısı → LONG; VWAP altı + satıcıya dönüş → SHORT;
                                         karşı momentumda erken çıkış; stop ATR, hedef 1.1R   (strategy.name: squeeze_vwap)
                   ── ensemble stratejisi (strategy.name: ensemble) ──
                   signals/technical.py  ema_trend, macd, breakout, rsi_mr, bb_mr     ─┐
                   signals/statarb.py    pairs trading z-skoru                          │ her biri [-1, 1]
                   signals/ml.py         Random Forest, walk-forward eğitim             │
                   data/orderbook.py     orderflow (yalnız canlı)                       │
                   data/alt_data.py      sentiment (yalnız canlı)                      ─┘
                   signals/ensemble.py   rejim ağırlıklı ortalama → composite → güven 0-100
                   strategy/pipeline.py  hepsini tek DataFrame'de birleştirir
                   strategy/decision.py  ENTER / EXIT / HOLD (+ stil: TREND veya MR çıkış kuralı)
                   strategy/explain.py   Türkçe gerekçe satırları (Telegram + işlem kaydı)
                   strategy/factory.py   aktif stratejinin karar motoru, risk ayarı, açıklaması, GARCH çarpanı
                             │  SIG
 5. Risk           risk/sizing.py        Kelly risk payı / ATR stop → miktar → tavanlar → × GARCH / yüksek vol çarpanı
                   risk/stops.py         stop, izleyen stop, take-profit
                   risk/manager.py       kill-switch (%15 DD), günlük limit, max pozisyon, flash koruması
                             │  ORD
 6. Emir Yürütme   execution/costs.py    komisyon + kayma + etki + gecikme (backtest)
                   execution/paper.py    canlı defterde dolum + gerçek bekleme (paper)
                   execution/binance_futures.py  ccxt USD-M futures: bakiye, market emir, STOP/TP algo emirleri,
                                         pozisyon okuma, idempotent tekrar deneme (mode: live)
                   execution/account.py  nakit/pozisyon defteri (canlıda bakiye Binance'ten)
                             │
                   live/state.py (SQLite, mod başına ayrı dosya)  live/lock.py (tek bot kilidi)
                   notify/telegram.py + notify/messages.py
```

## Bir bar kapandığında (canlı)

1. `kline` mesajı `x=true` ile gelir → `LiveEngine.on_kline` barı ekler, eksik bar varsa REST ile tamamlar, temizler.
2. Online HMM ve GARCH backtest'le aynı barlarda yeniden eğitilir. Yeniden eğitim takvimi zamana sabitlidir
   (`zaman damgası // (bar × retrain_every)`). Eğitim, o dilimden önceki barlarla ve bir thread'de yapılır; bir veri
   kesintisi dilimin ilk barını yutsa bile aynı pencere kullanılır. Son bar için HMM durumu ileri filtrelemeyle,
   GARCH tahmini özyinelemeyle hesaplanır. Yalnızca ensemble stratejisinin kullandığı ML modeli bu takvime bağlı
   değildir. Bot başladığından beri her `retrain_every` barda bir eğitilir, bu yüzden ML oyu canlıda backtest'tekinden
   biraz farklı olabilir.
3. Emir defteri + işlem akışından `orderflow`, funding/F&G/haberden `sentiment` skoru alınır.
4. `prepare_universe` (sembol + pairs ortakları) → `strategy/factory.decide` (Squeeze/VWAP veya ensemble).
5. Giriş onaylanırsa `size_position` → kağıt hesapta `PaperBroker.market_order` (gecikme kadar bekler, defterde
   yürür); canlıda bakiye Binance'ten okunur, `BinanceFuturesBroker.market_order` gerçek emri gönderir ve borsaya
   STOP_MARKET + TAKE_PROFIT_MARKET girilir → `Account.open_position` → SQLite → Telegram.
6. Her tick'te (`aggTrade`): absürt print filtresi, flash kontrolü, stop/TP kontrolü, izleyen stop güncellemesi.
7. Veri 30 sn gelmezse veya WS koparsa **koruma modu**: yeni işlem yok, açık pozisyonların stopları REST fiyatıyla izlenir.
8. Canlıda her `live.sync_interval_s` saniyede bakiye ve pozisyonlar borsayla eşleştirilir: borsada kapanan pozisyon
   (borsa stop/hedef emri) işlem kaydına geçer, botun bilmediği pozisyon devralınıp korunur.

## Backtest ile canlı neden aynı sonucu üretir

- Göstergeler, rejim, statarb ve ensemble vektörel ama **nedensel**: t anındaki değer yalnızca ≤ t verisini kullanır.
  `tests/test_signals.py::test_pipeline_is_causal` gelecek barlar eklendiğinde geçmiş satırların değişmediğini doğrular.
- ML olasılıkları walk-forward üretilir: t barını puanlayan model, etiket ufku t'den önce biten verilerle eğitilmiştir.
- HMM t barındaki durumu yalnızca 1..t gözlemlerinden ileri filtrelemeyle bulur (Viterbi / smoothing geleceği görür,
  kullanılmaz); model t'den önceki verilerle eğitilmiştir. GARCH t barında t+1 için tahmin üretir.
  `test_ml_walk_forward_has_no_lookahead` geleceği bozup geçmiş tahminlerin aynı kaldığını doğrular.
- Yeniden eğitim barları zamana sabitlendiği ve HMM özellikleri her pencerede ayrı (kayan toplam olmadan)
  hesaplandığı için sonuç, verinin nerede başladığına bağlı değildir. Böylece ortada başlatılan canlı bir çalışma
  backtest'in rejim ve GARCH değerlerini bit düzeyinde aynen üretir
  (`tests/test_hmm_garch.py::test_live_loop_matches_walk_forward_on_every_bar`).
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

Ensemble stratejisinde stop mesafesi, beklenen gidiş-dönüş maliyetin riskin (1R) en fazla `risk.max_cost_r` kadarı olacağı şekilde alttan sınırlanır. Squeeze/VWAP stratejisinde bu sınır kapalıdır: stop tam `atr_stop_mult`×ATR, hedef `take_profit_r`×risk (`strategy/factory.py:strategy_risk`).

## Walk-forward

```
|------ IS 4000 bar (ızgara optimize) ------|-- OOS 1000 bar (en iyi parametreyle işlem) --|
                    |------ IS ------|-- OOS --|   … OOS uzunluğu kadar kaydırılır
```

GEÇTİ şartları: birleştirilmiş OOS getirisi > 0, OOS pencerelerinin en az yarısı kârlı, walk-forward verimliliği
(OOS/IS yıllık getiri) ≥ 0.5, OOS Sharpe ≥ 0.5.
