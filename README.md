# CappaTrader

Binance verisiyle çalışan, **kağıt hesapta (paper trading) 1000 USDT sanal bakiye** ile kripto pariteleri ve
tokenize altın (**PAXG/USDT**) üzerinde işlem yapan kantitatif analiz sistemi. Her alım/satımı nedeni, P&L'i ve
güncel bakiyesiyle Telegram'a bildirir.

Gereksinimler [`docs/PRD.md`](docs/PRD.md) (AlphaBot PRD v1.0) ve 6 maddelik mimari tarifine göre inşa edildi.
Modüllerin birbirine bağlanışı: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

> ⚠️ Bu yazılım yatırım tavsiyesi değildir. Bu sürüm **gerçek emir göndermez**. Kâr garantisi yoktur: sistemin
> asıl değeri, bir stratejinin işe yaramadığını **parayı kaybetmeden önce** göstermesidir (walk-forward + maliyet
> stres testi).

## Neler var

| # | Katman | Uygulama | Kod |
|---|--------|----------|-----|
| 1 | Veri altyapısı | WebSocket tick (aggTrade), Level-2 emir defteri (depth20@100ms), likidite duvarları, spoofing tespiti, alıcı/satıcı akışı, REST geçmiş + rate-limit koruması, veri temizleme (eksik mum, iğne, OHLC hatası), funding rate, Korku & Açgözlülük endeksi, RSS haber duyarlılığı | `cappatrader/data/` |
| 2 | Piyasa rejimi | ADX ile Trend↑ / Trend↓ / Yatay / Geçiş; ATR%'nin kendi geçmişindeki persentiliyle volatilite: yüksekse pozisyon ×0.5, aşırıysa yeni işlem yok | `cappatrader/regime/` |
| 3 | Sinyal motoru (ensemble) | EMA kesişimi, MACD sıfır çizgisi, Donchian kırılımı, RSI ve Bollinger ortalamaya dönüş, ETH/BTC ve SOL/ETH pairs trading (rolling hedge oranı + z-skor), Random Forest (geleceğe sızıntısız walk-forward eğitim, %65 olasılık eşiği), emir akışı, duyarlılık. Rejime göre ağırlıklandırılıp 0-100 güven skoruna çevrilir | `cappatrader/signals/`, `cappatrader/strategy/` |
| 4 | Dinamik risk | Volatiliteye göre lot (risk / (2×ATR)), yarım Kelly (geçmiş kazanma oranı ve K/Z oranından), maliyete duyarlı stop tabanı, izleyen stop (1R'de devreye girer, en az başabaş), %15 max-drawdown kill-switch, %4 günlük zarar limiti, ani hareket (flash) koruması | `cappatrader/risk/` |
| 5 | Walk-forward | In-sample'da parametre ızgarası optimize edilir, sadece hiç görülmemiş out-of-sample'da değerlendirilir, OOS'lar birleştirilir, GEÇTİ/BAŞARISIZ kararı verilir | `cappatrader/backtest/walkforward.py` |
| 6 | Slippage & gecikme | Her dolum: komisyon + taban kayma + volatilite + piyasa etkisi + gecikme kayması. Stres testi "dolum başına X bps daha kötü / N ms / N bar geç olursa hâlâ kârlı mı?" sorusunu yanıtlar ve başabaş kaymasını hesaplar. Canlıda gecikme gerçekten beklenir ve emir canlı defterde yürütülür | `cappatrader/execution/`, `cappatrader/backtest/stress.py` |

Backtest ve canlı kağıt hesap **aynı** karar, risk, stop ve maliyet kodunu kullanır. Testler, hiçbir göstergenin ve
ML tahmininin gelecekteki veriyi görmediğini doğrular.

## Kurulum

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # anahtarları .env'e yazın, asla commit etmeyin
python -m pytest -q         # 65 test, internet gerektirmez
```

`.env`:

```
BINANCE_API_KEY=...      # paper mod için ŞART DEĞİL; verilirse yalnızca yetki kontrolü yapılır
BINANCE_SECRET_KEY=...
TELEGRAM_TOKEN=...       # BotFather token
TELEGRAM_CHAT_ID=...
```

Strateji/risk ayarları `config/config.yaml` dosyasında (açıklamalı).

## Kullanım

```bash
python -m cappatrader check                 # Binance + Telegram bağlantısı, API anahtarı yetkileri
python -m cappatrader demo --skip-wf        # internet olmadan sentetik veriyle tüm zincir (~1 dk)

python -m cappatrader download --days 730   # 2 yıllık 1h mum indir (data/ altına önbellek)
python -m cappatrader backtest --days 730   # maliyetler dahil backtest + PRD KPI kontrolü
python -m cappatrader walkforward --days 730   # IS optimizasyon / OOS doğrulama, GEÇTİ/BAŞARISIZ
python -m cappatrader stress --days 730     # slippage / gecikme duyarlılığı ve başabaş noktası
python -m cappatrader pairs --days 365      # pairs trading çiftleri için eşbütünleşme testi

python -m cappatrader paper                 # canlı veriyle 1000 USDT kağıt hesap + Telegram
python -m cappatrader status                # bakiye, açık pozisyonlar, son işlemler
python -m cappatrader resume                # kill-switch sonrası manuel devam
python -m cappatrader reset --yes           # kağıt hesabı 1000 USDT'ye sıfırla
```

Raporlar `reports/<tür>_<zaman>/` altına (`summary.txt`, `metrics.json`, `trades.csv`, `equity.csv`) yazılır.
Kağıt hesap durumu `state/cappatrader.sqlite` içinde kalıcıdır: bot yeniden başlarsa pozisyon ve stop takibi devam eder.

### Önerilen sıra

1. `walkforward` ve `stress` gerçek veride **GEÇTİ** demeden paper sonuçlarına güvenmeyin.
2. `paper` modunu en az 4-8 hafta çalıştırın; paper sonuçları OOS sonuçlarıyla tutarlı mı bakın.
3. Gerçek para ancak bundan sonra (bu sürümde canlı emir yok, bilinçli olarak).

## Telegram mesajı örneği

Kodun ürettiği gerçek çıktı (fiyatlar sentetik veriden geldiği için gerçekçi değil):

```
🟢 ALIŞ (LONG) — ETH/USDT · KAĞIT HESAP
Fiyat: 49,672.27 (sinyal 49,667.30 · slippage 1.0 bps · gecikme 0 ms)
Miktar: 0.00604 ETH ≈ 300.02 USDT · komisyon 0.30 USDT
Stop: 48,405.05 (-2.55%) · risk 7.65 USDT (%0.77 kasa)
Rejim: Trend ↑ (yükseliş) · güven %89 · strateji: Trend takibi / momentum

Neden alındı?
• Rejim: Trend ↑ (yükseliş) · ADX 49.7 · volatilite normal (ATR 1.28%, 51. persentil)
• Ensemble skoru +0.78 → güven %89 (eşik %65)
• ▲ EMA20 > EMA50 (+2.53 ATR)
• ▲ MACD sıfır çizgisinin üstünde (+663.7, histogram -91.54)
• ▲ Fiyat 55 barlık kanalın %82 seviyesinde

Pozisyon boyutu
• Kelly için yetersiz geçmiş (0/30 işlem) → sabit %1.00 risk
• Stop mesafesi 2×ATR = 2.55% (ATR yüksekse pozisyon otomatik küçülür)
• Pozisyon tavanı (%30 kasa / kullanılabilir nakit) → gerçek risk 7.65 USDT

💼 Güncel bakiye: 999.67 USDT (-0.03% başlangıç 1,000.00 USDT)
Nakit 699.68 USDT · açık pozisyon 1
```

```
🔴 SATIŞ — pozisyon kapandı — ETH/USDT · KAĞIT HESAP
Giriş 49,672.27 → Çıkış 50,705.24 (slippage 2.0 bps) · süre 0 sa 0 dk
✅ P&L: +5.63 USDT (+1.88% · +0.74R) · komisyon 0.61 USDT dahil

Neden satıldı?
• İzleyen stop (trailing) tetiklendi — stop 50,715.39, kilitlenen hareket +2.10%

Giriş gerekçesi (özet)
• Rejim: Trend ↑ (yükseliş) · ADX 49.7 · volatilite normal (ATR 1.28%, 51. persentil)
• Ensemble skoru +0.78 → güven %89 (eşik %65)
• ▲ EMA20 > EMA50 (+2.53 ATR)

💼 Güncel bakiye: 1,005.63 USDT (+0.56% başlangıç 1,000.00 USDT)
Nakit 1,005.63 USDT · açık pozisyon 0
Realize P&L toplam +5.63 USDT · 1 işlem · kazanma %100
```

Ayrıca: başlangıç mesajı (API anahtarı uyarılarıyla), günlük özet (20:00 UTC), kill-switch, günlük zarar limiti,
flash hareket ve veri kesintisi (koruma modu) uyarıları.

## Gerçekçi beklentiler

- **Komisyon en büyük düşman.** Binance spot taker %0.1 (BNB ile %0.075); gidiş-dönüş ~%0.2 + kayma. 15 dakikalık
  barlarda ATR stopları bu maliyete yakındır, bu yüzden varsayılan 1 saat ve stop, maliyet riskin en fazla %20'si
  olacak şekilde genişletilir. `demo` çıktısı bunu sayılarla gösterir (sıfır maliyetle +%103 görünen strateji,
  gerçekçi maliyetle zarar eder).
- **PRD KPI'ları (Sharpe > 1.5, max DD < %15) hedeftir, garanti değil.** Raporlar bunları dürüstçe ✔/✘ olarak işaretler.
  Bu hedefleri tutturmak için parametre zorlamak tam olarak kaçınılan aşırı optimizasyondur.
- 1000 USDT ile TWAP/VWAP, FIX, C++/Rust ve <50 ms tick-to-trade bir avantaj sağlamaz; Binance L3 defter vermez.
  Bunlar bilinçli olarak kapsam dışı bırakıldı (bkz. Yol haritası).
- Spot hesap yalnızca long açar. Pairs trading sinyali bu yüzden "görece ucuz bacağı al" olarak kullanılır;
  `account.allow_short: true` ile 1x perpetual (short) simülasyonu açılabilir.
- Emir akışı ve duyarlılık geçmişte arşivlenmediği için yalnızca canlıda katkı verir; backtest bu iki bileşen olmadan ölçer.

## Sunucuda çalıştırma

PRD gereği borsa sunucularına yakın bir bölge seçin (Binance için ör. AWS Tokyo `ap-northeast-1`).
Binance.com bazı ülkelerden (ör. ABD) erişimi engeller; o durumda `config.yaml` içindeki `data-api.binance.vision`
satırlarını açın (salt piyasa verisi).

```bash
docker compose up -d --build     # .env dosyasını okur, state/ ve logs/ kalıcıdır
docker compose logs -f
```

## Güvenlik

- Anahtarlar yalnızca `.env` / ortam değişkeninde; loglarda maskelenir; YAML'a yazılması engellenir.
- Paper mod için Binance anahtarı gerekmez. Verirseniz yalnızca **Okuma** yetkili, **IP kısıtlı** ve
  **Para Çekme kapalı** bir anahtar kullanın; bot başlarken bunu kontrol edip Telegram'a uyarır.

## Yol haritası

- Canlı emir yürütme (Binance spot testnet → küçük gerçek bakiye), borsa tarafı stop emirleri
- LSTM / gradient boosting modelleri, HMM rejim modeli, GARCH volatilite tahmini
- On-chain metrikler (Glassnode/CryptoQuant, ücretli API), daha iyi haber NLP'si
- Maker (limit) emirlerle komisyon düşürme, Telegram'dan `/durum` komutu
