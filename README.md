# CappaTrader

Binance verisiyle çalışan kantitatif kripto işlem sistemi. Kripto pariteler ve tokenize altın (**PAXG/USDT**)
üzerinde iki modda çalışır:

- **Kağıt hesap (`paper`):** canlı piyasa verisi, sanal bakiye (varsayılan 1000 USDT ya da gerçek Binance bakiyeniz).
- **Canlı (`live`):** Binance **USD-M futures**'ta gerçek emir. Varsayılan ortam **demo** (Binance demo trading,
  sanal para). Gerçek para (mainnet) ancak açık bir onay cümlesiyle açılır.

Her alım/satımı nedeni, P&L'i ve güncel bakiyesiyle Telegram'a bildirir.

Gereksinimler: [`docs/PRD.md`](docs/PRD.md) (AlphaBot PRD v1.0) ve [`docs/PRD_HMM_GARCH.md`](docs/PRD_HMM_GARCH.md)
(HMM & GARCH tabanlı Squeeze Momentum/VWAP botu). Modüllerin bağlanışı: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

> ⚠️ Bu yazılım yatırım tavsiyesi değildir ve kâr garantisi yoktur. Kaldıraçlı futures işlemi yatırdığınız paranın
> tamamını kaybetmenize yol açabilir. Canlı emir kodu bu geliştirme ortamında **gerçek Binance'e karşı hiç
> çalıştırılmadı** (ağ erişimi yoktu); yalnızca sahte borsa ile test edildi. Önce `live.env: demo` ile deneyin.

## Neler var

| # | Katman | Uygulama | Kod |
|---|--------|----------|-----|
| 1 | Veri | WebSocket tick (aggTrade), Level-2 emir defteri, spot veya USD-M futures piyasa verisi, REST geçmiş + rate-limit koruması, veri temizleme, funding rate, Korku & Açgözlülük, RSS haber duyarlılığı, **Binance bakiye okuma** | `cappatrader/data/`, `execution/binance_futures.py` |
| 2 | Piyasa rejimi | **Hidden Markov Model** (ADX yerine): getiri ve volatiliteden piyasa durumlarını kendisi öğrenir (Trend↑, Trend↓, Yatay, Yüksek volatilite), yalnızca ileri filtreleme ile (geleceği görmez). **GARCH(1,1)**: bir sonraki barın volatilitesini tahmin eder; uzun dönem ortalamasının üstündeyse pozisyon küçülür, 2.5 katını aşarsa yeni işlem açılmaz | `cappatrader/regime/` |
| 3 | Strateji | **Squeeze Momentum + VWAP** (Pine Script'inizin birebir Python portu, varsayılan) veya eski çok sinyalli **ensemble** | `signals/squeeze.py`, `strategy/` |
| 4 | Risk | Kasa yüzdesi risk / ATR stop mesafesi ile lot, yarım Kelly, GARCH boyut çarpanı, maliyet tabanı, %15 drawdown kill-switch, %4 günlük limit, flash koruması | `cappatrader/risk/` |
| 5 | Walk-forward | In-sample optimizasyon, yalnız out-of-sample değerlendirme, GEÇTİ/BAŞARISIZ | `backtest/walkforward.py` |
| 6 | Slippage & gecikme | Komisyon + kayma + etki + gecikme; "X bps daha kötü dolumda hâlâ kârlı mı?" stres testi | `execution/`, `backtest/stress.py` |
| 7 | Canlı yürütme | ccxt ile Binance USD-M futures: market emir, borsada STOP_MARKET + TAKE_PROFIT_MARKET koruma emirleri, başlangıçta ve 30 sn'de bir pozisyon senkronu, sahipsiz pozisyon devralma, kaldıraç/marjin ayarı, tekrar denemede çift emir koruması | `execution/binance_futures.py`, `live/runner.py` |

Backtest, kağıt hesap ve canlı mod **aynı** karar, risk, stop ve maliyet kodunu kullanır. Testler hiçbir
göstergenin, HMM/GARCH tahmininin ve ML modelinin gelecekteki veriyi görmediğini doğrular.

## Squeeze Momentum + VWAP stratejisi

Pine Script'teki momentum: `val = linreg(close − ort(ort(en yüksek, en düşük), SMA), 60)`.

- **Alıcı baskısı:** `val` bir önceki bardan yüksek (sıfırın altında veya üstünde olması fark etmez).
  **Satıcı baskısı:** düşük.
- **LONG:** kapanış **VWAP üstünde** ve momentumda **alıcı baskısı** var.
- **SHORT:** kapanış **VWAP altında** ve momentum **alıcıdan satıcıya döndü**.
- **Stop:** giriş anındaki ATR'ye göre (varsayılan 1.5×ATR). **Hedef:** riskin **1.1 katı** (1.1R).
- **Erken çıkış:** hedefe varmadan LONG'da satıcı baskısı, SHORT'ta alıcı baskısı gelirse pozisyon mum
  kapanışında piyasa emriyle kapanır.
- **Filtre:** HMM rejimi `TREND_UP`, `TREND_DOWN` veya `RANGE` olmalı (yüksek volatilite durumunda işlem yok) ve
  GARCH volatilitesi aşırı olmamalı.
- VWAP her gün 00:00 UTC'de sıfırlanır (TradingView'ın kripto varsayılanı; `vwap_anchor` ile değişir).
- Pine'daki sıkışma (squeeze), pivot üçgenleri ve "ardışık 3 bar" bilgileri hesaplanır ve Telegram açıklamasında
  gösterilir. `confirm_bars: 3` ile ardışık 3 bar şartı girişe de uygulanabilir.

Bütün ayarlar `config/config.yaml` dosyasında, `strategy.squeeze` altında.

## Kurulum

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # anahtarları .env'e yazın, asla commit etmeyin
python -m pytest -q         # internet gerektirmez
```

`.env`:

```
BINANCE_API_KEY=...      # paper: isteğe bağlı (salt okuma). live: demo.binance.com veya binance.com futures anahtarı
BINANCE_SECRET_KEY=...
TELEGRAM_TOKEN=...       # BotFather token
TELEGRAM_CHAT_ID=...
```

## Kullanım

```bash
python -m cappatrader check                 # Binance + Telegram bağlantısı, API anahtarı yetkileri
python -m cappatrader balance               # Binance futures ve spot bakiyesi (salt okuma)
python -m cappatrader demo --skip-wf        # internet olmadan sentetik veriyle tüm zincir

python -m cappatrader download --days 730   # 2 yıllık 1h mum indir (data/ altına önbellek)
python -m cappatrader backtest --days 730   # maliyetler dahil backtest + PRD KPI kontrolü
python -m cappatrader walkforward --days 730   # IS optimizasyon / OOS doğrulama, GEÇTİ (çıkış 0) / BAŞARISIZ (çıkış 2)
python -m cappatrader stress --days 730     # slippage / gecikme duyarlılığı ve başabaş noktası

python -m cappatrader paper                 # canlı veriyle kağıt hesap + Telegram
python -m cappatrader live                  # mode: live — Binance futures'ta gerçek emir (önce demo)
python -m cappatrader status                # bakiye, açık pozisyonlar, son işlemler
python -m cappatrader resume                # kill-switch sonrası manuel devam (bot durdurulmuşken)
python -m cappatrader reset --yes           # hesap kaydını sıfırla (bot durdurulmuşken)
```

Araştırma komutları HMM ve GARCH'ı her sembol için bir kez walk-forward eğitir; 2 yıllık 1h veride sembol başına
yaklaşık 15-20 saniye sürer. Raporlar `reports/<tür>_<zaman>/` altına yazılır. Kağıt hesap `state/cappatrader.sqlite`,
canlı hesap `state/cappatrader_live_demo.sqlite` / `_live_mainnet.sqlite` içinde ayrı tutulur.

## Canlı işlem (Binance USD-M futures)

1. **Demo anahtarı alın:** [demo.binance.com](https://demo.binance.com) → API Management. Demo ortamı gerçek
   piyasa fiyatlarıyla sanal para kullanır.
2. Binance futures hesabınızı **One-way (tek yönlü) pozisyon moduna** alın; hedge modunda bot çalışmayı reddeder.
3. `.env` dosyasına demo anahtarını yazın. `config/config.yaml` içinde `mode: live` ayarlayın (`live.env: demo`
   varsayılan).
4. `python -m cappatrader check` ile bağlantıyı doğrulayın, sonra `python -m cappatrader live`.
5. Haftalarca demo sonuçlarını izleyin. Gerçek paraya geçmek için şunlar gerekir:
   - binance.com'da yalnızca **Futures** yetkili, **IP kısıtlı**, **para çekme kapalı** bir anahtar;
   - `live.env: mainnet`;
   - `live.mainnet_confirm: "GERCEK PARA ILE ISLEM YAPMAYI ONAYLIYORUM"`.

   Bu cümle yoksa bot gerçek parayla başlamaz.

Canlı moddaki korumalar:
- Her girişten sonra borsaya STOP_MARKET ve TAKE_PROFIT_MARKET emirleri girilir; bot kapansa da pozisyon korunur.
  Koruma emri reddedilirse pozisyon hemen kapatılır.
- Bakiye her girişten önce Binance'ten okunur. Risk: kasanın yüzdesi / stop mesafesi; kaldıraç yalnızca gereken
  teminatı belirler.
- Başlangıçta ve 30 saniyede bir pozisyonlar borsayla eşleştirilir. Borsada kapanmış pozisyon işlem kaydına
  geçer; botun bilmediği bir pozisyon devralınır ve korunur.
- Emir zaman aşımında, tekrar denemeden önce aynı istemci kimliğiyle emir aranır; böylece çift pozisyon açılmaz.
- Aynı hesap için ikinci bir bot başlatılamaz. `resume` ve `reset` bot çalışırken reddedilir.

### Önerilen sıra

1. `walkforward` ve `stress` gerçek veride **GEÇTİ** demeden sonuçlara güvenmeyin.
2. `paper` modunu, sonra `live` + `demo` modunu en az 4-8 hafta çalıştırın.
3. Gerçek para ancak bundan sonra ve küçük bir bakiyeyle.

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
  barlarda ATR stopları bu maliyete yakındır. Bu yüzden varsayılan zaman dilimi 1 saattir. Ensemble stratejisinde stop,
  maliyet riskin en fazla %20'si olacak şekilde genişletilir. Squeeze/VWAP stratejisinde ise sizin kuralınız geçerlidir:
  stop her zaman tam 1.5×ATR, hedef 1.1R olur. Maliyet bu dar riskin önemli bir kısmını yiyebilir.
  `stress` çıktısındaki "İdeal" ve "Gerçekçi" satırları aradaki farkı sayılarla gösterir.
  USD-M futures taker komisyonu %0.05'tir. Canlı futures ile karşılaştırırken backtest'te `costs.taker_fee: 0.0005`
  kullanılabilir. Bu ayar sonuçları güzelleştirmek için değil, gerçek ücreti yansıtmak içindir.
- **PRD KPI'ları (Sharpe > 1.5, max DD < %15) hedeftir, garanti değil.** Raporlar bunları dürüstçe ✔/✘ olarak işaretler.
  Bu hedefleri tutturmak için parametre zorlamak tam olarak kaçınılan aşırı optimizasyondur.
- 1000 USDT ile TWAP/VWAP, FIX, C++/Rust ve <50 ms tick-to-trade bir avantaj sağlamaz; Binance L3 defter vermez.
  Bunlar bilinçli olarak kapsam dışı bırakıldı (bkz. Yol haritası).
- 1.1R hedefli bir stratejide kârlılık için kazanma oranının komisyon dahil yaklaşık %50'nin belirgin
  üstünde olması gerekir; momentum dönüşündeki erken çıkışlar bu oranı belirler. Bunu walk-forward ile ölçün.
- Kağıt hesapta short işlemler 1x perpetual olarak simüle edilir (`account.allow_short: true`); canlı futures
  modunda funding ödemeleri P&L'e yansır, backtest'e yansımaz.
- Emir akışı ve duyarlılık geçmişte arşivlenmediği için yalnızca canlıda katkı verir; backtest bu iki bileşen olmadan ölçer.

## Sunucuda çalıştırma

PRD gereği borsa sunucularına yakın bir bölge seçin (Binance için ör. AWS Tokyo `ap-northeast-1`).
Binance.com bazı ülkelerden (ör. ABD) erişimi engeller; o durumda `config.yaml` içindeki `data-api.binance.vision`
satırlarını açın (salt piyasa verisi).

```bash
mkdir -p state logs && sudo chown -R 1000:1000 state logs   # konteyner 1000 numaralı kullanıcıyla yazar
docker compose up -d --build     # .env dosyasını okur, state/ ve logs/ kalıcıdır
docker compose logs -f
```

Canlı modda konteynerin komutu `live` olmalıdır: `docker compose run --rm cappatrader python -m cappatrader live`
ya da `docker-compose.yml` içinde `command: ["python", "-m", "cappatrader", "live"]`.

## Güvenlik

- Anahtarlar yalnızca `.env` / ortam değişkeninde; loglarda maskelenir; YAML'a yazılması engellenir.
- Paper mod için Binance anahtarı gerekmez. Verirseniz yalnızca **Okuma** yetkili, **IP kısıtlı** ve
  **Para Çekme kapalı** bir anahtar kullanın; bot başlarken bunu kontrol edip Telegram'a uyarır.
- Canlı mod için anahtarda yalnızca **Futures** işlem yetkisi açık olsun; **para çekme her zaman kapalı**,
  IP kısıtlaması açık. Mainnet, config'te onay cümlesi olmadan başlamaz.

## Yol haritası

- Kullanıcı veri akışı (listenKey) ile anlık dolum bildirimi (şu an 30 sn senkron + tick takibi)
- LSTM / gradient boosting modelleri, meta-etiketleme
- On-chain metrikler (Glassnode/CryptoQuant, ücretli API), daha iyi haber NLP'si
- Maker (limit) emirlerle komisyon düşürme, Telegram'dan `/durum` komutu
