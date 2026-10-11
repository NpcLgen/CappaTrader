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
  kapanışında piyasa emriyle kapanır. O mumda ters yönün giriş kuralı da sağlanıyorsa pozisyon hemen ters çevrilir.
  Örneğin LONG kapanırken fiyat VWAP'ın altındaysa SHORT açılır.
- **Filtre:** HMM rejimi `TREND_UP`, `TREND_DOWN` veya `RANGE` olmalı (yüksek volatilite durumunda işlem yok) ve
  GARCH volatilitesi aşırı olmamalı.
- VWAP her gün 00:00 UTC'de sıfırlanır (TradingView'ın kripto varsayılanı; `vwap_anchor` ile değişir).
- Pine'daki sıkışma (squeeze), pivot üçgenleri ve "ardışık 3 bar" bilgileri hesaplanır ve Telegram açıklamasında
  gösterilir. `confirm_bars: 3` ile ardışık 3 bar şartı girişe de uygulanabilir.

Bütün ayarlar `config/config.yaml` dosyasında, `strategy.squeeze` altında.

## Kurulum

**Python 3.11, 3.12 veya 3.13 gerekir.** Python 3.14 henüz desteklenmiyor: `hmmlearn` paketinin 3.14 için hazır
paketi yok, pip onu derlemeye çalışır ve C++ derleyicisi olmayan bir bilgisayarda kurulum başarısız olur.

Linux / macOS:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # anahtarları .env'e yazın, asla commit etmeyin
python -m pytest -q         # internet gerektirmez
```

Windows (PowerShell). `&&` ve `source` PowerShell'de çalışmaz, komutları tek tek girin:

```powershell
py -0                                   # kurulu Python sürümleri; 3.11-3.13 listede olmalı
py -3.13 -m venv .venv                  # 3.13 yoksa python.org'dan 3.13 kurun ("Add to PATH" işaretli)
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # bir kez: betik çalıştırma izni
.\.venv\Scripts\Activate.ps1             # satırın başında (.venv) görünmeli
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
notepad .env                            # anahtarları yazıp kaydedin
python -m pytest -q
```

`py` komutu bulunamazsa Python python.org'dan kurulmamıştır (Microsoft Store sürümü yerine python.org sürümünü
kullanın). Türkçe karakterler bozuk görünürse önce `chcp 65001` çalıştırın.

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

Araştırma komutları HMM ve GARCH'ı her sembol için bir kez walk-forward eğitir. 2 yıllık 1h veride bu, sembol başına
yaklaşık 3-4 saniye sürer. `mode: live` iken araştırma verisi de USD-M futures mumlarından indirilir; botun işlem
yaptığı piyasa budur. Raporlar `reports/<tür>_<zaman>/` altına yazılır. Kağıt hesap `state/cappatrader.sqlite`,
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
  Koruma emri reddedilirse pozisyon hemen kapatılır. Kapatma da başarısız olursa her senkronda hem koruma hem
  kapatma yeniden denenir, ve Telegram'da borsada stop olmadığı açıkça yazılır.
- Yeniden başlatmada koruma emirleri yenilenirken önce yeni stop konur, sonra eski emir iptal edilir. Böylece
  pozisyon hiçbir an stopsuz kalmaz. Durdurulurken yoldaki emirlerin bitmesi 25 saniyeye kadar beklenir.
- Bakiye her girişten önce Binance'ten okunur. Risk: kasanın yüzdesi / stop mesafesi; kaldıraç yalnızca gereken
  teminatı belirler.
- Başlangıçta ve 30 saniyede bir pozisyonlar borsayla eşleştirilir. Borsada kapanmış pozisyon işlem kaydına
  geçer. Botun bilmediği bir pozisyon devralınır ve korunur. Fiyat giriş bazlı stopu veya hedefi zaten geçmişse
  yalnızca o seviye güncel fiyata göre konur, böylece hedef hiçbir zaman girişin zarar tarafına düşmez.
  Pozisyonda daha önceden duran emirler, yeni stop konduktan sonra iptal edilir.
- Kill-switch açıkken bot, tüm pozisyonlar kapanana kadar kapatmayı tekrar dener.
- Emir zaman aşımında, tekrar denemeden önce aynı istemci kimliğiyle emir aranır; böylece çift pozisyon açılmaz.
- Aynı hesap için ikinci bir bot başlatılamaz; bu, aynı `state/` klasörünü paylaşan Docker container'ları için de
  geçerlidir. `resume` ve `reset` bot çalışırken reddedilir.
- Gerçek parada (`mainnet`) API anahtarında para çekme yetkisi açıksa bot başlamaz; IP kısıtlaması yoksa uyarır.
- **Bot çalışırken futures cüzdanından para çekmeyin veya transfer yapmayın.** Bot bakiyedeki düşüşü zarar sayar ve
  kill-switch'i tetikleyip pozisyonları kapatabilir. Transferden önce botu durdurun, sonra `resume` çalıştırın;
  zirve bakiye başlangıçta Binance'ten yeniden okunur.

### Önerilen sıra

1. `walkforward` ve `stress` gerçek veride **GEÇTİ** demeden sonuçlara güvenmeyin.
2. `paper` modunu, sonra `live` + `demo` modunu en az 4-8 hafta çalıştırın.
3. Gerçek para ancak bundan sonra ve küçük bir bakiyeyle.

## Telegram mesajı örneği

Kodun ürettiği gerçek çıktı. Varsayılan Squeeze/VWAP stratejisi ve kağıt hesap kullanıldı; fiyatlar sentetik
veriden geldiği için gerçekçi değil:

```
🟢 ALIŞ (LONG) — ETH/USDT · KAĞIT HESAP
Fiyat: 52,528.22 (sinyal 52,494.72 · slippage 6.4 bps · gecikme 0 ms)
Miktar: 0.00571 ETH ≈ 299.94 USDT · komisyon 0.30 USDT
Stop: 51,841.99 (-1.31%) · risk 3.92 USDT (%0.39 kasa) · hedef 53,283.07
Rejim: Yatay bant · HMM olasılığı %100 · strateji: Squeeze momentum + VWAP

Neden alındı?
• Rejim: Yatay bant · HMM olasılığı %100 · volatilite normal · GARCH sonraki bar σ 0.77% (uzun dönem 2.26%, 0.34×) → boyut çarpanı ×1.00
• ▲ Fiyat günlük VWAP'ın üstünde: kapanış 52,494.72 · VWAP 51,844.06 (+1.26%)
• ▲ Squeeze momentum -173.5 (sıfırın altında), 1 bardır yükseliyor — alıcı baskısı
• Squeeze yok: Bollinger bantları Keltner kanalının dışında
• ▲ Boğa pivotu: momentum alt eşiğin (0) altında yukarı döndü
• Stop 1.5×ATR, hedef 1.1R (sinyal kapanışına göre ≈ stop 51,808.49, hedef 53,249.57) · hedeften önce momentum satıcıya dönerse erken çıkış

Pozisyon boyutu
• Kelly için yetersiz geçmiş (0/30 işlem) → sabit %1.00 risk
• Stop mesafesi 1.5×ATR = 1.31% (ATR yüksekse pozisyon otomatik küçülür)
• Pozisyon tavanı (%30 kasa / kullanılabilir nakit) → gerçek risk 3.92 USDT

💼 Güncel bakiye: 999.51 USDT (-0.05% başlangıç 1,000.00 USDT)
Nakit 699.76 USDT · açık pozisyon 1
```

```
🔴 SATIŞ — pozisyon kapandı — ETH/USDT · KAĞIT HESAP
Giriş 52,528.22 → Çıkış 53,277.74 (slippage 1.0 bps)
Miktar: 0.00571 ETH ≈ 304.22 USDT
✅ P&L: +3.68 USDT (+1.23% · +0.94R) · komisyon 0.60 USDT dahil

Neden satıldı?
• Hedef kâr (take-profit) seviyesine ulaşıldı

Giriş gerekçesi (özet)
• Rejim: Yatay bant · HMM olasılığı %100 · volatilite normal · GARCH sonraki bar σ 0.77% (uzun dönem 2.26%, 0.34×) → boyut çarpanı ×1.00
• ▲ Fiyat günlük VWAP'ın üstünde: kapanış 52,494.72 · VWAP 51,844.06 (+1.26%)
• ▲ Squeeze momentum -173.5 (sıfırın altında), 1 bardır yükseliyor — alıcı baskısı

💼 Güncel bakiye: 1,003.68 USDT (+0.37% başlangıç 1,000.00 USDT)
Nakit 1,003.68 USDT · açık pozisyon 0
Realize P&L toplam +3.68 USDT · 1 işlem · kazanma %100
```

Hedefe varmadan momentum dönerse çıkış gerekçesi şöyle görünür:
`• Momentum satıcıya döndü — hedefe ulaşmadan erken çıkış (momentum -29.06 < önceki -14.7)`.
Fiyat VWAP'ın altındaysa aynı dönüş SHORT kuralını da sağlar. Bu durumda pozisyon kapatılır ve hemen SHORT açılır
(stop-and-reverse); backtest de aynı şekilde çalışır. Canlı modda başlıkta `CANLI · DEMO (sanal para)` veya
`CANLI · GERÇEK PARA` yazar ve bakiye Binance'ten okunur.

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
