# Ürün Gereksinim Belgesi (PRD): Kantitatif Algoritmik Trading Botu

## 1. Belge Bilgileri
| Alan | Detay |
| :--- | :--- |
| **Ürün Adı** | AlphaBot Analiz ve Yürütme Sistemi (Kod Adı) |
| **Sürüm** | v1.0 (MVP) |
| **Durum** | Taslak / İnceleme Bekliyor |
| **Hedef Platform** | Kripto Borsaları (Binance, Bybit) ve Geleneksel Piyasalar (Interactive Brokers) |

## 2. Proje Özeti (Executive Summary)
Bu belge, 7/24 piyasa verilerini analiz ederek insan müdahalesi olmadan karar alabilen, dinamik risk yönetimi algoritmalarıyla donatılmış kantitatif bir algoritmik trading sisteminin mimari ve işlevsel gereksinimlerini tanımlar. Sistem; veri zenginleştirme, piyasa rejimi tespiti, çoklu sinyal üretimi ve risk yönetim katmanlarından oluşan modüler bir "beyin" altyapısı üzerine inşa edilecektir.

## 3. Ürün Hedefleri ve Başarı Kriterleri (KPIs)
* **Kârlılık (Alpha):** İlgili piyasa endeksinin (Benchmark) üzerinde, riske göre ayarlanmış getiri (Sharpe Ratio > 1.5) sağlamak.
* **Sermaye Koruması (Drawdown):** Maksimum portföy düşüşünü (Max Drawdown) %15'in altında tutmak.
* **Gecikme (Latency):** Piyasa verisinin alınması ile borsaya emrin iletilmesi arasındaki sürenin (Tick-to-Trade) 50 milisaniyenin altında olması.
* **Sistem Dayanıklılığı:** API rate limit (istek sınırı) aşımlarını sıfıra indirmek ve %99.99 çalışma zamanı (uptime) sağlamak.

## 4. Fonksiyonel Gereksinimler (Epic ve Özellikler)

### Epic 1: Veri İşleme ve Zenginleştirme Katmanı
Sistemin doğru analiz yapabilmesi için hatasız, kesintisiz ve derinlikli veri beslemesine ihtiyacı vardır.
* **F-1.1 WebSocket Entegrasyonu:** Fiyat dalgalanmaları ve son işlemler (tick data) saniyelik gecikmeler olmadan anlık dinlenmelidir.
* **F-1.2 Emir Defteri (Order Book) Analizi:** Sadece fiyat değil, Level-2 piyasa derinliği çekilmeli; alım/satım duvarları (liquidity pools) ve olası "spoofing" (sahte emir) aktiviteleri tespit edilmelidir.
* **F-1.3 Veri Temizleme (Data Cleansing):** Eksik mumlar (dropped packets) veya API taraflı anlık absürt iğneler (outliers) bot işlem yapmadan önce filtrelenmelidir.

### Epic 2: Piyasa Rejimi Tespit Modülü
Bot, içinde bulunduğu piyasa koşullarını (Trend, Range, Volatil) sınıflandırarak strateji değiştirmelidir.
* **F-2.1 Trend Analizi:** Fiyatın yönsel gücü ölçülmeli (örn. ADX > 25, Hareketli Ortalama eğimleri) ve piyasa "Trendde" olarak etiketlenmelidir.
* **F-2.2 Volatilite Ölçümü:** VIX (geleneksel piyasalar için) veya ATR (Average True Range) metrikleri anlık hesaplanmalı, olağanüstü piyasa koşullarında (flash crash/pump) acil durum (kill-switch) protokelleri devreye alınmalıdır.
* **F-2.3 Ortalamaya Dönüş (Mean-Reversion) Tespiti:** Piyasada net bir yön olmadığında ve varlık belirli bir bantta hareket ettiğinde sistem "Yatay" rejim statüsüne geçmelidir.

### Epic 3: Çoklu Sinyal ve Karar Motoru (Ensemble Alpha)
İşlem kararları tek bir indikatöre değil, birbiriyle onaylaşan veya birbirinden bağımsız ağırlıklandırılmış modellere dayanmalıdır.
* **F-3.1 Teknik Sinyal Üreticiler:** Momentum, RSI, MACD ve Bollinger Bantları gibi klasik indikatörlerden oluşan strateji kütüphanesi aktif edilmelidir.
* **F-3.2 İstatistiksel Sinyaller:** Varlıklar arası korelasyon sapmalarını izleyen "Pairs Trading" (İkili İşlem) makasları hesaplanmalıdır.
* **F-3.3 Güven Skoru (Confidence Score):** Üretilen sinyaller ML (Makine Öğrenmesi) modelleri veya mantıksal ağırlıklandırma süzgecinden geçerek 0 ile 100 arasında bir güven skoru oluşturmalıdır. (Örn: İşleme giriş barajı > %75).

### Epic 4: Dinamik Risk ve Pozisyon Yönetimi
Piyasa tahmin edilemez olduğunda sermayeyi koruyacak matematiksel katman.
* **F-4.1 Volatiliteye Duyarlı Lot Boyutu:** Pozisyon büyüklükleri sabit kalmamalıdır. ATR yüksekse (risk yüksekse) alınacak varlık miktarı orantılı olarak düşürülmelidir.
* **F-4.2 Kelly Kriteri Entegrasyonu:** Stratejinin geçmiş "Kazanma Oranı" ve "Kâr/Zarar Katsayısına" göre kasanın riske edilecek maksimum yüzdesi algoritmik olarak belirlenmelidir.
* **F-4.3 Dinamik Çıkış Yönetimi (Trailing Stop):** Pozisyon kâra geçtiğinde, Stop-Loss emri fiyatı geriden takip edecek şekilde (Trailing) otomatik güncellenmeli ve kâr kitlenmelidir.

### Epic 5: Backtest ve Güvenli Yürütme (Execution)
Simülasyon ve canlı işlemler arasındaki sapmalar (slippage) minimize edilmelidir.
* **F-5.1 Walk-Forward Optimizasyonu:** Kurulan stratejiler In-Sample (eğitim) ve Out-of-Sample (test) verilerinde ayrı ayrı sınanarak "Curve Fitting" (aşırı optimizasyon/ezber) engellenmelidir.
* **F-5.2 Slippage ve Komisyon Simülasyonu:** Gerçek dünyadaki likidite eksikliği ve borsa komisyonları (maker/taker fee) tüm backtest sonuçlarına net bir şekilde yansıtılmalıdır.
* **F-5.3 Asenkron Emir Yürütme (Async Execution):** Aynı anda birden fazla borsada veya paritede asenkron işlem açılıp kapatılabilmelidir.

## 5. Teknik (Non-Fonksiyonel) Gereksinimler

* **Altyapı:** Bot işlemleri, hedef borsa sunucularına coğrafi olarak en yakın bulut veri merkezlerinde (AWS, Google Cloud veya Equinix) barındırılmalıdır (Latency optimizasyonu için).
* **Dil ve Framework:** Ağır veri analizi ve makine öğrenmesi entegrasyonları için **Python** (Pandas, NumPy, Scikit-Learn); yüksek frekanslı emir iletimi kısımları için **C++** veya **Rust** tercih edilmelidir. Alternatif olarak stabilite için *Freqtrade* gibi açık kaynaklı güçlü Python iskeletleri modifiye edilebilir.
* **Veritabanı:** Tick-level (milisaniyelik) verilerin depolanması için InfluxDB veya TimescaleDB gibi zaman serisi odaklı (Time-Series) veritabanları kullanılmalıdır.
* **Güvenlik:** Borsa API anahtarları sunucuda çevresel değişken (Environment Variable) olarak şifrelenmiş tutulmalı ve "Para Çekme" (Withdrawal) yetkileri borsadan kesinlikle kapalı olmalıdır.
* **Loglama ve Hata Yönetimi:** Ağ kopmaları, borsa bakımları veya API hatalarında sistem otomatik olarak mevcut pozisyonları koruma moduna almalı ve Telegram/Discord/Slack üzerinden yöneticiye anlık hata logları iletmelidir.