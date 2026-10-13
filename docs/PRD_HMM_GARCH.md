# Ürün Gereksinim Belgesi (PRD)

**Proje Adı:** HMM & GARCH Tabanlı Akıllı Alım-Satım Botu
**Belge Sürümü:** 1.0
**Durum:** Taslak / Geliştirme Bekliyor

## 1. Proje Özeti
Bu proje, geleneksel teknik indikatörlerin (örn. ADX) ötesine geçerek piyasa durumunu makine öğrenmesi algoritmalarıyla (Hidden Markov Model) kendi kendine öğrenen, volatiliteyi istatistiksel modellerle (GARCH) tahmin eden ve Pine Script tabanlı Momentum/VWAP stratejisini otomatize eden bir algoritmik ticaret botunun geliştirilmesini amaçlamaktadır. Sistem, işlemleri doğrudan Binance borsası üzerinden gerçekleştirecek ve hesap bakiye yönetimini dinamik olarak yapacaktır.

## 2. Temel Hedefler
- Statik trend belirleyiciler (ADX vb.) yerine veriden öğrenen dinamik piyasa rejimi tespiti (HMM) sağlamak.
- GARCH modeli ile gelecekteki volatiliteyi tahmin ederek risk ve lot yönetimini optimize etmek.
- Squeeze Momentum ve VWAP indikatörlerini kullanarak yüksek doğruluklu giriş-çıkış (Long/Short) sinyalleri üretmek.
- İşlemleri, bakiye kontrollerini ve risk yönetimini (ATR tabanlı Stop-Loss) Binance API üzerinden tam otopilot şeklinde yönetmek.

## 3. Sistem Mimarisi ve Teknoloji Yığını
- **Programlama Dili:** Python 3.9+ (TradingView altyapısı bu karmaşıklıktaki ML modellerini desteklemediği için sistem Python tabanlı bir sunucu mimarisinde çalışacaktır).
- **Borsa Entegrasyonu:** ccxt kütüphanesi (Binance Futures API).
- **Makine Öğrenmesi (HMM):** hmmlearn kütüphanesi.
- **Volatilite Modelleme (GARCH):** arch kütüphanesi.
- **Teknik Analiz ve Veri İşleme:** pandas, numpy, pandas_ta.

## 4. Fonksiyonel Gereksinimler

### 4.1. Veri ve Borsa Entegrasyonu
- **Bakiye Sorgulama:** Sistem, her işlem öncesi Binance API üzerinden anlık USDT/M-Coin bakiyesini çekecek ve risk hesaplamasını bu bakiye üzerinden yapacaktır.
- **Canlı Veri Akışı:** İlgili parite (örn. BTC/USDT) için OHLCV (Açılış, Yüksek, Düşük, Kapanış, Hacim) verileri belirlenen zaman diliminde (örn. 15m, 1h) sürekli çekilecektir.

### 4.2. Yapay Zeka ve İstatistiksel Modeller
- **Hidden Markov Model (HMM) Modülü:** Geçmiş fiyat hareketleri ve getiri (return) serileri üzerinden piyasanın mevcut rejimini (Trend, Yatay, Yüksek Volatilite vb.) tespit edecektir. ADX indikatörünün yerini alarak, sistemin sadece uygun piyasa koşullarında işleme girmesini sağlayacaktır.
- **GARCH Modülü:** Zaman serisi analizi ile bir sonraki periyodun volatilitesini tahmin edecektir. Bu tahmin, kaldıraç oranının belirlenmesinde veya ani piyasa şoklarından korunmada filtre olarak kullanılacaktır.

### 4.3. Strateji ve Sinyal Mantığı (Pine Script Adaptasyonu)
Sistem, referans alınan Squeeze Momentum ve VWAP indikatörlerinin Python karşılıklarını hesaplayacaktır. Sinyal üretim mantığı aşağıdaki kurallara göre işletilecektir:

- **Momentum Tanımları:**
  - **Alıcı Baskısı:** Momentum değerinin sıfırın altında veya üstünde olması fark etmeksizin, mevcut barın momentum değerinin bir önceki bardan yüksek olması (yönün yukarı ivmelenmesi).
  - **Satıcı Baskısı:** Mevcut barın momentum değerinin bir önceki bardan düşük olması (yönün aşağı ivmelenmesi).
- **Giriş Kuralları:**
  - **LONG Pozisyon:** Fiyat VWAP seviyesinin üzerinde ise ve Momentum osilatöründe alıcılar ağırlıktaysa (momentum ivmesi yukarı).
  - **SHORT Pozisyon:** Fiyat VWAP seviyesinin altında ise ve Momentum alıcılardan satıcılara dönmüşse (momentum ivmesi aşağı).
- **Çıkış ve Risk Yönetimi Kuralları:**
  - **Stop-Loss (Zarar Kes):** Pozisyon açıldığı andaki ATR (Average True Range) değerine göre dinamik olarak belirlenecektir.
  - **Take Profit / Erken Kapanış (Kar Al):**
    - Hedeflenen Risk/Ödül oranı 1.1 olarak ayarlanacaktır (Kâr hedefi, alınan ATR riskinin 1.1 katı olacaktır).
    - **Dinamik Erken Çıkış (LONG):** Fiyat hedefe ulaşmadan dönerse ve momentumda satıcı ağırlıklı baskı başlarsa pozisyon anında piyasa emriyle kapatılacaktır.
    - **Dinamik Erken Çıkış (SHORT):** Fiyat hedefe ulaşmadan dönerse ve momentumda alıcı baskını geri gelirse pozisyon anında piyasa emriyle kapatılacaktır.

## 5. Operasyonel Süreç Akışı (Workflow)
1. **Döngü Başlangıcı:** Sistem her yeni mum kapanışında tetiklenir.
2. **Veri Toplama:** Binance API üzerinden güncel OHLCV verisi ve hesap bakiyesi çekilir.
3. **Hesaplamalar:** VWAP, Momentum (Lineer Regresyon bazlı), ATR, HMM (Rejim Tespiti) ve GARCH (Volatilite) değerleri hesaplanır.
4. **Koşul Denetimi:**
  - HMM modeli piyasanın işlem yapılabilir bir rejimde olduğunu onaylarsa strateji koşullarına (VWAP ve Momentum) bakılır.
5. **Karar ve İletim:**
  - Alım/Satım koşulu oluşmuşsa, GARCH ve Bakiye verisiyle lot büyüklüğü hesaplanır.
  - Binance'e Market/Limit emri iletilir.
  - Eş zamanlı olarak ATR tabanlı Stop Loss ve 1.1 R/R tabanlı Take Profit emirleri sisteme girilir veya bot tarafından lokal olarak takip edilir.
6. **Pozisyon Takibi:** Açık pozisyon varken, momentum dönüşleri anlık izlenir; ters momentum oluşursa pozisyon hedefleri beklemeden erken kapatılır.

## 6. Güvenlik ve Hata Yönetimi
- **API Güvenliği:** Binance API anahtarları çevre değişkenleri (Environment Variables) kullanılarak şifrelenmiş şekilde tutulacaktır. IP kısıtlaması aktif edilecektir.
- **Bağlantı Kopmaları:** API erişim hatalarında veya Rate Limit aşımlarında sistem exponential backoff (kademeli bekleme) yöntemiyle yeniden bağlanmayı deneyecektir.
- **Açık Pozisyon Koruması:** Sunucu çökmesi veya yeniden başlatılması durumunda, açık pozisyonlar Binance üzerinden sorgulanıp senkronize edilecek ve sahipsiz pozisyon bırakılmayacaktır.