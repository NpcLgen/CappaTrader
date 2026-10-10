# `/goal` için hazır metin

`/goal` bir **bitiş koşulu** ister ve 4000 karakterle sınırlıdır. İlk denemede 6 maddelik mimari metnin tamamı
(4208 karakter) yapıştırıldığı için reddedildi. O metin "ne inşa edilecek" anlatıyordu, "ne zaman bitmiş sayılır"
söylemiyordu. Doğru yaklaşım:

- **Spesifikasyon repoda dursun, hedef ona atıf yapsın.** `docs/PRD.md` ve `docs/ARCHITECTURE.md` zaten repoda.
- **Hedef, komutla doğrulanabilir olsun:** "testler geçer", "şu komut hatasız biter, şu raporu üretir".
- **Kâr şartı KOYMAYIN.** "Sharpe > 1.5 olana kadar devam et" gibi bir hedef, ajanı parametreleri geçmiş veriye
  uydurmaya iter. Bu tam olarak 5. maddede kaçınmak istediğiniz aşırı optimizasyondur. Hedef, doğrulama
  mekanizmasının doğru çalışmasını ve sonuçların dürüst raporlanmasını istemeli.
- **Kapsam sınırı koyun:** paper mod, hangi pariteler, hangi zaman dilimi, başlangıç bakiyesi.
- **Ortamı unutmayın:** bulut oturumunun ağ politikası Binance ve Telegram'ı engelliyorsa canlı adımlar orada
  doğrulanamaz. Ya ortamın izinli alan adlarına `api.binance.com`, `stream.binance.com`, `fapi.binance.com`,
  `api.telegram.org`, `api.alternative.me`, `fstream.binance.com`, `demo-fapi.binance.com`, `www.coindesk.com`, `cointelegraph.com` eklenmeli, ya da canlı doğrulama kendi sunucunuzda yapılmalı.

Eksik olanlar (ilk metinde yoktu, aşağıdaki hedefte var): ölçülebilir bitiş kriterleri, doğrulama komutları,
kapsam (paper / long-only / pariteler / 1h), güvenlik kuralı (anahtarlar yalnızca `.env`), kâr şartı yerine dürüst
raporlama şartı ve ağ ön koşulu.

## Kopyalanacak metin (~1650 karakter)

```
/goal CappaTrader (bu repo) docs/PRD.md ve docs/ARCHITECTURE.md'deki 6 katmanlı mimariyi paper trading modunda eksiksiz uygular. Aşağıdakilerin HEPSİ sağlandığında hedef tamamdır:

1. `python -m pytest -q` sıfır hata ile geçer; nedensellik testleri (test_pipeline_is_causal, test_ml_walk_forward_has_no_lookahead) dahil hiçbir test atlanmaz veya gevşetilmez.
2. `python -m cappatrader demo --bars 9000` hatasız biter ve sırasıyla backtest, walk-forward (pencere tablosu + GEÇTİ/BAŞARISIZ) ve slippage/gecikme stres testi (ideal vs gerçekçi + başabaş bps) çıktısı verir.
3. Ağ erişimi varsa: `python -m cappatrader check` Binance REST'e erişir, API anahtarında para çekme yetkisinin kapalı olduğunu raporlar ve Telegram test mesajını gönderir; `python -m cappatrader walkforward --days 730` ve `stress --days 730` gerçek veride raporu reports/ altına yazar.
4. `python -m cappatrader paper` 1000 USDT sanal bakiye ile BTC, ETH, SOL, BNB, XRP ve PAXG USDT paritelerinde 1h barlarla çalışır; her alım ve satımda Telegram'a parite, yön, fiyat, miktar, slippage, P&L, güncel bakiye ve Türkçe "neden" açıklaması gider; yeniden başlatmada pozisyonlar SQLite'tan geri yüklenir.
5. Risk kuralları kodda ve testlerde: ATR bazlı boyut, yarım Kelly, izleyen stop, %15 drawdown kill-switch, %4 günlük limit, aşırı volatilitede yeni işlem yok.
6. Hiçbir gizli anahtar git geçmişinde yok (.env gitignore'da), loglarda maskelenir.

Kısıtlar: canlı (gerçek para) emir kodu ekleme. Strateji parametrelerini metrikleri güzelleştirmek için geçmiş veriye uydurma; kâr veya Sharpe hedefi bu görevin bitiş şartı DEĞİLDİR. Sonuçlar ne çıkarsa dürüstçe raporla.
```

Hedef tamamlandıktan sonra ayrı bir `/goal` ile bir sonraki adım (ör. "Binance spot testnet'te canlı emir yürütme,
borsa tarafı stop emirleri, testnet'te 10 tam işlem döngüsü") tanımlanabilir.

## İkinci hedef: HMM + GARCH + Squeeze/VWAP + Binance (~1300 karakter)

İkinci `/goal` denemesi Pine Script'in tamamı metne eklendiği için 6202 karakterle reddedildi. Pine kodu ve PRD
artık repoda (`docs/PRD_HMM_GARCH.md`, `cappatrader/signals/squeeze.py`); hedef onlara atıf yapmalı:

```
/goal CappaTrader'da docs/PRD_HMM_GARCH.md uygulanmış sayılır, ancak ve ancak:
1. Rejim ADX yerine HMM ile (regime.method: hmm), volatilite GARCH ile (regime.vol_method: garch) belirlenir; HMM durumları yalnızca ileri filtrelemeyle hesaplanır ve nedensellik testleri geçer.
2. strategy.name: squeeze_vwap, Pine "SQZ_Breakout_Pro_NEW" momentumunu birebir hesaplar (tests/test_squeeze.py bar bar referansla karşılaştırır) ve kurallar: VWAP üstü + alıcı baskısı → LONG; VWAP altı + alıcıdan satıcıya dönüş → SHORT; stop ATR'ye göre, hedef 1.1R; hedefe varmadan karşı momentumda erken çıkış; HMM işlem yapılabilir rejimde değilse giriş yok.
3. GARCH tahmini pozisyon boyutunu küçültür ve aşırı volatilitede yeni işlemi engeller.
4. Bakiye Binance'ten okunur (live modda her işlemden önce; paper modda account.balance_source: binance ile).
5. mode: live, ccxt üzerinden Binance USD-M futures'ta emir verir; varsayılan ortam demo, mainnet yalnızca live.mainnet_confirm onay cümlesiyle; borsada STOP_MARKET/TAKE_PROFIT_MARKET koruma emirleri; yeniden başlatmada pozisyonlar borsayla eşleştirilir, sahipsiz pozisyon kalmaz.
6. python -m pytest -q sıfır hata; python -m cappatrader demo --skip-wf hatasız biter.
Kısıt: kâr/Sharpe hedefi bitiş şartı DEĞİL; parametreleri geçmişe uydurma; anahtarlar yalnızca .env'de.
```
