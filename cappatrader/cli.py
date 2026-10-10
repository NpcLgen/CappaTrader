"""Command line interface: research (backtest / walk-forward / stress) and live paper trading."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from cappatrader import fmt
from cappatrader.config import Config, load_config

log = logging.getLogger("cappatrader")


# -- research helpers ----------------------------------------------------------------
def _data(args, cfg: Config) -> dict[str, pd.DataFrame]:
    if getattr(args, "synthetic", False):
        from cappatrader.data.synthetic import generate_universe

        print(f"Sentetik veri: {args.bars} bar × {len(cfg.symbols)} parite ({cfg.timeframe}) — yalnızca mekanizma testi")
        return generate_universe(cfg.symbols, n_bars=args.bars, timeframe=cfg.timeframe, seed=args.seed)
    from cappatrader.data.history import load_history

    return load_history(cfg, args.days, refresh=getattr(args, "refresh", False))


def _ml(cfg: Config, data: dict[str, pd.DataFrame]) -> dict[str, pd.Series] | None:
    if not cfg.signals.ml.enabled or cfg.strategy.name != "ensemble":
        return None  # the ML direction model only votes in the ensemble strategy
    from cappatrader.signals.ml import walk_forward_proba

    out = {}
    for s, df in data.items():
        t = time.time()
        proba, st = walk_forward_proba(df, cfg.signals.ml)
        out[s] = proba
        if st.fits == 0:
            print(f"ML {fmt.pair(s)}: yetersiz veri, model eğitilmedi (ML oyu bu paritede kullanılmıyor)")
            continue
        print(f"ML {fmt.pair(s)}: {st.fits} eğitim, OOS isabet %{st.accuracy * 100:.1f}, eşik üstü tahmin "
              f"%{st.confident_share * 100:.1f} (isabet %{st.confident_accuracy * 100:.1f}) [{time.time() - t:.1f} sn]")
    return out


def _layers(cfg: Config, data: dict[str, pd.DataFrame]):
    """Walk-forward HMM / GARCH layers, computed once per symbol and shared by every run."""
    from cappatrader.strategy.pipeline import compute_layers

    if cfg.regime.method != "hmm" and cfg.regime.vol_method != "garch":
        return None
    out = {}
    for s, df in data.items():
        t = time.time()
        layers = compute_layers(df, cfg)
        out[s] = layers
        parts = []
        h = layers.get("hmm")
        if h is not None:
            counts = h["hmm_label"].value_counts(normalize=True)
            parts.append("HMM " + ", ".join(f"{k} %{v * 100:.0f}" for k, v in counts.items()))
        g = layers.get("garch")
        if g is not None and g["garch_ratio"].notna().any():
            parts.append(f"GARCH oran ort. {g['garch_ratio'].mean():.2f}")
        print(f"Rejim {fmt.pair(s)}: {' · '.join(parts)} [{time.time() - t:.1f} sn]")
    return out


def _report_dir(cfg: Config, kind: str) -> Path:
    path = Path(cfg.runtime.report_dir) / f"{kind}_{pd.Timestamp.now(tz='UTC'):%Y%m%d_%H%M%S}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_safe(obj):
    """Strict JSON: NaN / ±Infinity become null (Python's json would write bare NaN/Infinity tokens)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _save(path: Path, summary: str, metrics: dict, trades=None, equity: pd.Series | None = None) -> None:
    (path / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    safe = _json_safe(json.loads(json.dumps(metrics, default=str)))
    (path / "metrics.json").write_text(json.dumps(safe, indent=2, allow_nan=False), encoding="utf-8")
    if trades:
        pd.DataFrame([asdict(t) for t in trades]).to_csv(path / "trades.csv", index=False)
    if equity is not None and len(equity):
        equity.to_csv(path / "equity.csv")
    print(f"Rapor kaydedildi: {path}")


def cmd_backtest(args, cfg: Config) -> int:
    from cappatrader.backtest.engine import run_backtest

    data = _data(args, cfg)
    res = run_backtest(cfg, data, _ml(cfg, data), layers=_layers(cfg, data))
    text = res.summary("Backtest (maliyetler dahil)")
    for ts, ev, m in res.events:
        text += f"\n[{ts}] {ev}: {m}"
    print(text)
    _save(_report_dir(cfg, "backtest"), text, res.metrics, res.trades, res.equity)
    return 0


def cmd_walkforward(args, cfg: Config) -> int:
    from cappatrader.backtest.walkforward import walk_forward

    data = _data(args, cfg)
    res = walk_forward(cfg, data, _ml(cfg, data), progress=print, layers=_layers(cfg, data))
    text = res.summary()
    print(text)
    _save(_report_dir(cfg, "walkforward"), text, {"oos": res.oos_metrics, "wfe": res.wfe, "verdict": res.verdict,
          "windows": [asdict(w) for w in res.windows]}, res.oos_trades, res.oos_equity)
    return 0 if res.verdict else 2


def cmd_stress(args, cfg: Config) -> int:
    from cappatrader.backtest.stress import stress_test

    data = _data(args, cfg)
    res = stress_test(cfg, data, _ml(cfg, data), layers=_layers(cfg, data))
    text = res.summary()
    print(text)
    _save(_report_dir(cfg, "stress"), text, asdict(res))
    return 0


def cmd_demo(args, cfg: Config) -> int:
    args.synthetic = True
    print("═" * 70 + "\nDEMO — sentetik veriyle uçtan uca analiz zinciri\n" + "═" * 70)
    from cappatrader.backtest.engine import run_backtest
    from cappatrader.backtest.stress import stress_test
    from cappatrader.backtest.walkforward import walk_forward

    data = _data(args, cfg)
    probas = _ml(cfg, data)
    layers = _layers(cfg, data)
    print(run_backtest(cfg, data, probas, layers=layers).summary("1) Backtest"))
    need = cfg.walkforward.is_bars + cfg.walkforward.oos_bars
    if not args.skip_wf and args.bars < need:
        print(f"(Walk-forward atlandı: en az {need} bar gerekli, --bars {args.bars})")
    elif not args.skip_wf:
        print(walk_forward(cfg, data, probas, progress=print, layers=layers).summary())
    print(stress_test(cfg, data, probas, layers=layers).summary())
    return 0


def cmd_pairs(args, cfg: Config) -> int:
    from cappatrader.signals.statarb import EG_CRITICAL, engle_granger

    if not cfg.pairs:
        print("Yapılandırmada pairs listesi boş; test edilecek çift yok.")
        return 0
    data = _data(args, cfg)
    print(f"Engle-Granger eşbütünleşme testi (5% kritik değer {EG_CRITICAL['5%']})")
    for a, b in cfg.pairs:
        if min(len(data[a]), len(data[b])) < 50:
            print(f"{fmt.pair(a)} ~ {fmt.pair(b)}: yetersiz veri (en az 50 bar gerekli)")
            continue
        r = engle_granger(data[a]["close"], data[b]["close"], f"{a}/{b}")
        print(f"{fmt.pair(a)} ~ {fmt.pair(b)}: beta {r.beta:.3f}, ADF t {r.adf_t:.2f} "
              f"({'eşbütünleşik ✔' if r.cointegrated else 'eşbütünleşik değil ✘'}), yarı ömür "
              f"{r.half_life_bars:.0f} bar, getiri korelasyonu {r.corr:.2f}")
    return 0


def cmd_download(args, cfg: Config) -> int:
    from cappatrader.data.history import load_history

    if args.days <= 0:
        print("✘ --days pozitif olmalı.")
        return 1
    data = load_history(cfg, args.days, refresh=True)
    ok = True
    for s, df in data.items():
        if df.empty:
            ok = False
            print(f"✘ {s}: veri gelmedi (parite bu piyasada yok ya da çok yeni)")
        else:
            print(f"{s}: {len(df)} bar {df.index[0]} → {df.index[-1]}")
    return 0 if ok else 1


# -- live ----------------------------------------------------------------------------
def _run_engine(cfg: Config) -> int:
    from cappatrader.live.runner import LiveEngine

    if not cfg.secrets.has_telegram:
        print("Uyarı: TELEGRAM_TOKEN / TELEGRAM_CHAT_ID tanımlı değil, bildirimler yalnızca loga yazılacak.")
    engine = LiveEngine(cfg)
    try:
        asyncio.run(engine.run())
    except Exception as exc:  # startup failure (lock held, exchange unreachable, bad keys...)
        print(f"✘ Bot başlatılamadı: {_explain_error(exc)}", file=sys.stderr)
        return 1
    return 1 if engine.crashed else 0


def cmd_paper(args, cfg: Config) -> int:
    if cfg.mode == "live":
        print("config: mode 'live'. Kağıt hesap için mode: paper yapın; canlı işlem için: python -m cappatrader live")
        return 1
    return _run_engine(cfg)


def cmd_live(args, cfg: Config) -> int:
    if cfg.mode != "live":
        print("Canlı işlem kapalı: config/config.yaml içinde mode: live ayarlanmalı (önce live.env: demo ile deneyin).")
        return 1
    if not cfg.secrets.has_binance:
        print("✘ BINANCE_API_KEY / BINANCE_SECRET_KEY tanımlı değil (.env).")
        return 1
    sq = cfg.strategy.squeeze
    print("═" * 70)
    print(f"CANLI İŞLEM — {cfg.mode_label} · Binance USD-M futures · kaldıraç {cfg.live.leverage}x "
          f"({cfg.live.margin_mode})")
    print(f"Pariteler: {', '.join(cfg.symbols)} · {cfg.timeframe} · strateji {cfg.strategy.name}"
          + (f" (stop {sq.atr_stop_mult:g}×ATR, hedef {sq.take_profit_r:g}R)" if cfg.strategy.name == "squeeze_vwap" else ""))
    print(f"Durum dosyası: {cfg.state_db}")
    print("═" * 70)
    if cfg.live.env == "mainnet" and not args.yes:
        print("GERÇEK PARA ile işlem başlayacak. İptal için 10 sn içinde Ctrl+C...")
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            print("İptal edildi.")
            return 1
    return _run_engine(cfg)


async def _print_balances(cfg: Config) -> int:
    """Read-only: USD-M futures and spot USDT balances of the account behind the API key."""
    from cappatrader.execution.binance_futures import BinanceAccountReader

    reader = BinanceAccountReader(cfg)
    env = "DEMO (demo.binance.com, sanal para)" if reader.env == "demo" else "GERÇEK HESAP"
    print(f"Binance bakiyesi — {env} · salt okuma")
    failed = 0
    try:
        try:
            b = await reader.balance()
            print(f"  USD-M futures: cüzdan {fmt.usd(b.wallet)} · kullanılabilir {fmt.usd(b.available)} · "
                  f"gerçekleşmemiş K/Z {fmt.usd(b.unrealized, signed=True)} · özsermaye {fmt.usd(b.equity)}")
        except Exception as exc:
            failed += 1
            print(f"  USD-M futures: okunamadı ({exc})")
        if reader.env == "mainnet":  # demo trading has no spot wallet behind the futures demo keys
            try:
                print(f"  Spot USDT    : {fmt.usd(await reader.spot_usdt_balance())}")
            except Exception as exc:
                failed += 1
                print(f"  Spot USDT    : okunamadı ({exc})")
    finally:
        await reader.close()
    src = cfg.account.binance_wallet
    print(f"Kağıt hesap 'balance_source: binance' iken başlangıç bakiyesi olarak {src} cüzdanı kullanılır.")
    return 1 if failed == (2 if reader.env == "mainnet" else 1) else 0


def cmd_balance(args, cfg: Config) -> int:
    if not cfg.secrets.has_binance:
        print("✘ BINANCE_API_KEY / BINANCE_SECRET_KEY tanımlı değil (.env).")
        return 1
    try:
        return asyncio.run(_print_balances(cfg))
    except Exception as exc:
        print(f"✘ Bakiye okunamadı: {type(exc).__name__}: {exc}")
        return 1


def cmd_status(args, cfg: Config) -> int:
    from cappatrader.live.state import StateStore

    store = StateStore(cfg.state_db)
    acct = store.load_account(cfg.account.starting_balance)
    print(f"[{cfg.mode_label}] {cfg.state_db}")
    prices = {s: p.entry_price for s, p in acct.positions.items()}
    if acct.positions and not args.offline:
        from cappatrader.data.binance_rest import BinanceREST

        async def fetch():
            rest = BinanceREST(cfg.exchange, market=cfg.data_market)
            try:
                for s in acct.positions:
                    prices[s] = await rest.ticker_price(s)
            finally:
                await rest.close()

        try:
            asyncio.run(fetch())
        except Exception as exc:
            print(f"(Güncel fiyat alınamadı: {exc}; giriş fiyatları kullanılıyor)")
    eq = acct.equity(prices)
    print(f"Kağıt hesap: başlangıç {fmt.usd(acct.starting_balance)} · bakiye {fmt.usd(eq)} "
          f"({fmt.pct(eq / acct.starting_balance - 1, signed=True)}) · nakit {fmt.usd(acct.cash)}")
    print(f"Realize P&L {fmt.usd(acct.realized_pnl, signed=True)} · {len(acct.trades)} işlem · toplam komisyon "
          f"{acct.total_fees:.2f} USDT")
    if acct.halted:
        print(f"KILL-SWITCH AKTİF: {acct.halt_reason} (devam için: python -m cappatrader resume)")
    for s, p in acct.positions.items():
        print(f"  Açık {fmt.pair(s)} {fmt.qty(p.qty)} @ {fmt.price(p.entry_price)} | stop {fmt.price(p.stop_price)} | "
              f"açık P&L {fmt.usd(p.unrealized(prices[s]), signed=True)}")
    for t in acct.trades[-10:]:
        print(f"  {t.exit_time[:16]} {fmt.pair(t.symbol)} {fmt.usd(t.pnl, signed=True)} ({t.r_multiple:+.2f}R) — {t.exit_reason}")
    store.close()
    return 0


def _engine_running(cfg: Config) -> bool:
    from cappatrader.live.lock import running_pid

    pid = running_pid(cfg.state_db)
    if pid is not None:
        print(f"✘ Bot şu an çalışıyor (PID {pid}). Değişiklik bir sonraki kayıtta geri yazılır; önce botu durdurun "
              "(Ctrl+C veya docker compose stop), sonra tekrar deneyin.")
        return True
    return False


def cmd_reset(args, cfg: Config) -> int:
    from cappatrader.live.state import StateStore

    if not args.yes:
        print("Hesap kaydını sıfırlamak tüm işlem geçmişini siler. Onaylamak için --yes ekleyin.")
        return 1
    if _engine_running(cfg):
        return 1
    if args.balance is not None and not (args.balance > 0 and args.balance < float("inf")):
        print("✘ --balance pozitif bir sayı olmalı.")
        return 1
    if cfg.mode == "live":
        print("Not: canlı modda bakiye her zaman Binance'ten okunur; bu komut yalnızca yerel kaydı siler.")
    store = StateStore(cfg.state_db)
    acct = store.reset(args.balance or cfg.account.starting_balance)
    print(f"Kağıt hesap sıfırlandı: {fmt.usd(acct.cash)}")
    store.close()
    return 0


def cmd_resume(args, cfg: Config) -> int:
    from cappatrader.live.state import StateStore

    if _engine_running(cfg):
        return 1
    store = StateStore(cfg.state_db)
    acct = store.load_account(cfg.account.starting_balance)
    acct.halted, acct.halt_reason, acct.paused_until = False, "", None
    acct.peak_equity = acct.cash + sum(p.collateral for p in acct.positions.values())
    # restart the daily window too, otherwise today's loss re-triggers the pause immediately
    acct.day, acct.day_start_equity = None, acct.peak_equity
    store.save_account(acct)
    print("Kill-switch / günlük duraklatma kaldırıldı; zirve bakiye mevcut değere sıfırlandı.")
    store.close()
    return 0


def cmd_check(args, cfg: Config) -> int:
    from cappatrader.data.binance_rest import BinanceREST
    from cappatrader.notify.telegram import TelegramNotifier

    async def run() -> int:
        ok = True
        print(f"Mod: {cfg.mode} · pariteler: {', '.join(cfg.symbols)} · zaman dilimi {cfg.timeframe}")
        rest = BinanceREST(cfg.exchange, cfg.secrets, market=cfg.data_market)
        try:
            await rest.ping()
            df = await rest.recent_klines(cfg.symbols[0], cfg.timeframe, 5)
            print(f"✔ Binance REST erişimi: {cfg.symbols[0]} son kapanış {fmt.price(df['close'].iloc[-1])}")
            filters = await rest.exchange_filters(cfg.symbols)
            for s in cfg.symbols:
                f = filters.get(s)
                good = bool(f and f["status"] == "TRADING")
                ok = ok and good
                print(f"  {'✔' if good else '✘'} {s}: "
                      + (f"adım {f['step_size']:g}, min. emir {f['min_notional']:g} USDT" if f else "bulunamadı"))
            if cfg.secrets.has_binance and not (cfg.mode == "live" and cfg.live.env == "demo"):
                # demo.binance.com keys have no /sapi endpoint; restrictions are checked only for real keys
                r = await rest.api_restrictions()
                print(f"{'✘' if r.get('enableWithdrawals') else '✔'} API anahtarı: para çekme "
                      f"{'AÇIK — hemen kapatın!' if r.get('enableWithdrawals') else 'kapalı'}, IP kısıtlaması "
                      f"{'var' if r.get('ipRestrict') else 'yok'}")
                ok = ok and not r.get("enableWithdrawals")
            elif not cfg.secrets.has_binance:
                print("· Binance API anahtarı yok (paper mod için gerekli değil)" if cfg.mode != "live"
                      else "✘ Binance API anahtarı yok (live mod için gerekli)")
                ok = ok and cfg.mode != "live"
        except Exception as exc:
            ok = False
            print(f"✘ Binance erişimi başarısız: {type(exc).__name__}: {exc}")
        finally:
            await rest.close()
        if cfg.mode == "live" and cfg.secrets.has_binance:
            from cappatrader.execution.binance_futures import BinanceAccountReader

            reader = BinanceAccountReader(cfg)
            try:
                b = await reader.balance()
                print(f"✔ Futures hesabı ({cfg.mode_label}): özsermaye {fmt.usd(b.equity)}, "
                      f"kullanılabilir {fmt.usd(b.available)}")
            except Exception as exc:
                ok = False
                print(f"✘ Futures hesabına erişilemedi ({cfg.mode_label}): {exc}")
            finally:
                await reader.close()
        if args.no_telegram:
            print("· Telegram testi atlandı")
        elif cfg.secrets.has_telegram:
            tg = TelegramNotifier(cfg.secrets.telegram_token, cfg.secrets.telegram_chat_id)
            sent = await tg.send_now("✅ CappaTrader bağlantı testi: Telegram bildirimleri çalışıyor.")
            await tg.stop()
            print(f"{'✔' if sent else '✘'} Telegram test mesajı {'gönderildi' if sent else 'gönderilemedi'}")
            ok = ok and sent
        else:
            print("✘ TELEGRAM_TOKEN / TELEGRAM_CHAT_ID tanımlı değil")
            ok = False
        return 0 if ok else 1

    return asyncio.run(run())


# -- entry point -----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cappatrader", description="CappaTrader kantitatif trading sistemi")
    p.add_argument("--config", help="YAML yapılandırma dosyası (varsayılan: config/config.yaml)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def research(name: str, help_: str):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--days", type=int, default=730, help="geçmiş veri gün sayısı (Binance)")
        sp.add_argument("--refresh", action="store_true", help="önbelleği yok say, yeniden indir")
        sp.add_argument("--synthetic", action="store_true", help="internet olmadan sentetik veri kullan")
        sp.add_argument("--bars", type=int, default=9000, help="sentetik bar sayısı")
        sp.add_argument("--seed", type=int, default=7)
        sp.add_argument("--no-ml", action="store_true", help="ML bileşenini kapat")
        return sp

    research("backtest", "maliyet/slippage dahil backtest")
    research("walkforward", "in-sample optimizasyon + out-of-sample doğrulama")
    research("stress", "slippage ve gecikme stres testi")
    research("pairs", "pairs trading eşbütünleşme raporu")
    demo = sub.add_parser("demo", help="sentetik veriyle tüm zinciri çalıştır (internet gerekmez)")
    demo.add_argument("--bars", type=int, default=9000, help="sentetik bar sayısı")
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--no-ml", action="store_true", help="ML bileşenini kapat")
    demo.add_argument("--skip-wf", action="store_true", help="walk-forward adımını atla (hızlı)")
    dl = sub.add_parser("download", help="geçmiş mumları indir ve önbelleğe al")
    dl.add_argument("--days", type=int, default=730)
    sub.add_parser("paper", help="canlı veriyle kağıt hesap (1000 USDT) işlemleri + Telegram")
    lv = sub.add_parser("live", help="Binance USD-M futures'ta gerçek emirle işlem (mode: live, önce demo)")
    lv.add_argument("--yes", action="store_true", help="mainnet için 10 sn bekleme onayını atla")
    sub.add_parser("balance", help="Binance futures ve spot bakiyesini göster (salt okuma)")
    st = sub.add_parser("status", help="kağıt hesap durumu")
    st.add_argument("--offline", action="store_true", help="güncel fiyat çekme")
    rs = sub.add_parser("reset", help="kağıt hesabı sıfırla")
    rs.add_argument("--yes", action="store_true")
    rs.add_argument("--balance", type=float)
    sub.add_parser("resume", help="kill-switch sonrası işlemlere manuel devam")
    ck = sub.add_parser("check", help="Binance/Telegram bağlantı ve API yetki kontrolü")
    ck.add_argument("--no-telegram", action="store_true")
    return p


COMMANDS = {
    "backtest": cmd_backtest, "walkforward": cmd_walkforward, "stress": cmd_stress, "demo": cmd_demo,
    "pairs": cmd_pairs, "download": cmd_download, "paper": cmd_paper, "live": cmd_live, "balance": cmd_balance,
    "status": cmd_status,
    "reset": cmd_reset, "resume": cmd_resume, "check": cmd_check,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
        if getattr(args, "no_ml", False):
            cfg = cfg.with_overrides({"signals.ml.enabled": False})
    except FileNotFoundError as exc:
        print(f"✘ Yapılandırma dosyası bulunamadı: {exc.filename}", file=sys.stderr)
        return 1
    except (ValueError, TypeError) as exc:
        print(f"✘ {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # e.g. yaml.YAMLError
        print(f"✘ Yapılandırma okunamadı: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    from cappatrader.logging_setup import setup_logging

    setup_logging(cfg, args.verbose)
    try:
        return COMMANDS[args.command](args, cfg)
    except KeyboardInterrupt:
        print("İptal edildi.")
        return 130
    except Exception as exc:  # one clear Turkish line instead of a traceback; details go to the log file
        log.debug("Komut hatası", exc_info=True)
        print(f"✘ {args.command} başarısız: {_explain_error(exc)}", file=sys.stderr)
        print("  (Ayrıntı: logs/cappatrader.log; tam hata izi için -v ile çalıştırın)", file=sys.stderr)
        if args.verbose:
            raise
        return 1


def _explain_error(exc: Exception) -> str:
    from cappatrader.data.binance_rest import BinanceAPIError

    text = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, BinanceAPIError):
        if exc.status == 451:
            return text + " — Binance bu bölgeden erişimi kısıtlıyor; config.yaml'daki data-api.binance.vision satırlarını açın"
        if exc.status == 400 and "Invalid symbol" in str(exc):
            return text + " — config'teki parite adlarından biri bu piyasada yok"
    try:
        import aiohttp

        if isinstance(exc, (aiohttp.ClientError, asyncio.TimeoutError, OSError)):
            return text + " — Binance'e bağlanılamadı (internet / güvenlik duvarı / proxy)"
    except ImportError:
        pass
    return text
