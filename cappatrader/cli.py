"""Command line interface: research (backtest / walk-forward / stress) and live paper trading."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
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
    if not cfg.signals.ml.enabled:
        return None
    from cappatrader.signals.ml import walk_forward_proba

    out = {}
    for s, df in data.items():
        t = time.time()
        proba, st = walk_forward_proba(df, cfg.signals.ml)
        out[s] = proba
        print(f"ML {fmt.pair(s)}: {st.fits} eğitim, OOS isabet %{st.accuracy * 100:.1f}, eşik üstü tahmin "
              f"%{st.confident_share * 100:.1f} (isabet %{st.confident_accuracy * 100:.1f}) [{time.time() - t:.1f} sn]")
    return out


def _report_dir(cfg: Config, kind: str) -> Path:
    path = Path(cfg.runtime.report_dir) / f"{kind}_{pd.Timestamp.now(tz='UTC'):%Y%m%d_%H%M%S}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save(path: Path, summary: str, metrics: dict, trades=None, equity: pd.Series | None = None) -> None:
    (path / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    (path / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    if trades:
        pd.DataFrame([asdict(t) for t in trades]).to_csv(path / "trades.csv", index=False)
    if equity is not None and len(equity):
        equity.to_csv(path / "equity.csv")
    print(f"Rapor kaydedildi: {path}")


def cmd_backtest(args, cfg: Config) -> int:
    from cappatrader.backtest.engine import run_backtest

    data = _data(args, cfg)
    res = run_backtest(cfg, data, _ml(cfg, data))
    text = res.summary("Backtest (maliyetler dahil)")
    for ts, ev, m in res.events:
        text += f"\n[{ts}] {ev}: {m}"
    print(text)
    _save(_report_dir(cfg, "backtest"), text, res.metrics, res.trades, res.equity)
    return 0


def cmd_walkforward(args, cfg: Config) -> int:
    from cappatrader.backtest.walkforward import walk_forward

    data = _data(args, cfg)
    res = walk_forward(cfg, data, _ml(cfg, data), progress=print)
    text = res.summary()
    print(text)
    _save(_report_dir(cfg, "walkforward"), text, {"oos": res.oos_metrics, "wfe": res.wfe, "verdict": res.verdict,
          "windows": [asdict(w) for w in res.windows]}, res.oos_trades, res.oos_equity)
    return 0 if res.verdict else 2


def cmd_stress(args, cfg: Config) -> int:
    from cappatrader.backtest.stress import stress_test

    data = _data(args, cfg)
    res = stress_test(cfg, data, _ml(cfg, data))
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
    print(run_backtest(cfg, data, probas).summary("1) Backtest"))
    if not args.skip_wf:
        print(walk_forward(cfg, data, probas, progress=print).summary())
    print(stress_test(cfg, data, probas).summary())
    return 0


def cmd_pairs(args, cfg: Config) -> int:
    from cappatrader.signals.statarb import EG_CRITICAL, engle_granger

    data = _data(args, cfg)
    print(f"Engle-Granger eşbütünleşme testi (5% kritik değer {EG_CRITICAL['5%']})")
    for a, b in cfg.pairs:
        r = engle_granger(data[a]["close"], data[b]["close"], f"{a}/{b}")
        print(f"{fmt.pair(a)} ~ {fmt.pair(b)}: beta {r.beta:.3f}, ADF t {r.adf_t:.2f} "
              f"({'eşbütünleşik ✔' if r.cointegrated else 'eşbütünleşik değil ✘'}), yarı ömür "
              f"{r.half_life_bars:.0f} bar, getiri korelasyonu {r.corr:.2f}")
    return 0


def cmd_download(args, cfg: Config) -> int:
    from cappatrader.data.history import load_history

    data = load_history(cfg, args.days, refresh=True)
    for s, df in data.items():
        print(f"{s}: {len(df)} bar {df.index[0]} → {df.index[-1]}")
    return 0


# -- live ----------------------------------------------------------------------------
def cmd_paper(args, cfg: Config) -> int:
    from cappatrader.live.runner import LiveEngine

    if not cfg.secrets.has_telegram:
        print("Uyarı: TELEGRAM_TOKEN / TELEGRAM_CHAT_ID tanımlı değil, bildirimler yalnızca loga yazılacak.")
    asyncio.run(LiveEngine(cfg).run())
    return 0


def cmd_status(args, cfg: Config) -> int:
    from cappatrader.live.state import StateStore

    store = StateStore(cfg.runtime.state_db)
    acct = store.load_account(cfg.account.starting_balance)
    prices = {s: p.entry_price for s, p in acct.positions.items()}
    if acct.positions and not args.offline:
        from cappatrader.data.binance_rest import BinanceREST

        async def fetch():
            rest = BinanceREST(cfg.exchange)
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


def cmd_reset(args, cfg: Config) -> int:
    from cappatrader.live.state import StateStore

    if not args.yes:
        print("Kağıt hesabı sıfırlamak tüm işlem geçmişini siler. Onaylamak için --yes ekleyin.")
        return 1
    store = StateStore(cfg.runtime.state_db)
    acct = store.reset(args.balance or cfg.account.starting_balance)
    print(f"Kağıt hesap sıfırlandı: {fmt.usd(acct.cash)}")
    store.close()
    return 0


def cmd_resume(args, cfg: Config) -> int:
    from cappatrader.live.state import StateStore

    store = StateStore(cfg.runtime.state_db)
    acct = store.load_account(cfg.account.starting_balance)
    acct.halted, acct.halt_reason, acct.paused_until = False, "", None
    acct.peak_equity = acct.cash + sum(p.collateral for p in acct.positions.values())
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
        rest = BinanceREST(cfg.exchange, cfg.secrets)
        try:
            await rest.ping()
            df = await rest.recent_klines(cfg.symbols[0], cfg.timeframe, 5)
            print(f"✔ Binance REST erişimi: {cfg.symbols[0]} son kapanış {fmt.price(df['close'].iloc[-1])}")
            filters = await rest.exchange_filters(cfg.symbols)
            for s in cfg.symbols:
                f = filters.get(s)
                print(f"  {'✔' if f and f['status'] == 'TRADING' else '✘'} {s}: "
                      + (f"adım {f['step_size']:g}, min. emir {f['min_notional']:g} USDT" if f else "bulunamadı"))
            if cfg.secrets.has_binance:
                r = await rest.api_restrictions()
                print(f"{'✘' if r.get('enableWithdrawals') else '✔'} API anahtarı: para çekme "
                      f"{'AÇIK — hemen kapatın!' if r.get('enableWithdrawals') else 'kapalı'}, IP kısıtlaması "
                      f"{'var' if r.get('ipRestrict') else 'yok'}")
                ok = ok and not r.get("enableWithdrawals")
            else:
                print("· Binance API anahtarı yok (paper mod için gerekli değil)")
        except Exception as exc:
            ok = False
            print(f"✘ Binance erişimi başarısız: {type(exc).__name__}: {exc}")
        finally:
            await rest.close()
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
    demo = research("demo", "sentetik veriyle tüm zinciri çalıştır (internet gerekmez)")
    demo.add_argument("--skip-wf", action="store_true", help="walk-forward adımını atla (hızlı)")
    dl = sub.add_parser("download", help="geçmiş mumları indir ve önbelleğe al")
    dl.add_argument("--days", type=int, default=730)
    sub.add_parser("paper", help="canlı veriyle kağıt hesap (1000 USDT) işlemleri + Telegram")
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
    "pairs": cmd_pairs, "download": cmd_download, "paper": cmd_paper, "status": cmd_status,
    "reset": cmd_reset, "resume": cmd_resume, "check": cmd_check,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
        if getattr(args, "no_ml", False):
            cfg = cfg.with_overrides({"signals.ml.enabled": False})
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    from cappatrader.logging_setup import setup_logging

    setup_logging(cfg, args.verbose)
    return COMMANDS[args.command](args, cfg)
