# CappaTrader: notes for Claude

Quant crypto trading system: paper trading on live Binance data, and live order execution on Binance USD-M
futures through ccxt (`mode: live`, demo environment by default). Telegram alerts.
Specs: `docs/PRD.md`, `docs/PRD_HMM_GARCH.md`. Data flow / module map: `docs/ARCHITECTURE.md`.
Default strategy: Squeeze Momentum + VWAP (`strategy/squeeze_decision.py`) gated by an HMM regime
(`regime/hmm.py`) with GARCH volatility sizing (`regime/garch.py`); the older ensemble is `strategy.name: ensemble`.

## Commands
- Tests (offline, ~20 s): `python -m pytest -q`
- Offline end-to-end: `python -m cappatrader demo --skip-wf` (add `--bars 9000` without `--skip-wf` for walk-forward)
- Real data research: `python -m cappatrader backtest|walkforward|stress --days 730`
- Live paper trading: `python -m cappatrader paper`; real orders: `python -m cappatrader live` (needs `mode: live`)
- Account: `status`, `balance`, `resume`, `reset --yes` (refused while an engine holds the state lock)

## Rules
- Never commit secrets. Keys live only in `.env` (gitignored); `load_config` refuses a `secrets` key in YAML.
- Backtest and live must share logic: `strategy/pipeline.py` + `strategy/factory.py` (decider, strategy risk,
  explanation, GARCH size multiplier) + `risk/*` + `execution/costs.py`.
  Do not fork decision logic into the live runner.
- Everything in the prepared frame must stay causal. `test_pipeline_is_causal`,
  `test_full_pipeline_with_hmm_garch_is_causal`, `test_ml_walk_forward_has_no_lookahead` and the HMM/GARCH
  causality tests must keep passing; never weaken them. HMM states come from forward filtering only (never
  Viterbi / predict_proba over a window that extends past the bar).
- Do not tune parameters to make backtests look good. Use `walkforward` and report OOS results honestly.
- User-facing text (Telegram, CLI reports, docs) is Turkish; code identifiers, comments and docstrings are English.
- Live execution safety is not negotiable: `live.env` defaults to `demo`; mainnet requires
  `live.mainnet_confirm == MAINNET_CONFIRM_PHRASE` (checked in `Config.validate` and again in the broker). Keep
  exchange-side protection orders, reduce-only closes, client-id idempotency and startup reconciliation intact.
  Never weaken these guards or default to mainnet.
