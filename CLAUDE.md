# CappaTrader: notes for Claude

Quant crypto trading system, **paper mode only** (1000 USDT virtual balance, Binance market data, Telegram alerts).
Spec: `docs/PRD.md`. Data flow / module map: `docs/ARCHITECTURE.md`.

## Commands
- Tests (offline, ~20 s): `python -m pytest -q`
- Offline end-to-end: `python -m cappatrader demo --skip-wf` (add `--bars 9000` without `--skip-wf` for walk-forward)
- Real data research: `python -m cappatrader backtest|walkforward|stress --days 730`
- Live paper trading: `python -m cappatrader paper`; account: `status`, `resume`, `reset --yes`

## Rules
- Never commit secrets. Keys live only in `.env` (gitignored); `load_config` refuses a `secrets` key in YAML.
- Backtest and live must share logic: `strategy/pipeline.py` + `strategy/decision.py` + `risk/*` + `execution/costs.py`.
  Do not fork decision logic into the live runner.
- Everything in the prepared frame must stay causal. `test_pipeline_is_causal` and
  `test_ml_walk_forward_has_no_lookahead` must keep passing; never weaken them.
- Do not tune parameters to make backtests look good. Use `walkforward` and report OOS results honestly.
- User-facing text (Telegram, CLI reports, docs) is Turkish; code identifiers, comments and docstrings are English.
- No live (real money) order execution without an explicit request; the `Config.validate` mode check enforces paper.
