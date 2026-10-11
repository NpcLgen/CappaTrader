"""User-facing configuration checks, CLI guard rails and report wording."""

import pytest

from cappatrader import cli
from cappatrader.backtest.stress import StressResult, StressRow
from cappatrader.config import load_config


def _cfg_file(tmp_path, body: str = ""):
    p = tmp_path / "c.yaml"
    p.write_text(f"runtime:\n  state_db: {tmp_path / 'st.sqlite'}\n  log_dir: {tmp_path / 'logs'}\n{body}")
    return p


def test_yaml_type_errors_are_reported_in_turkish(tmp_path):
    with pytest.raises(ValueError, match="live.leverage: tam sayı bekleniyor"):
        load_config(_cfg_file(tmp_path, 'live:\n  leverage: "3"\n'), env_file=None)
    with pytest.raises(ValueError, match="account.allow_short: true/false bekleniyor"):
        load_config(_cfg_file(tmp_path, "account:\n  allow_short: 1\n"), env_file=None)
    c = load_config(_cfg_file(tmp_path, "account:\n  starting_balance: 500\nlive:\n  leverage: 3.0\n"), env_file=None)
    assert isinstance(c.account.starting_balance, float) and c.live.leverage == 3 and isinstance(c.live.leverage, int)


def test_tradeable_regime_typos_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="bilinmeyen rejim"):
        load_config(_cfg_file(tmp_path, "strategy:\n  squeeze:\n    tradeable_regimes: [trend_up, TREND-DOWN]\n"),
                    env_file=None)


def test_single_symbol_squeeze_config_ignores_pairs(tmp_path):
    body = "symbols: [BTCUSDT]\npairs:\n  - [ETHUSDT, BTCUSDT]\n"
    c = load_config(_cfg_file(tmp_path, body), env_file=None)
    assert c.symbols == ["BTCUSDT"] and c.strategy.name == "squeeze_vwap"
    with pytest.raises(ValueError, match="pairs"):  # the ensemble's stat-arb signal does need both legs
        load_config(_cfg_file(tmp_path, body + "strategy:\n  name: ensemble\n"), env_file=None)


def test_walkforward_refuses_short_history_before_fitting_models(tmp_path, monkeypatch, capsys):
    import cappatrader.cli as c

    monkeypatch.setattr(c, "_layers", lambda *a, **k: pytest.fail("models fitted despite too little data"))
    rc = cli.main(["--config", str(_cfg_file(tmp_path)), "walkforward", "--synthetic", "--bars", "1200", "--no-ml"])
    out = capsys.readouterr().out
    assert rc == 1 and "en az" in out and "--bars" in out  # synthetic data: --days would not help


def test_stress_verdict_distinguishes_no_edge_from_costs():
    row = lambda r: StressRow("x", 0.0, r, 0.0, 0.1, 1.0, 10, 1.0, 1.0)  # noqa: E731
    assert "avantaj yok" in StressResult(row(-0.05), row(-0.2)).summary()
    assert "maliyetlere gidiyor" in StressResult(row(0.3), row(-0.2)).summary()


def test_alerts_name_the_account(tmp_path):
    from cappatrader.live.runner import LiveEngine
    from cappatrader.live.state import StateStore
    from tests.test_live import CaptureNotifier

    c = load_config(_cfg_file(tmp_path), env_file=None)
    eng = LiveEngine(c, rest=object(), notifier=CaptureNotifier(), store=StateStore(":memory:"))
    eng.alert("KILL-SWITCH", "test")
    assert "KAĞIT HESAP" in eng.notifier.messages[-1]


def test_walkforward_precheck_counts_the_union_of_bars(tmp_path, monkeypatch, capsys):
    import pandas as pd

    import cappatrader.cli as c
    from cappatrader.data.synthetic import generate_universe

    u = generate_universe(["BTCUSDT", "ETHUSDT"], 1200, "1h", seed=3)
    u["ETHUSDT"] = u["ETHUSDT"].iloc[700:]  # listed later: the union still has 1200 bars
    monkeypatch.setattr(c, "_data", lambda args, cfg: u)
    seen = {}

    def fake_wf(cfg, data, *a, **k):
        seen["called"] = True
        raise RuntimeError("stop here")

    import cappatrader.backtest.walkforward as wf

    monkeypatch.setattr(wf, "walk_forward", fake_wf)
    monkeypatch.setattr(c, "_layers", lambda *a, **k: None)
    body = "symbols: [BTCUSDT, ETHUSDT]\npairs: []\nwalkforward:\n  is_bars: 800\n  oos_bars: 300\n"
    cli.main(["--config", str(_cfg_file(tmp_path, body)), "walkforward", "--no-ml"])
    assert seen.get("called")  # 1200 union bars >= 1100: not refused because ETH alone is shorter
    assert pd.Timestamp(u["ETHUSDT"].index[0]) > pd.Timestamp(u["BTCUSDT"].index[0])


def test_empty_yaml_collections_are_empty_not_none(tmp_path):
    c = load_config(_cfg_file(tmp_path, "walkforward:\n  grid:\n"), env_file=None)
    assert c.walkforward.grid == {}
    with pytest.raises(ValueError, match="en az bir rejim"):
        load_config(_cfg_file(tmp_path, "strategy:\n  squeeze:\n    tradeable_regimes:\n"), env_file=None)
    with pytest.raises(ValueError, match="live.leverage: tam sayı bekleniyor, boş"):
        load_config(_cfg_file(tmp_path, "live:\n  leverage:\n"), env_file=None)


def test_pairs_command_skips_pairs_outside_the_symbols(tmp_path, capsys):
    rc = cli.main(["--config", str(_cfg_file(tmp_path, "symbols: [BTCUSDT]\npairs:\n  - [ETHUSDT, BTCUSDT]\n")),
                   "pairs", "--synthetic", "--bars", "300"])
    assert rc == 0 and "atlandı" in capsys.readouterr().out


def test_squeeze_entry_shows_hmm_probability_only_with_an_hmm():
    import pandas as pd

    from cappatrader.models import Decision, Fill, Position
    from cappatrader.notify import messages as msg

    pos = Position(symbol="BTCUSDT", side=1, qty=0.01, entry_price=100.0, entry_time="2026-01-01", stop_price=98.0,
                   initial_stop=98.0, risk_per_unit=2.0, best_price=100.0, entry_regime="TREND_UP",
                   entry_confidence=50.0, entry_fee=0.0, entry_slippage_cost=0.0, reasons=["x"], take_profit=102.2,
                   style="SQZ")
    fill = Fill("BTCUSDT", 1, 0.01, 100.0, 100.0, 0.001, 0.0, 0.0, pd.Timestamp.now(tz="UTC"))
    d = Decision("BTCUSDT", None, "ENTER_LONG", 0.5, 50.0, "TREND_UP", "NORMAL", 100.0, 1.0, style="SQZ")
    head = lambda **k: msg.entry_message(pos, fill, d, [], 1000, 900, 1000, 1, **k).splitlines()[4]  # noqa: E731
    assert "HMM" not in head() and "HMM" not in head(hmm_prob=float("nan"))
    assert "HMM olasılığı %66" in head(hmm_prob=0.66)
