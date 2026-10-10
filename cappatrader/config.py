"""Configuration: YAML file for strategy/risk settings, environment for secrets."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import yaml

TIMEFRAME_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

COMPONENTS = (
    "ema_trend",
    "macd",
    "breakout",
    "rsi_mr",
    "bb_mr",
    "statarb",
    "ml",
    "orderflow",
    "sentiment",
)


def _default_weights() -> dict[str, dict[str, float]]:
    # Regime-dependent ensemble weights. Momentum dominates in trends,
    # mean-reversion dominates in ranges, a blend in transition.
    return {
        "TREND": {
            "ema_trend": 1.0, "macd": 0.7, "breakout": 0.8, "rsi_mr": 0.0, "bb_mr": 0.0,
            "statarb": 0.3, "ml": 1.0, "orderflow": 0.4, "sentiment": 0.2,
        },
        "RANGE": {
            "ema_trend": 0.0, "macd": 0.2, "breakout": 0.0, "rsi_mr": 1.0, "bb_mr": 1.0,
            "statarb": 0.6, "ml": 1.0, "orderflow": 0.4, "sentiment": 0.2,
        },
        "TRANSITION": {
            "ema_trend": 0.4, "macd": 0.4, "breakout": 0.3, "rsi_mr": 0.3, "bb_mr": 0.3,
            "statarb": 0.4, "ml": 1.0, "orderflow": 0.3, "sentiment": 0.2,
        },
    }


@dataclass
class ExchangeConfig:
    rest_base: str = "https://api.binance.com"
    ws_base: str = "wss://stream.binance.com:9443"
    futures_rest_base: str = "https://fapi.binance.com"
    futures_ws_base: str = "wss://fstream.binance.com"
    # Market data source for paper trading: spot or USD-M futures. Live mode always uses futures.
    market_data: str = "spot"
    recv_window_ms: int = 5000
    request_timeout_s: float = 10.0
    max_weight_per_minute: int = 6000
    weight_safety: float = 0.8


@dataclass
class AccountConfig:
    starting_balance: float = 1000.0
    quote_asset: str = "USDT"
    # Spot is long-only. True simulates a 1x perpetual account (shorts allowed).
    allow_short: bool = False
    # "virtual": paper account starts from starting_balance.
    # "binance": read the real USDT balance (read-only key) and seed a fresh paper account with it.
    # In live mode the balance always comes from Binance.
    balance_source: str = "virtual"
    binance_wallet: str = "futures"  # futures (USD-M) | spot


@dataclass
class HMMConfig:
    """Hidden Markov Model regime detection (replaces ADX when regime.method == "hmm")."""

    n_states: int = 3
    feature_window: int = 12
    train_bars: int = 2000
    retrain_every: int = 250
    min_train: int = 500
    n_iter: int = 100
    covariance_type: str = "full"
    # A state is a trend when |mean smoothed return| >= trend_threshold * its mean volatility.
    trend_threshold: float = 0.15
    # The highest-volatility state is HIGH_VOL when its volatility >= high_vol_ratio * the calmest state's.
    high_vol_ratio: float = 1.6
    random_state: int = 42


@dataclass
class GARCHConfig:
    """GARCH(p, q) next-bar volatility forecast (replaces ATR percentile when regime.vol_method == "garch")."""

    p: int = 1
    q: int = 1
    train_bars: int = 1500
    retrain_every: int = 250
    min_train: int = 500
    # forecast / long-run volatility ratio thresholds
    high_ratio: float = 1.5
    extreme_ratio: float = 2.5
    # position size multiplier = clip(1 / ratio, min_size_mult, 1)
    min_size_mult: float = 0.25


@dataclass
class RegimeConfig:
    method: str = "hmm"  # hmm | adx
    vol_method: str = "garch"  # garch | atr
    hmm: HMMConfig = field(default_factory=HMMConfig)
    garch: GARCHConfig = field(default_factory=GARCHConfig)
    adx_period: int = 14
    adx_trend: float = 25.0
    adx_range: float = 20.0
    atr_period: int = 14
    vol_lookback: int = 500
    high_vol_pct: float = 0.90
    extreme_vol_pct: float = 0.98


@dataclass
class StatArbConfig:
    window: int = 200
    entry_z: float = 2.0
    min_corr: float = 0.6


@dataclass
class MLConfig:
    enabled: bool = True
    train_bars: int = 3000
    retrain_every: int = 1000
    horizon: int = 4
    prob_threshold: float = 0.65
    n_estimators: int = 150
    max_depth: int = 6
    min_samples_leaf: int = 50
    random_state: int = 42


@dataclass
class SignalConfig:
    ema_fast: int = 20
    ema_slow: int = 50
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    bb_period: int = 20
    bb_std: float = 2.0
    breakout_period: int = 55
    statarb: StatArbConfig = field(default_factory=StatArbConfig)
    ml: MLConfig = field(default_factory=MLConfig)


@dataclass
class EnsembleConfig:
    # Confidence is 50 + 50*composite (50 = neutral, 100 = every signal at max).
    entry_confidence: float = 65.0
    trend_exit_score: float = -0.2
    mr_exit_score: float = 0.0
    mr_max_hold_bars: int = 48
    reentry_cooldown_bars: int = 4
    weights: dict[str, dict[str, float]] = field(default_factory=_default_weights)


@dataclass
class SqueezeConfig:
    """Port of the "SQZ_Breakout_Pro_NEW" Pine Script (Squeeze Momentum) combined with VWAP."""

    length: int = 60
    bb_mult: float = 2.0
    kc_mult: float = 1.5
    calc_mode: str = "fixed"  # fixed | dynamic | percent  (Pine: Sabit | Dinamik | Yüzdesel)
    fixed_val: float = 0.0
    dyn_mult: float = 2.5
    perc_val: float = 75.0
    vwap_anchor: str = "D"  # D (UTC day) | W (UTC week) | rolling
    vwap_rolling_bars: int = 24
    # LONG when close > VWAP and momentum is rising ("state") or just turned up ("turn").
    long_trigger: str = "state"
    # SHORT when close < VWAP and momentum just turned from rising to falling ("turn") or is falling ("state").
    short_trigger: str = "turn"
    confirm_bars: int = 1  # consecutive bars the momentum direction must hold (Pine "ardışık 3" = 3)
    atr_stop_mult: float = 1.5
    take_profit_r: float = 1.1
    # HMM regimes in which new positions may be opened
    tradeable_regimes: list[str] = field(default_factory=lambda: ["TREND_UP", "TREND_DOWN", "RANGE"])


@dataclass
class StrategyConfig:
    name: str = "squeeze_vwap"  # squeeze_vwap | ensemble
    squeeze: SqueezeConfig = field(default_factory=SqueezeConfig)


@dataclass
class LiveConfig:
    """Real order execution on Binance USD-M futures (mode: live)."""

    env: str = "demo"  # demo (Binance demo trading, fake money) | mainnet (REAL money)
    # Required, verbatim, before any mainnet order is sent.
    mainnet_confirm: str = ""
    leverage: int = 2
    margin_mode: str = "isolated"  # isolated | cross
    place_exchange_stops: bool = True  # STOP_MARKET / TAKE_PROFIT_MARKET on the exchange as a safety net
    sync_interval_s: float = 30.0
    max_retries: int = 4


MAINNET_CONFIRM_PHRASE = "GERCEK PARA ILE ISLEM YAPMAYI ONAYLIYORUM"


@dataclass
class RiskConfig:
    base_risk_per_trade: float = 0.01
    max_risk_per_trade: float = 0.02
    min_risk_per_trade: float = 0.0025
    kelly_fraction: float = 0.5
    kelly_min_trades: int = 30
    kelly_lookback: int = 100
    atr_stop_mult: float = 2.0
    # Round-trip trading cost may use at most this share of the initial risk (1R);
    # a tighter ATR stop is widened (and the position shrunk) to respect it.
    max_cost_r: float = 0.2
    trail_atr_mult: float = 3.0
    trail_activate_r: float = 1.0
    take_profit_r: float | None = None
    max_position_pct: float = 0.30
    max_open_positions: int = 4
    max_drawdown: float = 0.15
    daily_loss_limit: float = 0.04
    high_vol_size_mult: float = 0.5
    min_notional: float = 10.0
    flash_move_pct: float = 0.05
    flash_window_s: int = 60
    flash_cooldown_min: int = 30


@dataclass
class CostConfig:
    taker_fee: float = 0.001
    base_slippage_bps: float = 2.0
    vol_slippage_coef: float = 0.05
    impact_coef_bps: float = 10.0
    latency_ms: float = 250.0
    latency_z: float = 1.0
    extra_slippage_bps: float = 0.0
    execution_delay_bars: int = 0


@dataclass
class WalkForwardConfig:
    is_bars: int = 4000
    oos_bars: int = 1000
    objective: str = "sharpe"
    min_trades: int = 5
    # Default grid targets the default squeeze_vwap strategy. For strategy "ensemble" use e.g.
    # {"signals.ema_fast": [12, 20], "signals.ema_slow": [50, 100], "risk.atr_stop_mult": [1.5, 2.5],
    #  "ensemble.entry_confidence": [60, 65, 75]}.
    grid: dict[str, list[Any]] = field(default_factory=lambda: {
        "strategy.squeeze.atr_stop_mult": [1.0, 1.5, 2.0],
        "strategy.squeeze.confirm_bars": [1, 2],
        "strategy.squeeze.length": [30, 60],
    })


@dataclass
class StressConfig:
    extra_slippage_bps: list[float] = field(default_factory=lambda: [0, 2, 5, 10, 20, 30, 50])
    latency_ms: list[float] = field(default_factory=lambda: [0, 250, 1000, 5000])
    delay_bars: list[int] = field(default_factory=lambda: [0, 1, 2])


@dataclass
class DataConfig:
    data_dir: str = "data"
    alt_data_enabled: bool = True
    alt_refresh_min: int = 15
    news_feeds: list[str] = field(default_factory=lambda: [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
    ])
    tick_max_deviation: float = 0.03
    outlier_mad_k: float = 8.0
    stale_after_s: float = 30.0


@dataclass
class TelegramConfig:
    enabled: bool = True
    send_startup: bool = True
    daily_summary_hour_utc: int = 20
    min_interval_s: float = 1.1


@dataclass
class RuntimeConfig:
    state_db: str = "state/cappatrader.sqlite"
    log_dir: str = "logs"
    history_bars: int = 1500
    report_dir: str = "reports"


@dataclass
class Secrets:
    binance_api_key: str = field(default="", repr=False)
    binance_secret_key: str = field(default="", repr=False)
    telegram_token: str = field(default="", repr=False)
    telegram_chat_id: str = field(default="", repr=False)

    @classmethod
    def from_env(cls) -> Secrets:
        return cls(
            binance_api_key=os.getenv("BINANCE_API_KEY", "").strip(),
            binance_secret_key=os.getenv("BINANCE_SECRET_KEY", "").strip(),
            telegram_token=os.getenv("TELEGRAM_TOKEN", "").strip(),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
        )

    @property
    def has_binance(self) -> bool:
        return bool(self.binance_api_key and self.binance_secret_key)

    @property
    def has_telegram(self) -> bool:
        return bool(self.telegram_token and self.telegram_chat_id)


@dataclass
class Config:
    mode: str = "paper"
    symbols: list[str] = field(default_factory=lambda: [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "PAXGUSDT",
    ])
    timeframe: str = "1h"
    pairs: list[list[str]] = field(default_factory=lambda: [
        ["ETHUSDT", "BTCUSDT"],
        ["SOLUSDT", "ETHUSDT"],
    ])
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    account: AccountConfig = field(default_factory=AccountConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    live: LiveConfig = field(default_factory=LiveConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    walkforward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    stress: StressConfig = field(default_factory=StressConfig)
    data: DataConfig = field(default_factory=DataConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    secrets: Secrets = field(default_factory=Secrets, repr=False)

    @property
    def state_db(self) -> str:
        """Paper and live (demo / mainnet) accounts never share a state database."""
        if self.mode != "live":
            return self.runtime.state_db
        p = Path(self.runtime.state_db)
        return str(p.with_name(f"{p.stem}_live_{self.live.env}{p.suffix}"))

    @property
    def mode_label(self) -> str:
        if self.mode != "live":
            return "KAĞIT HESAP"
        return "CANLI · DEMO (sanal para)" if self.live.env == "demo" else "CANLI · GERÇEK PARA"

    @property
    def data_market(self) -> str:
        return "futures" if self.mode == "live" else self.exchange.market_data

    @property
    def bar_ms(self) -> int:
        return TIMEFRAME_MS[self.timeframe]

    @property
    def bars_per_year(self) -> float:
        return 365 * 86_400_000 / self.bar_ms

    def with_overrides(self, overrides: dict[str, Any]) -> Config:
        """Return a copy with dotted-path overrides, e.g. {"risk.atr_stop_mult": 2.5}."""
        cfg = copy.deepcopy(self)
        for path, value in overrides.items():
            obj: Any = cfg
            parts = path.split(".")
            for part in parts[:-1]:
                obj = getattr(obj, part)
            if not hasattr(obj, parts[-1]):
                raise ValueError(f"Bilinmeyen ayar: {path}")
            setattr(obj, parts[-1], value)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        errors = []
        if self.mode not in ("paper", "live"):
            errors.append("mode: 'paper' veya 'live' olmalı")
        if self.mode == "live":
            if self.live.env not in ("demo", "mainnet"):
                errors.append("live.env: 'demo' veya 'mainnet' olmalı")
            if self.live.env == "mainnet" and self.live.mainnet_confirm != MAINNET_CONFIRM_PHRASE:
                errors.append(f"live.mainnet_confirm: gerçek parayla işlem için tam olarak '{MAINNET_CONFIRM_PHRASE}' yazılmalı")
            if not 1 <= self.live.leverage <= 20:
                errors.append("live.leverage 1-20 arasında olmalı")
            if self.live.margin_mode not in ("isolated", "cross"):
                errors.append("live.margin_mode: 'isolated' veya 'cross' olmalı")
        if self.exchange.market_data not in ("spot", "futures"):
            errors.append("exchange.market_data: 'spot' veya 'futures' olmalı")
        if self.regime.method not in ("hmm", "adx"):
            errors.append("regime.method: 'hmm' veya 'adx' olmalı")
        if self.regime.vol_method not in ("garch", "atr"):
            errors.append("regime.vol_method: 'garch' veya 'atr' olmalı")
        if self.strategy.name not in ("squeeze_vwap", "ensemble"):
            errors.append("strategy.name: 'squeeze_vwap' veya 'ensemble' olmalı")
        sq = self.strategy.squeeze
        if sq.calc_mode not in ("fixed", "dynamic", "percent"):
            errors.append("strategy.squeeze.calc_mode: fixed | dynamic | percent")
        if sq.vwap_anchor not in ("D", "W", "rolling"):
            errors.append("strategy.squeeze.vwap_anchor: D | W | rolling")
        if sq.long_trigger not in ("state", "turn") or sq.short_trigger not in ("state", "turn"):
            errors.append("strategy.squeeze.long_trigger/short_trigger: state | turn")
        if sq.take_profit_r <= 0 or sq.atr_stop_mult <= 0:
            errors.append("strategy.squeeze: take_profit_r ve atr_stop_mult pozitif olmalı")
        if self.account.balance_source not in ("virtual", "binance"):
            errors.append("account.balance_source: 'virtual' veya 'binance' olmalı")
        if self.account.binance_wallet not in ("futures", "spot"):
            errors.append("account.binance_wallet: 'futures' veya 'spot' olmalı")
        if self.timeframe not in TIMEFRAME_MS:
            errors.append(f"timeframe: {self.timeframe} desteklenmiyor ({', '.join(TIMEFRAME_MS)})")
        if not self.symbols:
            errors.append("symbols: en az bir parite gerekli")
        if self.signals.ema_fast >= self.signals.ema_slow:
            errors.append("signals: ema_fast < ema_slow olmalı")
        if not 50 <= self.ensemble.entry_confidence <= 100:
            errors.append("ensemble.entry_confidence 50-100 arasında olmalı")
        if not 0.5 < self.signals.ml.prob_threshold < 1:
            errors.append("signals.ml.prob_threshold 0.5-1 arasında olmalı")
        if self.regime.adx_range > self.regime.adx_trend:
            errors.append("regime: adx_range <= adx_trend olmalı")
        for pair in self.pairs:
            if len(pair) != 2 or any(s not in self.symbols for s in pair):
                errors.append(f"pairs: {pair} iki geçerli sembol içermeli")
        for regime, weights in self.ensemble.weights.items():
            unknown = set(weights) - set(COMPONENTS)
            if unknown:
                errors.append(f"ensemble.weights.{regime}: bilinmeyen bileşen {sorted(unknown)}")
        if not 0 < self.risk.max_drawdown < 1:
            errors.append("risk.max_drawdown 0-1 arasında olmalı")
        if self.account.starting_balance <= 0:
            errors.append("account.starting_balance pozitif olmalı")
        if errors:
            raise ValueError("Geçersiz yapılandırma:\n  - " + "\n  - ".join(errors))


def _apply(obj: Any, data: dict[str, Any], path: str = "") -> None:
    for key, value in data.items():
        full = f"{path}{key}"
        if key == "secrets":
            raise ValueError("Gizli anahtarlar YAML'a yazılmaz; .env kullanın")
        if not hasattr(obj, key):
            raise ValueError(f"Bilinmeyen ayar: {full}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply(current, value, f"{full}.")
        elif full == "ensemble.weights" and isinstance(value, dict):
            for regime, weights in value.items():
                current.setdefault(regime, {}).update(weights)
        else:
            setattr(obj, key, value)


def load_config(path: str | os.PathLike | None = None, env_file: str | None = ".env") -> Config:
    """Load defaults, overlay the YAML file (if any) and read secrets from the environment."""
    if env_file:
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)
    cfg = Config()
    if path is None and Path("config/config.yaml").exists():
        path = "config/config.yaml"
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        _apply(cfg, data)
    cfg.secrets = Secrets.from_env()
    cfg.validate()
    return cfg
