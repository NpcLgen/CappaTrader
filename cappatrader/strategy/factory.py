"""Strategy selection shared by backtest, walk-forward and live trading.

cfg.strategy.name picks the decision engine ("squeeze_vwap" or "ensemble"); the
helpers below give every caller the same risk settings, explanation and GARCH
size multiplier for that strategy, so backtest and live cannot drift apart.
"""

from __future__ import annotations

import copy
import math

import pandas as pd

from cappatrader.config import Config, RiskConfig
from cappatrader.models import Decision, Position
from cappatrader.strategy.decision import DecisionEngine, FrameView
from cappatrader.strategy.explain import explain


def is_squeeze(cfg: Config) -> bool:
    return cfg.strategy.name == "squeeze_vwap"


def make_decider(cfg: Config):
    if is_squeeze(cfg):
        from cappatrader.strategy.squeeze_decision import SqueezeVwapEngine

        return SqueezeVwapEngine(cfg)
    return DecisionEngine(cfg)


def strategy_risk(cfg: Config) -> RiskConfig:
    """Risk settings with the active strategy's stop / target rules applied."""
    if not is_squeeze(cfg):
        return cfg.risk
    sq = cfg.strategy.squeeze
    r = copy.copy(cfg.risk)
    r.atr_stop_mult = sq.atr_stop_mult
    r.take_profit_r = sq.take_profit_r
    r.trail_activate_r = math.inf  # the Pine strategy exits on target, stop or momentum turn only
    r.max_cost_r = 0.0  # the user's rule is a pure ATR stop: never widen it for costs (that moves the 1.1R target)
    return r


def uses_trailing(cfg: Config) -> bool:
    return not is_squeeze(cfg)


def size_mult_at(cfg: Config, v: FrameView, i: int) -> float:
    if cfg.regime.vol_method != "garch" or "garch_size_mult" not in v.cols:
        return float("nan")
    x = v["garch_size_mult"][i]
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def decide(decider, cfg: Config, symbol: str, v: FrameView, i: int, position: Position | None,
           bars_since_exit: int | None) -> Decision:
    d = decider.evaluate(symbol, v, i, position, bars_since_exit)
    d.size_mult = size_mult_at(cfg, v, i)
    rev = getattr(d, "reverse", None)
    if rev is not None:
        rev.size_mult = d.size_mult
    return d


def explain_decision(cfg: Config, row: pd.Series, side: int, live_notes: dict[str, str] | None = None) -> list[str]:
    if is_squeeze(cfg):
        from cappatrader.strategy.squeeze_decision import explain_squeeze

        # the squeeze rules ignore order flow and sentiment: never list them as a reason for the trade
        return explain_squeeze(row, side, cfg)
    return explain(row, side, cfg, live_notes)
