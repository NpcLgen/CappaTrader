"""Synthetic regime-switching market data for offline demos and tests.

The generator produces trending, ranging and shock phases on a common crypto
factor, cointegrated altcoin legs, and a low-volatility gold-like asset. It is
for exercising the machinery only; profitability on it says nothing about
real markets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cappatrader.config import TIMEFRAME_MS

_PROFILE = {
    # symbol: (start price, beta to crypto factor, idiosyncratic vol per bar)
    "BTCUSDT": (60_000.0, 1.0, 0.0008),
    "ETHUSDT": (3_000.0, 1.2, 0.0015),
    "SOLUSDT": (150.0, 1.5, 0.0025),
    "BNBUSDT": (550.0, 0.8, 0.0015),
    "XRPUSDT": (0.6, 1.1, 0.0025),
}


def _regime_path(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    drift = np.zeros(n)
    volmult = np.ones(n)
    is_range = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        state = rng.choice(["up", "down", "range", "shock"], p=[0.3, 0.25, 0.37, 0.08])
        length = int(rng.geometric(1 / (120 if state == "shock" else 400)))
        j = min(n, i + max(length, 30))
        if state == "up":
            drift[i:j] = 0.0005
        elif state == "down":
            drift[i:j] = -0.0005
        elif state == "range":
            is_range[i:j] = True
            volmult[i:j] = 0.8
        else:
            volmult[i:j] = 3.0
        i = j
    return drift, volmult, is_range


def generate_universe(
    symbols: list[str] | None = None,
    n_bars: int = 8000,
    timeframe: str = "1h",
    seed: int = 7,
    start: str = "2025-01-01",
) -> dict[str, pd.DataFrame]:
    symbols = symbols or [*_PROFILE, "PAXGUSDT"]
    rng = np.random.default_rng(seed)
    sub = 4
    drift, volmult, is_range = _regime_path(n_bars, rng)
    scale = TIMEFRAME_MS[timeframe] / TIMEFRAME_MS["15m"]
    drift = drift * np.sqrt(scale)  # keep per-bar signal-to-noise constant across timeframes
    base_sigma = 0.004 * np.sqrt(scale)
    # Volatility clustering on top of the regime multiplier.
    log_vol = np.zeros(n_bars)
    for t in range(1, n_bars):
        log_vol[t] = 0.97 * log_vol[t - 1] + 0.12 * rng.standard_normal()
    sigma = base_sigma * volmult * np.exp(log_vol - 0.5 * 0.12**2 / (1 - 0.97**2))

    factor = np.zeros(n_bars * sub)
    x = 0.0
    anchor = 0.0
    for t in range(n_bars):
        if is_range[t] and (t == 0 or not is_range[t - 1]):
            anchor = x
        for k in range(sub):
            mr = -0.02 * (x - anchor) if is_range[t] else 0.0
            x += drift[t] / sub + mr / sub + sigma[t] / np.sqrt(sub) * rng.standard_normal()
            factor[t * sub + k] = x

    index = pd.date_range(start=start, periods=n_bars, freq=pd.Timedelta(milliseconds=TIMEFRAME_MS[timeframe]),
                          tz="UTC", name="open_time")
    out = {}
    for sym in symbols:
        if sym == "PAXGUSDT" or sym not in _PROFILE:
            p0, beta, idio = (2_400.0, 0.05, 0.0012) if sym == "PAXGUSDT" else (100.0, 1.0, 0.002)
        else:
            p0, beta, idio = _PROFILE[sym]
        # Mean-reverting idiosyncratic leg keeps altcoins cointegrated with the factor.
        e = np.zeros(n_bars * sub)
        idio_step = idio * np.sqrt(scale) / np.sqrt(sub)
        for t in range(1, len(e)):
            e[t] = 0.998 * e[t - 1] + idio_step * rng.standard_normal()
        drift_extra = 0.00002 * np.arange(len(e)) / sub if sym == "PAXGUSDT" else 0.0
        logp = np.log(p0) + beta * factor + e + drift_extra
        path = np.exp(logp).reshape(n_bars, sub)
        close = path[:, -1]
        open_ = np.concatenate([[p0], close[:-1]])
        high = np.maximum(path.max(axis=1), open_)
        low = np.minimum(path.min(axis=1), open_)
        ret = np.abs(np.log(close / open_))
        volume = rng.lognormal(mean=0.0, sigma=0.4, size=n_bars) * (1 + 150 * ret) * (1e6 / p0)
        out[sym] = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume,
             "quote_volume": volume * close},
            index=index,
        )
    return out
