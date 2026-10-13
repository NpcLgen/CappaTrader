import pytest

from cappatrader.config import Config
from cappatrader.data.synthetic import generate_universe


@pytest.fixture
def cfg() -> Config:
    c = Config()
    c.signals.ml.train_bars = 600
    c.signals.ml.retrain_every = 300
    c.signals.ml.n_estimators = 30
    c.costs.latency_ms = 0.0  # keeps live-engine tests instant
    c.data.alt_data_enabled = False
    # small HMM / GARCH windows keep the suite fast on 2,500-bar synthetic data
    c.regime.hmm.train_bars, c.regime.hmm.min_train, c.regime.hmm.retrain_every = 600, 300, 300
    c.regime.hmm.n_iter = 30
    c.regime.garch.train_bars, c.regime.garch.min_train, c.regime.garch.retrain_every = 600, 300, 300
    c.validate()
    return c


@pytest.fixture
def ens_cfg(cfg) -> Config:
    """The original multi-signal ensemble strategy (ADX-free: still uses the HMM/GARCH regime layer)."""
    return cfg.with_overrides({"strategy.name": "ensemble"})


@pytest.fixture(scope="session")
def universe():
    return generate_universe(n_bars=2500, seed=3)


@pytest.fixture
def btc(universe):
    return universe["BTCUSDT"]
