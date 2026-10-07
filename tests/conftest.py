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
    c.validate()
    return c


@pytest.fixture(scope="session")
def universe():
    return generate_universe(n_bars=2500, seed=3)


@pytest.fixture
def btc(universe):
    return universe["BTCUSDT"]
