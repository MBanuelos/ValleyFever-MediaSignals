"""Shared synthetic panels for the environmental harness tests."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env_forecast import FeatureBlock, HarnessConfig, RateSpec  # noqa: E402

MONTHS = pd.period_range("2001-01", "2024-12", freq="M")
OFFSET = 100_000  # column j stores j * OFFSET + month position, so every value names its source month
SOURCE_COLUMNS = ["rate", "w1", "w2", "f1", "c1"]


def decode_month(value: float) -> pd.Period:
    """Source month of a value from the encoded panel."""
    return MONTHS[int(round(value)) % OFFSET]


def encoded_panel() -> dict[str, pd.DataFrame]:
    position = np.arange(len(MONTHS))
    frame = pd.DataFrame({col: j * OFFSET + position for j, col in enumerate(SOURCE_COLUMNS, start=1)}, index=MONTHS)
    frame.index.name = "month"
    return {"A": frame.astype(float)}


def random_panel(seed: int = 0) -> dict[str, pd.DataFrame]:
    """Positive seasonal series with noise, for perturbation and reproducibility tests."""
    rng = np.random.default_rng(seed)
    season = 1 + 0.5 * np.sin(2 * np.pi * MONTHS.month.to_numpy() / 12)
    frame = pd.DataFrame({col: (3 + rng.gamma(2.0, 1.0, len(MONTHS))) * season for col in SOURCE_COLUMNS}, index=MONTHS)
    frame.index.name = "month"
    return {"A": frame, "B": frame * 1.7 + 0.3}


def make_config(**overrides) -> HarnessConfig:
    settings = dict(
        name="synthetic", counties=(), target_col="rate", window_months=120, horizons=(1, 3, 6),
        eval_start="2017-01", eval_end="2018-12", confirmation_start="2018-01",
        rate=RateSpec(availability_lag=1, lags=tuple(range(12)), seasonal_lag=True),
        env_families={
            "weather": FeatureBlock(("w1", "w2"), availability_lag=1, lags=(0, 1, 2), rolling_means=(3,)),
            "wildfire": FeatureBlock(("f1",), availability_lag=2, lags=(0, 1)),
            "ag_chemicals": FeatureBlock(("c1",), availability_lag=24, lags=(0,)),
        },
        models=("seasonal_naive", "persistence", "climatology", "linear", "ridge"),
        min_train_rows=24,
    )
    settings.update(overrides)
    return HarnessConfig(**settings)


@pytest.fixture
def config() -> HarnessConfig:
    return make_config()
