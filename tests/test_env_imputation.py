"""Tests for origin-consistent ETS forward filling and config inheritance."""

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from threadpoolctl import threadpool_limits

from conftest import make_config, random_panel
from src.env_forecast import FeatureBlock, HarnessConfig, build_design, build_models, mask_history, run_panel
from src.env_forecast.config import load_config_dict

TARGET = pd.Period("2019-07", "M")
ETS_FAMILIES = {
    "weather": FeatureBlock(("w1", "w2"), availability_lag=1, lags=(0, 1)),
    "ag_chemicals": FeatureBlock(("c1",), availability_lag=36, lags=(0, 2), rolling_means=(12,), forecast_fill="ets", fill_lag=1),
}


def ets_config(**overrides):
    settings = dict(env_families=ETS_FAMILIES, models=("seasonal_naive", "ridge", "xgboost"),
                    model_params={"xgboost": {"params": {"n_estimators": 40}}})
    settings.update(overrides)
    return make_config(**settings)


@pytest.mark.parametrize("horizon", [1, 6])
def test_filled_values_ignore_unreleased_pesticide_months(horizon):
    origin = TARGET - horizon
    config = ets_config(horizons=(horizon,), eval_start=str(TARGET), eval_end=str(TARGET))
    panel = random_panel()
    perturbed = {county: frame.copy() for county, frame in panel.items()}
    for frame in perturbed.values():
        frame.loc[frame.index > origin - 36, "c1"] *= 25
    original = run_panel(panel, config, build_models(config))
    changed = run_panel(perturbed, config, build_models(config))
    pd.testing.assert_series_equal(original["predicted_rate"], changed["predicted_rate"])
    # released pesticide months inside the ETS window must matter (non-vacuous check)
    for frame in perturbed.values():
        frame.loc[(frame.index <= origin - 36) & (frame.index > origin - 60), "c1"] *= 4
    moved = run_panel(perturbed, config, build_models(config))
    environmental = original["feature_set"].isin(["E", "RE"])
    assert not np.allclose(original.loc[environmental, "predicted_rate"], moved.loc[environmental, "predicted_rate"])


@pytest.mark.parametrize("horizon", [1, 3])
def test_training_rows_reuse_the_fill_their_own_origin_would_make(horizon):
    """A training row s must carry exactly the features a forecaster standing at s would have built."""
    config = ets_config(horizons=(horizon,))
    frame = random_panel()["A"]
    origin = pd.Period("2018-06", "M")
    design = build_design(mask_history(frame, origin, config), horizon, config)
    ets_columns = [c for c in design.X.columns if "_etsfill_" in c]
    for row in (origin - 1 - horizon, origin - 40, origin - 100):
        own = build_design(mask_history(frame, row, config), horizon, config)
        pd.testing.assert_series_equal(design.X.loc[row, ets_columns], own.X.loc[row, ets_columns], check_names=False)


def test_fill_matches_a_direct_holt_winters_forecast():
    config = ets_config()
    frame = random_panel()["A"]
    row = pd.Period("2018-06", "M")
    design = build_design(mask_history(frame, row, config), 1, config)
    window = frame["c1"].loc[:row - 36].iloc[-72:].to_numpy()
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("ignore")
        forecast = np.clip(ExponentialSmoothing(window, trend=None, seasonal="add", seasonal_periods=12).fit().forecast(35), 0, None)
    extended = np.concatenate([window, forecast])
    assert design.X.at[row, "c1_etsfill_lag1"] == pytest.approx(extended[-1])
    assert design.X.at[row, "c1_etsfill_lag3"] == pytest.approx(extended[-3])
    assert design.X.at[row, "c1_etsfill_mean12_lag1"] == pytest.approx(extended[-12:].mean())


def test_invalid_fill_settings_are_rejected():
    with pytest.raises(ValueError):
        make_config(env_families={"x": FeatureBlock(("c1",), availability_lag=3, forecast_fill="ets", fill_lag=3)})


def test_config_extends_merges_and_removes(tmp_path):
    base = {"name": "base", "target_col": "rate", "env_families": {
        "weather": {"columns": ["w1"], "availability_lag": 1}, "ag": {"columns": ["c1"], "availability_lag": 24}}}
    (tmp_path / "base.json").write_text(json.dumps(base))
    (tmp_path / "child.json").write_text(json.dumps({"extends": "base.json", "name": "child", "env_families": {"ag": None, "weather": {"lags": [0, 1]}}}))
    merged = load_config_dict(tmp_path / "child.json")
    assert merged["name"] == "child" and "ag" not in merged["env_families"]
    assert merged["env_families"]["weather"] == {"columns": ["w1"], "availability_lag": 1, "lags": [0, 1]}
    assert HarnessConfig.from_json(tmp_path / "child.json").env_families["weather"].lags == (0, 1)
