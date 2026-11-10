"""Permutation importance: no effect on forecasts, no unpublished data, correct columns, recovers a known driver."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from conftest import FAST_MODEL_PARAMS, MONTHS, make_config, random_panel
from src.env_forecast import FeatureBlock, build_models, run_panel
from src.env_forecast.importance import RATE, design_columns, importance_groups

IMPORTANCE_MODELS = ("linear", "lr", "xgboost", "lstm")


def importance_config(**overrides):
    settings = dict(models=IMPORTANCE_MODELS, model_params=FAST_MODEL_PARAMS, eval_start="2019-07", eval_end="2019-08",
                    horizons=(1, 6), importance_draws=4)
    settings.update(overrides)
    return make_config(**settings)


def test_importance_does_not_change_forecasts():
    config = importance_config()
    panel = random_panel()
    importance = []
    with_importance = run_panel(panel, config, build_models(config), importance=importance)
    plain = dataclasses.replace(config, importance_draws=0)
    without = run_panel(panel, plain, build_models(plain))
    pd.testing.assert_frame_equal(with_importance, without)
    assert importance and {"case_history", "weather", "w1"} <= {r["group"] for r in importance}


def test_importance_ignores_every_value_published_after_the_origin():
    """Substitute values come from training rows only, so unpublished data cannot move permuted forecasts."""
    config = importance_config(horizons=(3,), eval_start="2019-07", eval_end="2019-07")
    origin = pd.Period("2019-07", "M") - 3
    panel = random_panel()
    perturbed = {county: frame.copy() for county, frame in panel.items()}
    for frame in perturbed.values():
        frame.loc[frame.index > origin - config.rate.availability_lag, "rate"] *= 50
        for col, lag in config.env_cutoffs.items():
            frame.loc[frame.index > origin - lag, col] += 1_000
    original, changed = [], []
    run_panel(panel, config, build_models(config), importance=original)
    run_panel(perturbed, config, build_models(config), importance=changed)
    key = ["county", "model", "feature_set", "horizon", "group"]
    a = pd.DataFrame(original).set_index(key)["permuted_mean_prediction"].sort_index()
    b = pd.DataFrame(changed).set_index(key)["permuted_mean_prediction"].sort_index()
    pd.testing.assert_series_equal(a, b)


def test_groups_map_to_exactly_their_design_columns():
    columns = ["rate_lag1", "rate_same_month_last_year", "w1_lag1", "w1_mean3_lag1", "w1_etsfill_lag2",
               "w10_lag1", "w2_lag1", "target_month_sin"]
    assert design_columns(columns, (RATE,)) == ["rate_lag1", "rate_same_month_last_year"]
    assert design_columns(columns, ("w1",)) == ["w1_lag1", "w1_mean3_lag1", "w1_etsfill_lag2"]
    assert design_columns(columns, ("w1", "w2")) == ["w1_lag1", "w1_mean3_lag1", "w1_etsfill_lag2", "w2_lag1"]
    config = make_config()
    assert [g.name for g in importance_groups(config, "R")] == ["case_history"]
    assert "case_history" not in [g.name for g in importance_groups(config, "E")]
    assert {g.name for g in importance_groups(config, "RE") if g.level == "variable"} == set(config.env_columns)


def test_known_driver_is_important_and_noise_is_not():
    """rate[m] = 10 + 2 x[m-3]: permuting x must hurt the forecast; permuting an unrelated z must not."""
    rng = np.random.default_rng(1)
    x, z = rng.normal(size=len(MONTHS)), rng.normal(size=len(MONTHS))
    rate = 10 + 2 * pd.Series(x, index=MONTHS).shift(3).fillna(0)
    panel = {"A": pd.DataFrame({"rate": rate, "x": x, "z": z}, index=MONTHS)}
    config = make_config(
        horizons=(1,), feature_sets=("E",), calendar=False, models=("linear",), importance_draws=20,
        eval_start="2018-01", eval_end="2018-12",
        env_families={"driver": FeatureBlock(("x",), availability_lag=0, lags=(0, 1, 2)),
                      "noise": FeatureBlock(("z",), availability_lag=0, lags=(0, 1, 2))},
    )
    importance = []
    predictions = run_panel(panel, config, build_models(config), importance=importance)
    base = np.sqrt(np.mean((predictions.predicted_rate - predictions.actual_rate) ** 2))
    permuted = pd.DataFrame(importance).query("level == 'group'").groupby("group")["permuted_sq_error"].mean() ** 0.5
    assert base < 1e-6
    assert permuted["driver"] > 1.0
    assert permuted["noise"] < 1e-6


@pytest.mark.parametrize("draws", [0, 3])
def test_importance_draws_change_the_experiment_id_only_when_used(draws):
    config = make_config(importance_draws=draws)
    assert (config.experiment_id == make_config().experiment_id) == (draws == 0)
