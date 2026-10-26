"""Model-specific harness tests for SARIMA and XGBoost (alignment, rolling window, caching, parallel runs)."""

import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace.sarimax import SARIMAX
from threadpoolctl import threadpool_limits

from conftest import make_config, random_panel
from src.env_forecast import build_models, run_panel
from test_env_alignment import ALL_MODELS

TARGET = pd.Period("2019-07", "M")


FAST_XGBOOST = {"xgboost": {"params": {"n_estimators": 40}}}


def short_config(**overrides):
    settings = dict(eval_start="2019-01", eval_end="2019-04", models=ALL_MODELS, model_params=FAST_XGBOOST)
    settings.update(overrides)
    return make_config(**settings)


@pytest.mark.parametrize("horizon", [1, 3, 6])
def test_rolling_window_ignores_data_older_than_the_window(horizon):
    """Rates before the first training label, and inputs older than any lag reaches, must not matter."""
    config = make_config(horizons=(horizon,), eval_start=str(TARGET), eval_end=str(TARGET), models=ALL_MODELS, model_params=FAST_XGBOOST)
    first_label = TARGET - horizon - config.rate.availability_lag - config.window_months + 1
    oldest_input = first_label - horizon - 40  # beyond the longest lag + rolling span in make_config
    panel = random_panel()
    perturbed = {county: frame.copy() for county, frame in panel.items()}
    for frame in perturbed.values():
        frame.loc[frame.index < oldest_input] *= 7
    pd.testing.assert_frame_equal(run_panel(panel, config, build_models(config)), run_panel(perturbed, config, build_models(config)))
    sarima_only = dataclasses.replace(config, models=("sarima",))
    for frame in perturbed.values():
        frame.loc[frame.index < first_label, "rate"] *= 3
    pd.testing.assert_frame_equal(run_panel(panel, sarima_only, build_models(sarima_only)),
                                  run_panel(perturbed, sarima_only, build_models(sarima_only)))


@pytest.mark.parametrize("case_lag", [0, 1, 3])
def test_sarima_forecasts_the_target_month_from_the_published_window(case_lag):
    """SARIMA must fit on the last W published rates and return the (h + case lag)-step forecast."""
    config = make_config(horizons=(1, 3, 6), eval_start=str(TARGET), eval_end=str(TARGET), models=("sarima",),
                         rate=dataclasses.replace(make_config().rate, availability_lag=case_lag))
    panel = random_panel()
    predictions = run_panel(panel, config, build_models(config))
    for _, row in predictions.iterrows():
        origin = pd.Period(row["forecast_origin"], "M")
        window = panel[row["county"]]["rate"].loc[origin - case_lag - config.window_months + 1:origin - case_lag]
        with warnings.catch_warnings(), threadpool_limits(limits=1):
            warnings.simplefilter("ignore")
            fitted = SARIMAX(np.log1p(window.to_numpy()), order=(1, 0, 1), seasonal_order=(1, 0, 1, 12), trend="c").fit(disp=False)
        expected = np.expm1(fitted.forecast(row["horizon"] + case_lag)[-1])
        assert row["predicted_rate"] == pytest.approx(expected, rel=1e-6)
        assert row["feature_set"] == "R"


def test_sarima_cache_never_reuses_a_fit_for_different_data():
    config = short_config(models=("sarima",))
    shared_model = build_models(config)
    run_panel(random_panel(0), config, shared_model)
    reused = run_panel(random_panel(1), config, shared_model)
    fresh = run_panel(random_panel(1), config, build_models(config))
    pd.testing.assert_frame_equal(reused, fresh)


def test_new_models_share_targets_and_label_feature_sets():
    config = short_config()
    predictions = run_panel(random_panel(), config, build_models(config))
    labels = predictions.groupby("model")["feature_set"].apply(lambda s: tuple(sorted(set(s))))
    assert labels["sarima"] == ("R",) and labels["xgboost"] == ("E", "R", "RE")
    assert predictions.groupby(["county", "model", "feature_set", "horizon"])["target_month"].apply(frozenset).nunique() == 1
    assert predictions["predicted_rate"].notna().all()


def test_xgboost_is_reproducible_and_seed_sensitive():
    config = short_config(models=("xgboost",))
    first = run_panel(random_panel(), config, build_models(config))
    pd.testing.assert_frame_equal(first, run_panel(random_panel(), config, build_models(config)))
    reseeded = dataclasses.replace(config, seed=7)
    assert not np.allclose(first["predicted_rate"], run_panel(random_panel(), reseeded, build_models(reseeded))["predicted_rate"])


def test_parallel_counties_match_serial_run():
    config = short_config(models=("seasonal_naive", "ridge", "sarima", "xgboost"), eval_end="2019-02")
    serial = run_panel(random_panel(), config, build_models(config))
    parallel = run_panel(random_panel(), config, build_models(config), n_jobs=2)
    pd.testing.assert_frame_equal(serial, parallel)
