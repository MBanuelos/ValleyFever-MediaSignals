"""Model-specific harness tests for SARIMA and XGBoost (alignment, rolling window, caching, parallel runs)."""

import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace.sarimax import SARIMAX
from threadpoolctl import threadpool_limits

from conftest import ALL_MODELS, FAST_MODEL_PARAMS, make_config, random_panel
from src.env_forecast import build_models, run_panel
from src.env_forecast.features import window_mean

TARGET = pd.Period("2019-07", "M")




def short_config(**overrides):
    settings = dict(eval_start="2019-01", eval_end="2019-04", models=ALL_MODELS, model_params=FAST_MODEL_PARAMS)
    settings.update(overrides)
    return make_config(**settings)


@pytest.mark.parametrize("horizon", [1, 3, 6])
def test_rolling_window_ignores_data_older_than_the_window(horizon):
    """Rates before the first training label, and inputs older than any lag reaches, must not matter."""
    config = make_config(horizons=(horizon,), eval_start=str(TARGET), eval_end=str(TARGET), models=ALL_MODELS, model_params=FAST_MODEL_PARAMS)
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
    assert labels["sarima"] == ("R",) and labels["sarimax"] == ("RE",)
    assert labels["xgboost"] == labels["lr"] == labels["lstm"] == ("E", "R", "RE")
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


def test_lr_with_rate_history_is_an_ar12_ridge_on_rate_lags_only():
    """LR-R must use exactly the 12 published rate lags (AR(12)); E and RE keep their full inputs."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from src.env_forecast import build_design, mask_history
    from src.env_forecast.harness import training_rows

    config = make_config(horizons=(3,), eval_start=str(TARGET), eval_end=str(TARGET), models=("lr",))
    panel = random_panel()
    predictions = run_panel(panel, config, build_models(config))
    origin = TARGET - 3
    design = build_design(mask_history(panel["A"], origin, config), 3, config)
    rows = training_rows(design, origin, 3, config)
    ar = [f"rate_lag{k}" for k in range(1, 13)]
    manual = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                           RidgeCV(alphas=np.logspace(-2, 3, 11), cv=TimeSeriesSplit(n_splits=5)))
    manual.fit(design.X.loc[rows, ar], design.y.loc[rows])
    expected = max(float(manual.predict(design.X.loc[[origin], ar])[0]), 0.0)
    got = predictions[(predictions.county == "A") & (predictions.feature_set == "R")]["predicted_rate"].item()
    assert got == pytest.approx(expected)


@pytest.mark.parametrize("horizon", [1, 6])
def test_sarimax_regressors_are_lagged_beyond_the_forecast_horizon(horizon):
    """Each regressor enters as x[m - (h + availability lag)], so even the target month's value was published at the origin."""
    exog = FAST_MODEL_PARAMS["sarimax"]["exog"]
    config = make_config(horizons=(horizon,), eval_start=str(TARGET), eval_end=str(TARGET), models=("sarimax",),
                         model_params={"sarimax": {"exog": exog}})
    panel = random_panel()
    got = run_panel(panel, config, build_models(config))
    frame, origin = panel["A"], TARGET - horizon
    end = origin - 1
    train = pd.period_range(end - 119, end, freq="M")
    future = pd.period_range(end + 1, TARGET, freq="M")

    def regressors(months):
        w1 = window_mean(frame["w1"], 3).reindex(months - (horizon + 1))
        f1 = np.log1p(frame["f1"]).reindex(months - (horizon + 2))
        return np.column_stack([w1, f1])

    x_train, x_future = regressors(train), regressors(future)
    assert (future - (horizon + 2)).max() == origin - 2  # newest fire value used is exactly the newest published one
    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("ignore")
        fitted = SARIMAX(np.log1p(frame["rate"].reindex(train).to_numpy()), exog=(x_train - mean) / std,
                         order=(1, 0, 1), seasonal_order=(1, 0, 1, 12), trend="c").fit(disp=False)
    expected = np.expm1(fitted.forecast(len(future), exog=(x_future - mean) / std)[-1])
    row = got[got.county == "A"]
    assert row["feature_set"].item() == "RE" and row["predicted_rate"].item() == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("feature_set", ["R", "E", "RE"])
def test_lstm_sequences_only_hold_published_values(feature_set):
    """Every value in the LSTM's monthly input row k must come from a month published by k."""
    from conftest import decode_month, encoded_panel
    from src.env_forecast import ForecastTask, mask_history
    from src.env_forecast.lstm import monthly_inputs

    config = make_config()
    origin = pd.Period("2018-06", "M")
    history = mask_history(encoded_panel()["A"], origin, config)
    empty = pd.DataFrame(index=pd.PeriodIndex([], freq="M"))
    task = ForecastTask("A", origin, 3, feature_set, history, empty, pd.Series(dtype=float), empty,
                        pd.PeriodIndex([], freq="M"), config, 0)
    inputs = monthly_inputs(task)
    lags = {"rate": config.rate.availability_lag, **config.env_cutoffs}
    assert set(inputs.columns) - {"month_sin", "month_cos"} == ({"rate"} if feature_set == "R" else set(lags) if feature_set == "RE" else set(config.env_cutoffs))
    for column in set(inputs.columns) - {"month_sin", "month_cos"}:
        for month, value in inputs[column].dropna().items():
            assert decode_month(value) == month - lags[column]


def test_lstm_is_reproducible_and_seed_sensitive():
    config = short_config(models=("lstm",), eval_end="2019-02")
    first = run_panel(random_panel(), config, build_models(config))
    pd.testing.assert_frame_equal(first, run_panel(random_panel(), config, build_models(config)))
    reseeded = dataclasses.replace(config, seed=7)
    assert not np.allclose(first["predicted_rate"], run_panel(random_panel(), reseeded, build_models(reseeded))["predicted_rate"])
