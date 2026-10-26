"""
Synthetic tests for chronological correctness of the environmental harness.

The encoded panel stores, in every cell, the position of its own month, so any feature or
label can be traced back to the month it came from and checked against the origin cutoffs.
"""

import dataclasses
import re

import numpy as np
import pandas as pd
import pytest

from conftest import MONTHS, decode_month, encoded_panel, make_config, random_panel
from src.env_forecast import (
    PREDICTION_COLUMNS, FeatureBlock, RateSpec, build_design, build_models, diebold_mariano, mask_history,
    run_panel, summarize, training_rows,
)

ORIGIN = pd.Period("2016-06", "M")


def expected_source(column: str, row: pd.Period, horizon: int) -> pd.Period | None:
    """Month a design column should read for pseudo-origin `row`; None for calendar columns."""
    if column == "rate_same_month_last_year":
        return row + horizon - 12
    rolling = re.search(r"_mean(\d+)_lag(\d+)$", column)
    if rolling:
        return row - int(rolling.group(2))  # last month inside the rolling window
    lagged = re.search(r"_lag(\d+)$", column)
    return row - int(lagged.group(1)) if lagged else None


def test_mask_history_respects_each_publication_delay(config):
    history = mask_history(encoded_panel()["A"], ORIGIN, config)
    assert history.index.max() == ORIGIN
    assert history["rate"].last_valid_index() == ORIGIN - 1
    assert history["w1"].last_valid_index() == ORIGIN - 1
    assert history["f1"].last_valid_index() == ORIGIN - 2
    assert history["c1"].last_valid_index() == ORIGIN - 24


@pytest.mark.parametrize("horizon", [1, 3, 6, 12])
@pytest.mark.parametrize("case_lag", [0, 1, 3])
def test_every_design_cell_reads_the_intended_month(horizon, case_lag):
    config = make_config(rate=RateSpec(availability_lag=case_lag, lags=tuple(range(12))))
    design = build_design(mask_history(encoded_panel()["A"], ORIGIN, config), horizon, config)
    for column in design.X.columns:
        values = design.X[column].dropna()
        if column.startswith("target_month_"):
            months = (values.index + horizon).month.to_numpy()
            trig = np.sin if column.endswith("sin") else np.cos
            np.testing.assert_allclose(values.to_numpy(), trig(2 * np.pi * months / 12))
            continue
        rolling = re.search(r"_mean(\d+)_", column)
        centre_shift = (int(rolling.group(1)) - 1) / 2 if rolling else 0  # a mean of w months sits (w-1)/2 before its end
        for row, value in values.items():
            assert decode_month(value + centre_shift) == expected_source(column, row, horizon), (column, row)
    labels = design.y.dropna()
    assert all(decode_month(value) == row + horizon for row, value in labels.items())
    assert labels.index.max() + horizon == ORIGIN - case_lag


@pytest.mark.parametrize("horizon,case_lag,present", [(1, 1, True), (3, 1, True), (11, 1, False), (12, 1, False), (10, 3, False)])
def test_same_month_last_year_only_when_published(horizon, case_lag, present):
    """y[s+h-12] is added only if published by s (12-h >= case lag) and not already a rate lag (lags here are 1-2 months back)."""
    config = make_config(rate=RateSpec(availability_lag=case_lag, lags=(0, 1)))
    design = build_design(mask_history(encoded_panel()["A"], ORIGIN, config), horizon, config)
    assert ("rate_same_month_last_year" in design.X.columns) == present


@pytest.mark.parametrize("horizon", [1, 3, 6])
@pytest.mark.parametrize("window", [60, 120])
def test_training_labels_are_published_by_the_origin(horizon, window):
    config = make_config(window_months=window)
    design = build_design(mask_history(encoded_panel()["A"], ORIGIN, config), horizon, config)
    rows = training_rows(design, ORIGIN, horizon, config)
    labels = rows + horizon
    assert len(rows) == window
    assert labels.max() == ORIGIN - 1 and labels.min() == ORIGIN - window
    assert [decode_month(v) for v in design.y.loc[rows]] == list(labels)


class SpyModel:
    """Records everything a model can see and checks it against the origin cutoffs."""

    name, uses_features = "spy", True

    def __init__(self):
        self.calls = 0

    def fit_predict(self, task):
        self.calls += 1
        config, origin = task.config, task.origin
        assert task.history.index.max() == origin
        assert task.history[config.target_col].last_valid_index() <= origin - config.rate.availability_lag
        for col, lag in config.env_cutoffs.items():
            assert task.history[col].last_valid_index() <= origin - lag
        assert task.label_months.max() <= origin - config.rate.availability_lag
        assert list(task.X_forecast.index) == [origin]
        for frame in (task.X_train, task.X_forecast):
            assert frame.index.max() <= origin
            for column in frame.columns:
                if column.startswith("target_month_") or "_mean" in column:
                    continue
                lag = config.rate.availability_lag if column.startswith("rate_") else config.env_cutoffs[column.rsplit("_lag", 1)[0]]
                for row, value in frame[column].dropna().items():
                    assert decode_month(value) <= row - lag, (column, row)
        return 0.0


def test_full_run_never_exposes_unpublished_data(config):
    spy = SpyModel()
    predictions = run_panel(encoded_panel(), config, [spy])
    assert spy.calls == len(predictions) == 24 * 3 * 3
    offsets = [(pd.Period(t, "M") - pd.Period(o, "M")).n for t, o in zip(predictions["target_month"], predictions["forecast_origin"])]
    assert offsets == predictions["horizon"].tolist()


def test_all_models_horizons_and_feature_sets_share_target_months(config):
    predictions = run_panel(random_panel(), config, build_models(config))
    assert list(predictions.columns) == PREDICTION_COLUMNS
    target_sets = predictions.groupby(["county", "model", "feature_set", "horizon"])["target_month"].apply(frozenset)
    assert target_sets.nunique() == 1 and len(target_sets.iloc[0]) == 24
    assert set(predictions["period"]) == {"development", "confirmation"}
    assert (predictions["train_end"] == predictions["rate_info_cutoff"]).all()


ALL_MODELS = ("seasonal_naive", "persistence", "climatology", "linear", "ridge", "sarima", "xgboost")


@pytest.mark.parametrize("horizon", [1, 3, 6])
def test_predictions_ignore_every_value_published_after_the_origin(horizon):
    """Changing anything not yet published at the origin (including the target) must not move any forecast."""
    target = pd.Period("2019-07", "M")
    origin = target - horizon
    config = make_config(horizons=(horizon,), eval_start=str(target), eval_end=str(target), models=ALL_MODELS,
                         model_params={"ridge": {"alpha": 3.0, "log_target": True}, "xgboost": {"log_target": True, "params": {"n_estimators": 40}}})
    panel = random_panel()
    perturbed = {county: frame.copy() for county, frame in panel.items()}
    for frame in perturbed.values():
        frame.loc[frame.index > origin - config.rate.availability_lag, "rate"] *= 50
        for col, lag in config.env_cutoffs.items():
            frame.loc[frame.index > origin - lag, col] += 1_000
    original = run_panel(panel, config, build_models(config))
    changed = run_panel(perturbed, config, build_models(config))
    pd.testing.assert_series_equal(original["predicted_rate"], changed["predicted_rate"])
    assert not np.allclose(original["actual_rate"], changed["actual_rate"])


def test_features_published_at_the_origin_do_change_predictions(config):
    """Guard against a vacuous invariance test: data inside the information set must matter."""
    panel = random_panel()
    perturbed = {county: frame.assign(w1=frame["w1"] + np.where(frame.index <= pd.Period("2016-01", "M"), 5.0, 0.0)) for county, frame in panel.items()}
    single = dataclasses.replace(config, eval_start="2017-01", eval_end="2017-01")
    original = run_panel(panel, single, build_models(single))
    changed = run_panel(perturbed, single, build_models(single))
    environmental = original["feature_set"].isin(["E", "RE"])
    assert not np.allclose(original.loc[environmental, "predicted_rate"], changed.loc[environmental, "predicted_rate"])


def recovery_panel(driver_lag: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(1)
    driver = rng.normal(size=len(MONTHS))
    rate = 10 + 2 * pd.Series(driver, index=MONTHS).shift(driver_lag).fillna(0)
    frame = pd.DataFrame({"rate": rate, "x": driver}, index=MONTHS)
    return {"A": frame}


@pytest.mark.parametrize("availability_lag,horizon,recoverable", [(0, 1, True), (0, 3, True), (0, 6, False), (1, 3, False)])
def test_known_driver_is_recovered_only_when_it_is_available(availability_lag, horizon, recoverable):
    """rate[m] = 10 + 2 x[m-3]: exact forecasts are possible iff x[target-3] is published by the origin."""
    config = make_config(
        horizons=(horizon,), feature_sets=("E",), calendar=False, models=("linear",),
        env_families={"driver": FeatureBlock(("x",), availability_lag=availability_lag, lags=(0, 1, 2))},
    )
    predictions = run_panel(recovery_panel(3), config, build_models(config))
    rmse = np.sqrt(np.mean((predictions["predicted_rate"] - predictions["actual_rate"]) ** 2))
    assert (rmse < 1e-8) if recoverable else (rmse > 0.5)


def test_baselines_follow_their_definitions(config):
    panel = encoded_panel()
    predictions = run_panel(panel, config, build_models(dataclasses.replace(config, models=("seasonal_naive", "persistence", "climatology"))))
    for _, row in predictions.iterrows():
        target, origin = pd.Period(row["target_month"], "M"), pd.Period(row["forecast_origin"], "M")
        if row["model"] == "seasonal_naive":
            assert decode_month(row["predicted_rate"]) == target - 12
        elif row["model"] == "persistence":
            assert decode_month(row["predicted_rate"]) == origin - 1
        else:  # mean of the 10 same-month labels in a 120-month window: their middle month
            labels = pd.period_range(row["train_start"], row["train_end"], freq="M")
            expected = np.mean([panel["A"].at[m, "rate"] for m in labels if m.month == target.month])
            assert row["predicted_rate"] == pytest.approx(expected)


def test_missing_labels_are_dropped_and_missing_actuals_kept(config):
    panel = random_panel()
    panel["A"].loc["2012-01":"2012-06", "rate"] = np.nan
    panel["A"].loc["2017-03", "rate"] = np.nan
    predictions = run_panel(panel, dataclasses.replace(config, horizons=(1,)), build_models(config))
    county_a = predictions[predictions["county"] == "A"]
    affected = county_a[county_a["target_month"] == "2017-05"]
    assert (affected["n_train"] == 120 - 6 - 1).all()
    assert county_a.loc[county_a["target_month"] == "2017-03", "actual_rate"].isna().all()


def test_runs_are_reproducible(config):
    first = run_panel(random_panel(), config, build_models(config))
    second = run_panel(random_panel(), config, build_models(config))
    pd.testing.assert_frame_equal(first, second)


def test_reference_skill_is_zero_for_the_reference(config):
    predictions = run_panel(random_panel(), config, build_models(config))
    summary = summarize(predictions, "seasonal_naive")
    reference = summary[summary["model"] == "seasonal_naive"]
    np.testing.assert_allclose(reference["skill_vs_reference"], 0.0)
    assert set(summary["period"]) == {"development", "confirmation", "all"}


def test_diebold_mariano_direction_and_degenerate_case():
    rng = np.random.default_rng(3)
    base = rng.normal(size=200) ** 2
    statistic, p_value = diebold_mariano(base, base + 0.5 + 0.1 * rng.normal(size=200), horizon=3)
    assert statistic < 0 and p_value < 0.01
    assert np.isnan(diebold_mariano(base, base, horizon=1)[0])
