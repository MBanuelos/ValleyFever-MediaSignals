"""Rolling-origin evaluation loop shared by every environmental forecasting model."""

from __future__ import annotations

import warnings
from typing import Iterable

import pandas as pd
from joblib import Parallel, delayed

from .config import HarnessConfig
from .data import mask_history
from .features import Design, build_design
from .models import ForecastModel
from .tasks import ForecastTask

PREDICTION_COLUMNS = [
    "county", "model", "feature_set", "horizon", "forecast_origin", "target_month", "period", "seed",
    "actual_rate", "predicted_rate", "train_start", "train_end", "n_train", "rate_info_cutoff", "env_info_cutoff",
]


def evaluation_targets(config: HarnessConfig) -> pd.PeriodIndex:
    return pd.period_range(config.eval_start, config.eval_end, freq="M")


def period_label(target: pd.Period, config: HarnessConfig) -> str:
    if config.confirmation_start is None:
        return "evaluation"
    return "confirmation" if target >= pd.Period(config.confirmation_start, "M") else "development"


def training_rows(design: Design, origin: pd.Period, horizon: int, config: HarnessConfig) -> pd.PeriodIndex:
    """
    Pseudo-origins whose labels are the `window_months` most recent rates published by `origin`.

    Label month = pseudo-origin + horizon must satisfy label <= origin - case lag, so the last
    usable pseudo-origin is origin - case lag - horizon. Rows with an unobserved label are dropped.
    """
    last = origin - config.rate.availability_lag - horizon
    first = last - config.window_months + 1
    rows = design.y.loc[first:last].dropna().index
    if len(rows) and rows.max() + horizon > origin - config.rate.availability_lag:
        raise RuntimeError(f"Training label after the information cutoff at origin {origin}, h={horizon}.")
    return rows


def make_tasks(county: str, frame: pd.DataFrame, origin: pd.Period, horizon: int, model: ForecastModel,
               config: HarnessConfig, history: pd.DataFrame, design: Design, rows: pd.PeriodIndex) -> Iterable[ForecastTask]:
    """One task per feature set for feature-based models, a single labelled task otherwise."""
    feature_sets = config.feature_sets if model.uses_features else (getattr(model, "feature_set_label", "none"),)
    for feature_set in feature_sets:
        columns = design.feature_sets[feature_set] if feature_set != "none" else []
        yield ForecastTask(
            county=county, origin=origin, horizon=horizon, feature_set=feature_set, history=history,
            X_train=design.X.loc[rows, columns], y_train=design.y.loc[rows],
            X_forecast=design.X.loc[[origin], columns], label_months=rows + horizon, config=config, seed=config.seed,
        )


def run_county(county: str, frame: pd.DataFrame, config: HarnessConfig, models: list[ForecastModel]) -> list[dict]:
    """Forecast every target month at every horizon for one county, model and feature set."""
    records = []
    env_lag = min(config.env_cutoffs.values(), default=None)
    for horizon in config.horizons:
        for target in evaluation_targets(config):
            origin = target - horizon
            if origin not in frame.index or target not in frame.index:
                continue
            history = mask_history(frame, origin, config)
            design = build_design(history, horizon, config)
            rows = training_rows(design, origin, horizon, config)
            if len(rows) < config.min_train_rows:
                warnings.warn(f"{county} origin {origin} h={horizon}: {len(rows)} training rows, skipped.")
                continue
            base = {
                "county": county, "horizon": horizon, "forecast_origin": str(origin), "target_month": str(target),
                "period": period_label(target, config), "seed": config.seed,
                "actual_rate": float(frame.at[target, config.target_col]),
                "train_start": str(rows.min() + horizon), "train_end": str(rows.max() + horizon), "n_train": len(rows),
                "rate_info_cutoff": str(origin - config.rate.availability_lag),
            }
            for model in models:
                for task in make_tasks(county, frame, origin, horizon, model, config, history, design, rows):
                    uses_env = task.feature_set in ("E", "RE") and env_lag is not None
                    records.append({
                        **base, "model": model.name, "feature_set": task.feature_set,
                        "predicted_rate": float(model.fit_predict(task)),
                        "env_info_cutoff": str(origin - env_lag) if uses_env else "",
                    })
    return records


def run_panel(panel: dict[str, pd.DataFrame], config: HarnessConfig, models: list[ForecastModel], n_jobs: int = 1) -> pd.DataFrame:
    """
    Forecast every target month at every horizon for every county, model and feature set.

    Counties are independent, so n_jobs > 1 runs them in parallel processes (each gets its own
    copy of the models); results are identical to a serial run. Returns one row per forecast with
    the columns in PREDICTION_COLUMNS, sorted deterministically. Predictions are on the rate scale.
    """
    if n_jobs == 1:
        per_county = [run_county(county, frame, config, models) for county, frame in panel.items()]
    else:
        per_county = Parallel(n_jobs=n_jobs)(delayed(run_county)(county, frame, config, models) for county, frame in panel.items())
    predictions = pd.DataFrame([record for records in per_county for record in records], columns=PREDICTION_COLUMNS)
    sort_keys = ["county", "model", "feature_set", "horizon", "target_month"]
    return predictions.sort_values(sort_keys).reset_index(drop=True)
