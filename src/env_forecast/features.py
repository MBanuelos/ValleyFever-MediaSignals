"""Direct-forecast design matrices built from a masked history."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import HarnessConfig
from .imputation import ets_filled_features


def window_mean(series: pd.Series, width: int) -> pd.Series:
    """
    Trailing mean of `width` months computed from each window's own values.

    pandas' rolling mean uses a running sum, so rounding from much older months leaks into every
    later value; that makes ill-conditioned fits (e.g. SARIMAX) depend on data outside the window.
    """
    values = series.to_numpy(float)
    means = np.full(len(values), np.nan)
    if len(values) >= width:
        means[width - 1:] = np.lib.stride_tricks.sliding_window_view(values, width).mean(axis=1)
    return pd.Series(means, index=series.index, name=series.name)


@dataclass(frozen=True)
class Design:
    """
    Rows are pseudo-origins s; X holds what was known at s, y holds the label y[s + h].

    Because the design is built from a history already masked at the real origin t,
    labels later than t - case lag are NaN and can never enter training.
    """

    X: pd.DataFrame
    y: pd.Series
    feature_sets: dict[str, list[str]]
    families: dict[str, str]


def build_design(history: pd.DataFrame, horizon: int, config: HarnessConfig) -> Design:
    """
    Build the horizon-h design from a masked history.

    Every feature at row s is a value of month <= s - (availability lag of its source),
    so training rows respect the same publication delays as the forecast row.
    """
    rate = history[config.target_col]
    rate_lag = config.rate.availability_lag
    columns: dict[str, pd.Series] = {}
    families: dict[str, str] = {}

    shifts = [rate_lag + k for k in config.rate.lags]
    for shift in shifts:
        columns[f"rate_lag{shift}"] = rate.shift(shift)
    # same calendar month one year before the target, only if it is published by s
    seasonal_shift = 12 - horizon
    if config.rate.seasonal_lag and seasonal_shift >= rate_lag and seasonal_shift not in shifts:
        columns["rate_same_month_last_year"] = rate.shift(seasonal_shift)
    rate_columns = list(columns)
    families.update(dict.fromkeys(rate_columns, "rate"))

    env_columns = []
    # rows that can enter this origin's training window or be the forecast row
    origin = history.index[-1]
    needed_rows = history.index[history.index >= origin - rate_lag - horizon - config.window_months + 1]
    for family, block in config.env_families.items():
        for col in block.columns:
            series = history[col]
            if block.forecast_fill == "ets":
                filled = ets_filled_features(series, needed_rows, block.availability_lag, block.fill_lag,
                                             block.lags, block.rolling_means, block.fill_window)
                columns.update(filled)
                families.update(dict.fromkeys(filled, family))
                env_columns += list(filled)
                continue
            names = []
            for k in block.lags:
                names.append(f"{col}_lag{block.availability_lag + k}")
                columns[names[-1]] = series.shift(block.availability_lag + k)
            for width in block.rolling_means:
                names.append(f"{col}_mean{width}_lag{block.availability_lag}")
                columns[names[-1]] = window_mean(series, width).shift(block.availability_lag)
            families.update(dict.fromkeys(names, family))
            env_columns += names

    calendar_columns = []
    if config.calendar:
        target_month = (history.index + horizon).month.to_numpy()
        columns["target_month_sin"] = pd.Series(np.sin(2 * np.pi * target_month / 12), index=history.index)
        columns["target_month_cos"] = pd.Series(np.cos(2 * np.pi * target_month / 12), index=history.index)
        calendar_columns = ["target_month_sin", "target_month_cos"]
        families.update(dict.fromkeys(calendar_columns, "calendar"))

    X = pd.DataFrame(columns, index=history.index)
    y = rate.shift(-horizon).rename("target")
    feature_sets = {
        "R": rate_columns + calendar_columns,
        "E": env_columns + calendar_columns,
        "RE": rate_columns + env_columns + calendar_columns,
    }
    return Design(X=X, y=y, feature_sets=feature_sets, families=families)
