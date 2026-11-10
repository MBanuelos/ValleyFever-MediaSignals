"""Origin-consistent forward filling of late-published environmental series."""

from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from threadpoolctl import threadpool_limits

MIN_ETS_MONTHS = 24


@lru_cache(maxsize=200_000)
def _ets_forecast(values: tuple[float, ...], steps: int) -> tuple[float, ...]:
    """Additive-seasonal Holt-Winters forecast (no trend, period 12), clipped at zero; cached by input values."""
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("ignore")
        fitted = ExponentialSmoothing(np.asarray(values), trend=None, seasonal="add", seasonal_periods=12).fit()
        forecast = fitted.forecast(steps)
    return tuple(np.clip(forecast, 0.0, None))


def ets_extend(available: pd.Series, steps: int, window: int) -> np.ndarray | None:
    """
    Last `window` available values followed by `steps` ETS forecasts, or None if too little data.

    Internal gaps inside the window are interpolated from neighbouring available values only.
    """
    recent = available.iloc[-window:].interpolate(limit_area="inside")
    if recent.isna().any() or len(recent) < MIN_ETS_MONTHS:
        return None
    values = tuple(recent.to_numpy(float))
    if steps <= 0:
        return np.asarray(values)
    return np.concatenate([values, _ets_forecast(values, steps)])


def ets_filled_features(series: pd.Series, rows: pd.PeriodIndex, availability_lag: int, fill_lag: int,
                        lags: tuple[int, ...], rolling_means: tuple[int, ...], window: int) -> dict[str, pd.Series]:
    """
    Features from an ETS-filled series, computed separately for every row s in `rows`.

    Row s may only use values for months <= s - availability_lag. The gap up to s - fill_lag is
    forecast from those values, so a training row sees exactly what a forecaster standing at s
    would have imputed; nothing from after s - availability_lag enters any row.
    """
    name = series.name
    names = [f"{name}_etsfill_lag{fill_lag + k}" for k in lags] + [f"{name}_etsfill_mean{w}_lag{fill_lag}" for w in rolling_means]
    output = {feature: pd.Series(np.nan, index=series.index) for feature in names}
    steps = availability_lag - fill_lag
    for row in rows:
        available = series.loc[:row - availability_lag]
        if not len(available) or available.index[-1] != row - availability_lag:
            continue
        extended = ets_extend(available, steps, window)
        if extended is None:
            continue
        # extended[-1] is month row - fill_lag
        for k, feature in zip(lags, names):
            if k < len(extended):
                output[feature].at[row] = extended[-1 - k]
        for w, feature in zip(rolling_means, names[len(lags):]):
            if w <= len(extended):
                output[feature].at[row] = extended[-w:].mean()
    return output
