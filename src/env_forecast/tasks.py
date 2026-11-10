"""The single object every model receives from the harness."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import HarnessConfig


@dataclass(frozen=True)
class ForecastTask:
    """
    One forecast: predict `target_col` for `target_month` from information available at `origin`.

    Tabular (direct) models use X_train / y_train / X_forecast, which are already restricted to
    the rolling window and the requested feature set. Sequence models (SARIMAX, LSTM) may build
    their own inputs from `history`, which is masked at the origin: rows end at the origin and
    every column is blank after its publication cutoff. Nothing outside this object is visible
    to a model, so information cutoffs hold by construction.

    label_months[i] is the month whose rate y_train[i] is (pseudo-origin + horizon).
    """

    county: str
    origin: pd.Period
    horizon: int
    feature_set: str
    history: pd.DataFrame
    X_train: pd.DataFrame
    y_train: pd.Series
    X_forecast: pd.DataFrame
    label_months: pd.PeriodIndex
    config: HarnessConfig
    seed: int

    @property
    def target_month(self) -> pd.Period:
        return self.origin + self.horizon

    @property
    def target_col(self) -> str:
        return self.config.target_col
