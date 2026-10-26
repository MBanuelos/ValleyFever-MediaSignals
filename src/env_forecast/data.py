"""Panel loading and origin-specific information masking."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_validation import validate_data

from .config import HarnessConfig


def panel_from_frame(raw: pd.DataFrame, config: HarnessConfig) -> dict[str, pd.DataFrame]:
    """
    Split a long county-month table into one monthly frame per county.

    Frames are indexed by a monthly PeriodIndex and hold only the target and the configured
    environmental columns. No imputation happens here except an optional *causal* forward
    fill of environmental columns (each month is filled only from earlier months), which
    cannot leak future information. The target is never filled.
    """
    raw = raw.copy()
    raw[config.date_col] = pd.to_datetime(raw[config.date_col])
    columns = [config.target_col, *config.env_columns]
    validate_data(raw, [config.county_col, config.date_col, *columns], config.county_col, config.date_col)

    counties = config.counties or tuple(sorted(raw[config.county_col].unique()))
    missing = set(counties) - set(raw[config.county_col])
    if missing:
        raise ValueError(f"Counties not found in data: {sorted(missing)}")

    panel = {}
    for county in counties:
        frame = raw.loc[raw[config.county_col] == county].sort_values(config.date_col)
        frame = frame.set_index(pd.PeriodIndex(frame[config.date_col], freq="M"))[columns].astype(float)
        frame.index.name = "month"
        frame = frame.loc[config.panel_start:config.panel_end]
        if config.ffill_limit:
            frame[config.env_columns] = frame[config.env_columns].ffill(limit=config.ffill_limit)
        panel[county] = frame
    return panel


def load_panel(config: HarnessConfig, project_root: str | Path = ".") -> dict[str, pd.DataFrame]:
    """Read `config.data_path` (relative to project_root unless absolute) into a county panel."""
    path = Path(config.data_path)
    if not path.is_absolute():
        path = Path(project_root) / path
    return panel_from_frame(pd.read_csv(path), config)


def mask_history(frame: pd.DataFrame, origin: pd.Period, config: HarnessConfig) -> pd.DataFrame:
    """
    Return the information set available at the end of `origin`.

    Rows after the origin are removed. The target is blanked for months later than
    origin - rate availability lag, and each environmental column for months later than
    origin - its family's availability lag. Models only ever receive this masked frame
    (or features built from it), so they cannot read unpublished values.
    """
    history = frame.loc[:origin, [config.target_col, *config.env_columns]].copy()
    history.loc[history.index > origin - config.rate.availability_lag, config.target_col] = np.nan
    for col, lag in config.env_cutoffs.items():
        history.loc[history.index > origin - lag, col] = np.nan
    return history
