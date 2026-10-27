"""Configuration objects for the environmental rolling-origin forecasting harness."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

FEATURE_SETS = ("R", "E", "RE")


@dataclass(frozen=True)
class FeatureBlock:
    """
    A family of environmental columns that share one publication delay.

    At forecast origin t, column `c` is only known for months <= t - availability_lag.
    Lag k produces the feature c[t - availability_lag - k]; a rolling mean of width w
    averages the w most recent available months ending at t - availability_lag.

    With forecast_fill="ets", the gap between the last available month and t - fill_lag is
    filled by an additive Holt-Winters forecast fitted on the last `fill_window` available
    months, separately for every row's own information set; lags and rolling means are then
    taken relative to t - fill_lag on the filled series.
    """

    columns: tuple[str, ...]
    availability_lag: int = 1
    lags: tuple[int, ...] = (0,)
    rolling_means: tuple[int, ...] = ()
    forecast_fill: str | None = None
    fill_lag: int = 1
    fill_window: int = 72


@dataclass(frozen=True)
class RateSpec:
    """Case-rate history features; availability_lag is the case reporting delay in months."""

    availability_lag: int = 1
    lags: tuple[int, ...] = tuple(range(12))
    seasonal_lag: bool = True


@dataclass(frozen=True)
class HarnessConfig:
    """
    Every setting that defines an experiment.

    Evaluation is target-based: each month in [eval_start, eval_end] is forecast at every
    horizon h from origin = target - h, so all horizons, models and feature sets are scored
    on identical target months. Targets on or after `confirmation_start` are labelled
    "confirmation" and must not be used for model or feature selection.
    """

    name: str = "env_experiment"
    data_path: str = "data/processed/env_vf_aggregate/aggregate_all_counties_2001_2024.csv"
    counties: tuple[str, ...] = ()
    target_col: str = "rate_per_100k"
    county_col: str = "County"
    date_col: str = "Year-Month"
    panel_start: str | None = None
    panel_end: str | None = None
    window_months: int = 120
    min_train_rows: int = 36
    horizons: tuple[int, ...] = (1, 3, 6)
    eval_start: str = "2017-01"
    eval_end: str = "2024-12"
    eval_step: int = 1
    confirmation_start: str | None = "2021-01"
    rate: RateSpec = RateSpec()
    env_families: dict[str, FeatureBlock] = field(default_factory=dict)
    calendar: bool = True
    ffill_limit: int | None = None
    models: tuple[str, ...] = ("seasonal_naive", "linear")
    model_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    feature_sets: tuple[str, ...] = FEATURE_SETS
    reference_model: str = "seasonal_naive"
    contrasts: tuple[tuple[str, str], ...] = (("RE", "E"), ("RE", "R"))
    seed: int = 42
    output_dir: str = "results/env"

    def __post_init__(self) -> None:
        if self.window_months < 1 or any(h < 1 for h in self.horizons) or self.eval_step < 1:
            raise ValueError("window_months, eval_step and every horizon must be positive.")
        if self.rate.availability_lag < 0 or any(k < 0 for k in self.rate.lags):
            raise ValueError("Rate availability lag and lags must be non-negative.")
        unknown = set(self.feature_sets) - set(FEATURE_SETS)
        if unknown:
            raise ValueError(f"Unknown feature sets {unknown}; use {FEATURE_SETS}.")
        seen: set[str] = set()
        for family, block in self.env_families.items():
            if block.availability_lag < 0 or any(k < 0 for k in block.lags) or any(w < 1 for w in block.rolling_means):
                raise ValueError(f"Invalid lags in environmental family {family!r}.")
            if block.forecast_fill not in (None, "ets") or (block.forecast_fill and not 0 <= block.fill_lag < block.availability_lag):
                raise ValueError(f"Family {family!r}: forecast_fill must be None or 'ets' with 0 <= fill_lag < availability_lag.")
            overlap = seen.intersection(block.columns)
            if overlap or self.target_col in block.columns:
                raise ValueError(f"Column(s) {overlap or self.target_col} appear in more than one block.")
            seen.update(block.columns)

    @property
    def env_columns(self) -> list[str]:
        """All environmental source columns, in family order."""
        return [col for block in self.env_families.values() for col in block.columns]

    @property
    def env_cutoffs(self) -> dict[str, int]:
        """Publication delay (months) for every environmental source column."""
        return {col: block.availability_lag for block in self.env_families.values() for col in block.columns}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        """Stable short hash of the full configuration, used in the experiment id."""
        settings = self.to_dict()
        if settings["eval_step"] == 1:
            del settings["eval_step"]  # keeps ids of configs written before eval_step existed
        payload = json.dumps(settings, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:8]

    @property
    def experiment_id(self) -> str:
        return f"{self.name}_{self.config_hash()}"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "HarnessConfig":
        raw = dict(raw)
        raw.pop("_comment", None)
        if "rate" in raw:
            rate = dict(raw["rate"])
            rate["lags"] = tuple(rate.get("lags", RateSpec.lags))
            raw["rate"] = RateSpec(**rate)
        if "env_families" in raw:
            raw["env_families"] = {
                name: FeatureBlock(
                    columns=tuple(block["columns"]),
                    availability_lag=block.get("availability_lag", 1),
                    lags=tuple(block.get("lags", (0,))),
                    rolling_means=tuple(block.get("rolling_means", ())),
                    forecast_fill=block.get("forecast_fill"),
                    fill_lag=block.get("fill_lag", 1),
                    fill_window=block.get("fill_window", 72),
                )
                for name, block in raw["env_families"].items()
                if block is not None
            }
        for key in ("counties", "horizons", "models", "feature_sets"):
            if key in raw:
                raw[key] = tuple(raw[key])
        if "contrasts" in raw:
            raw["contrasts"] = tuple(tuple(pair) for pair in raw["contrasts"])
        return cls(**raw)

    @classmethod
    def from_json(cls, path: str | Path) -> "HarnessConfig":
        return cls.from_dict(load_config_dict(path))


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base; nested dicts merge, a None value removes the key."""
    merged = dict(base)
    for key, value in override.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config_dict(path: str | Path) -> dict[str, Any]:
    """Read a JSON config; `"extends": "<relative path>"` inherits from another config file first."""
    path = Path(path)
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    parent = raw.pop("extends", None)
    raw.pop("_comment", None)
    return merge_config(load_config_dict(path.parent / parent), raw) if parent else raw
