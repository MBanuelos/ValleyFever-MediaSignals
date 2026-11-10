"""
Permutation feature importance inside the rolling-origin loop.

For every forecast, after the model is fitted, each input group is "permuted": its values in the
forecast inputs are replaced by the same group's values from a randomly drawn training row (for the
LSTM, the whole lookback sequence of that group), and the model predicts again without refitting.
Repeating this `importance_draws` times breaks the link between the group and the target while
keeping the group's distribution. The harness scores these perturbed forecasts against the actual
value, so importance = RMSE(perturbed) / RMSE(original) - 1 on the evaluation targets.

Substitute values come only from the task's training rows, so importance uses exactly the information
the forecast itself was allowed to use.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

from .config import HarnessConfig
from .tasks import ForecastTask

RATE = "__rate__"  # source token for case-rate history


@dataclass(frozen=True)
class ImportanceGroup:
    level: str  # "group" (a whole environmental family, or case history) or "variable" (one source column)
    name: str
    sources: tuple[str, ...]  # environmental source columns, or RATE


def importance_groups(config: HarnessConfig, feature_set: str) -> list[ImportanceGroup]:
    """Groups that exist in a feature set: case history (R, RE), each family and each variable (E, RE)."""
    groups = []
    if feature_set in ("R", "RE"):
        groups.append(ImportanceGroup("group", "case_history", (RATE,)))
    if feature_set in ("E", "RE"):
        for family, block in config.env_families.items():
            groups.append(ImportanceGroup("group", family, tuple(block.columns)))
        for col in config.env_columns:
            groups.append(ImportanceGroup("variable", col, (col,)))
    return groups


def design_columns(columns: list[str], sources: tuple[str, ...]) -> list[str]:
    """Design-matrix columns built from the given sources (lags, rolling means, ETS-filled versions)."""
    selected = []
    for column in columns:
        for source in sources:
            if source == RATE:
                if column.startswith("rate_"):
                    selected.append(column)
            elif column.startswith(f"{source}_") and column[len(source) + 1:].startswith(("lag", "mean", "etsfill")):
                selected.append(column)
    return selected


def draw_rng(task: ForecastTask, group: ImportanceGroup) -> np.random.Generator:
    """Deterministic draws per task and group, independent of run order and parallelism."""
    return np.random.default_rng([task.seed, task.origin.ordinal, task.horizon, zlib.crc32(group.name.encode())])
