"""Error summaries and paired forecast comparisons on identical target months."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

GROUP_KEYS = ["county", "model", "feature_set", "horizon", "period"]
PAIR_KEYS = ["county", "horizon", "target_month"]


def _with_all_period(predictions: pd.DataFrame) -> pd.DataFrame:
    """Append a copy of every row with period='all' so each period and the full span are summarised."""
    return pd.concat([predictions, predictions.assign(period="all")], ignore_index=True)


def summarize(predictions: pd.DataFrame, reference_model: str = "seasonal_naive") -> pd.DataFrame:
    """
    Pooled RMSE / MAE / bias per county, model, feature set, horizon and period.

    Skill = 1 - RMSE / RMSE_reference, with both RMSEs computed on the same target months
    (rows where either forecast or the actual is missing are dropped from both).
    """
    reference = predictions.loc[predictions["model"] == reference_model, PAIR_KEYS + ["predicted_rate"]]
    paired = predictions.merge(reference.rename(columns={"predicted_rate": "reference_rate"}), on=PAIR_KEYS, how="left")
    paired = _with_all_period(paired.dropna(subset=["actual_rate", "predicted_rate"]))

    def _row(group: pd.DataFrame) -> pd.Series:
        error = group["predicted_rate"] - group["actual_rate"]
        both = group.dropna(subset=["reference_rate"])
        rmse_model = np.sqrt(np.mean((both["predicted_rate"] - both["actual_rate"]) ** 2)) if len(both) else np.nan
        rmse_reference = np.sqrt(np.mean((both["reference_rate"] - both["actual_rate"]) ** 2)) if len(both) else np.nan
        return pd.Series({
            "n": len(group), "rmse": np.sqrt(np.mean(error ** 2)), "mae": np.mean(np.abs(error)), "bias": np.mean(error),
            "n_paired_reference": len(both), "rmse_reference": rmse_reference,
            "skill_vs_reference": 1 - rmse_model / rmse_reference if rmse_reference > 0 else np.nan,
        })

    summary = paired.groupby(GROUP_KEYS, sort=True).apply(_row, include_groups=False).reset_index()
    summary["reference_model"] = reference_model
    return summary


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, horizon: int) -> tuple[float, float]:
    """
    Diebold-Mariano test of equal expected loss with the Harvey-Leybourne-Newbold correction.

    Uses a Newey-West (rectangular, h-1 lags) long-run variance because direct h-step errors
    from consecutive origins overlap. Negative statistic means A has lower loss than B.
    Returns (statistic, two-sided p-value from t with n-1 df).
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    if n < 3:
        return np.nan, np.nan
    centered = d - d.mean()
    autocov = [centered @ centered / n] + [centered[k:] @ centered[:-k] / n for k in range(1, min(horizon, n))]
    long_run_variance = autocov[0] + 2 * sum(autocov[1:])
    if long_run_variance <= 0:
        long_run_variance = autocov[0]
    if long_run_variance <= 0:
        return np.nan, np.nan
    statistic = d.mean() / np.sqrt(long_run_variance / n)
    statistic *= np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    return float(statistic), float(2 * stats.t.sf(abs(statistic), df=n - 1))


def paired_contrasts(predictions: pd.DataFrame, contrasts: tuple[tuple[str, str], ...]) -> pd.DataFrame:
    """
    Compare feature sets A vs B for each feature-based model on identical target months.

    Reports RMSE of each, relative RMSE change (A/B - 1; negative means A is better) and the
    DM test on squared errors, per county, horizon and period.
    """
    rows = []
    featured = predictions[predictions["feature_set"] != "none"].dropna(subset=["actual_rate", "predicted_rate"])
    for model, model_rows in featured.groupby("model"):
        for set_a, set_b in contrasts:
            a = model_rows[model_rows["feature_set"] == set_a]
            b = model_rows[model_rows["feature_set"] == set_b]
            merged = a.merge(b[PAIR_KEYS + ["predicted_rate"]], on=PAIR_KEYS, suffixes=("_a", "_b"))
            for (county, horizon, period), group in _with_all_period(merged).groupby(["county", "horizon", "period"]):
                group = group.sort_values("target_month")
                loss_a = (group["predicted_rate_a"] - group["actual_rate"]) ** 2
                loss_b = (group["predicted_rate_b"] - group["actual_rate"]) ** 2
                statistic, p_value = diebold_mariano(loss_a.to_numpy(), loss_b.to_numpy(), int(horizon))
                rmse_a, rmse_b = np.sqrt(loss_a.mean()), np.sqrt(loss_b.mean())
                rows.append({
                    "model": model, "contrast": f"{set_a}_vs_{set_b}", "county": county, "horizon": horizon, "period": period,
                    "n": len(group), "rmse_a": rmse_a, "rmse_b": rmse_b,
                    "relative_rmse_change": rmse_a / rmse_b - 1 if rmse_b > 0 else np.nan,
                    "dm_statistic": statistic, "dm_p_value": p_value,
                })
    return pd.DataFrame(rows)
