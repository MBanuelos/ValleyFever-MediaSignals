"""
Pooled (all-county) significance for the paper contrasts, from saved predictions (no refitting).

Statistic: the geometric-mean RMSE ratio across counties, exp(mean_c log(RMSE_a,c / RMSE_b,c)) - 1.
Negative means A is better. Using the log ratio keeps high-incidence counties (Kern) from dominating.

Uncertainty: a circular moving-block bootstrap over target months (block length 6 by default, at least
the 6-month horizon, so overlapping h-step errors stay together). The same resampled months are used for
every county, which preserves cross-county correlation (shared statewide shocks). Reports the 95%
percentile interval, a two-sided bootstrap p-value, Holm-adjusted p-values within each period, and a
sign test on the number of counties where A is better.

Usage: python scripts/env_pooled_significance.py <project_root> [--name env_main_tuned] [--block 6] [--draws 4000]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_plots import load  # noqa: E402  (seed-averaged predictions, same rule as the comparison script)

CONTRASTS = [
    (("xgboost", "RE"), ("xgboost", "E"), "XGBoost: RE vs E"),
    (("lr", "RE"), ("lr", "E"), "LR: RE vs E"),
    (("lstm", "RE"), ("lstm", "E"), "LSTM: RE vs E"),
    (("xgboost", "RE"), ("xgboost", "R"), "XGBoost: RE vs R"),
    (("lr", "RE"), ("lr", "R"), "LR: RE vs R"),
    (("lstm", "RE"), ("lstm", "R"), "LSTM: RE vs R"),
    (("sarimax", "RE"), ("sarima", "R"), "SARIMAX vs SARIMA"),
    (("xgboost", "RE"), ("lr", "R"), "XGBoost RE vs AR(12)"),
    (("xgboost", "RE"), ("sarimax", "RE"), "XGBoost RE vs SARIMAX"),
    (("lstm", "E"), ("lr", "E"), "LSTM E vs LR E"),
    (("xgboost", "RE"), ("seasonal_naive", "none"), "XGBoost RE vs seasonal naive"),
    (("lstm", "E"), ("seasonal_naive", "none"), "LSTM E vs seasonal naive"),
]


def squared_errors(ens, model, feature_set, period, horizon):
    """Months x counties matrix of squared errors."""
    s = ens[(ens.model == model) & (ens.feature_set == feature_set) & (ens.period == period) & (ens.horizon == horizon)]
    return s.assign(se=(s.pred - s.actual) ** 2).pivot(index="target_month", columns="county", values="se")


def block_indices(n, block, draws, rng):
    """Circular moving-block bootstrap: `draws` index vectors of length n."""
    starts = rng.integers(0, n, size=(draws, -(-n // block)))
    return ((starts[:, :, None] + np.arange(block)) % n).reshape(draws, -1)[:, :n]


def pooled_stat(se_a, se_b):
    """Mean over counties of log RMSE ratio; works on (..., months, counties) arrays."""
    return np.mean(0.5 * (np.log(se_a.mean(axis=-2)) - np.log(se_b.mean(axis=-2))), axis=-1)


def holm(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--name", default="env_main_tuned")
    parser.add_argument("--block", type=int, default=6)
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()
    ens = load(args.root, args.name)
    rng = np.random.default_rng(args.seed)
    rows = []
    for period in ("validation", "test"):
        for horizon in (1, 3, 6):
            for a, b, label in CONTRASTS:
                se_a, se_b = squared_errors(ens, *a, period, horizon), squared_errors(ens, *b, period, horizon)
                months = se_a.index.intersection(se_b.index)
                counties = se_a.columns.intersection(se_b.columns)
                A = se_a.loc[months, counties].to_numpy()
                B = se_b.loc[months, counties].to_numpy()
                keep = ~(np.isnan(A).any(axis=1) | np.isnan(B).any(axis=1))
                A, B = A[keep], B[keep]
                estimate = pooled_stat(A, B)
                idx = block_indices(len(A), args.block, args.draws, rng)
                boot = pooled_stat(A[idx], B[idx])
                lo, hi = np.percentile(boot, [2.5, 97.5])
                p_boot = min(1.0, 2 * min((boot >= 0).mean(), (boot <= 0).mean()))
                per_county = 0.5 * (np.log(A.mean(axis=0)) - np.log(B.mean(axis=0)))
                wins = int((per_county < 0).sum())
                rows.append({
                    "period": period, "horizon": horizon, "comparison": label, "n_months": len(A), "n_counties": len(counties),
                    "pooled_change": np.expm1(estimate), "ci_low": np.expm1(lo), "ci_high": np.expm1(hi),
                    "p_bootstrap": p_boot, "a_better_counties": wins,
                    "sign_test_p": stats.binomtest(wins, len(counties), 0.5).pvalue,
                })
    table = pd.DataFrame(rows)
    table["p_holm"] = table.groupby("period")["p_bootstrap"].transform(holm)
    out = args.root / "results" / "env" / "pooled_significance"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"pooled_contrasts_{args.name}_block{args.block}.csv", index=False)

    def fmt(r):
        star = "*" if r.p_holm < 0.05 else ""
        return f"{100 * r.pooled_change:+.0f}% [{100 * r.ci_low:+.0f}, {100 * r.ci_high:+.0f}] {r.a_better_counties}/8{star}"

    pd.set_option("display.width", 250)
    for period in ("validation", "test"):
        t = table[table.period == period].assign(cell=lambda d: d.apply(fmt, axis=1))
        print(f"\n{period}: pooled RMSE change, 95% block-bootstrap CI, counties where A better (* = Holm p < 0.05)")
        print(t.pivot(index="comparison", columns="horizon", values="cell").reindex([c[2] for c in CONTRASTS]).to_string())


if __name__ == "__main__":
    main()
