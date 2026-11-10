"""
Which environmental groups and variables matter, and how results change with the case-data delay.
Built only from saved harness outputs (no refitting). Writes tables and figures to results/env/importance/.

  ablation : RMSE change when one group is removed and the model retrained (ablate_no_<group> runs vs
             env_main_tuned, same seeds). Positive = the forecast got worse without the group, so it helped.
  pfi      : permutation importance from env_main_tuned_pfi/importance.csv. RMSE change when a group or
             variable is replaced by values from random training rows, without refitting.
  caselag  : env_main_tuned with case delay 0 and 3 months (env_main_tuned_caselag<L>) vs the main run (1 month).

Pooled values are geometric means over counties with a 95% moving-block bootstrap interval (block 6,
same months for every county), as in scripts/env_pooled_significance.py.

Usage: python scripts/env_importance_analysis.py <project_root> [--draws 2000]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_plots import INK, INK_2, SURFACE, load  # noqa: E402  (also applies the shared plot style)
from env_pooled_significance import block_indices, pooled_stat  # noqa: E402

GROUPS = ["precipitation", "weather", "air_quality", "wildfire"]
GROUP_LABELS = {"precipitation": "Precipitation", "weather": "Weather (soil temp., wind, humidity)",
                "air_quality": "Air quality (PM2.5, PM10)", "wildfire": "Wildfire acreage", "case_history": "Case history"}
SERIES = {("xgboost", "RE"): ("XGBoost RE", "#2a78d6"), ("xgboost", "E"): ("XGBoost E", "#eb6834"),
          ("lstm", "RE"): ("LSTM RE", "#1baf7a"), ("lstm", "E"): ("LSTM E", "#eda100")}
PERIODS = {"validation": "Validation 2017–2020", "test": "Test 2021–2024"}


def pooled(A, B, rng, draws, block=6):
    """Geometric-mean RMSE change of A vs B over counties, with a block-bootstrap 95% interval."""
    keep = ~(np.isnan(A).any(axis=1) | np.isnan(B).any(axis=1))
    A, B = A[keep], B[keep]
    idx = block_indices(len(A), block, draws, rng)
    boot = pooled_stat(A[idx], B[idx])
    per_county = 0.5 * (np.log(A.mean(axis=0)) - np.log(B.mean(axis=0)))
    return {"change": np.expm1(pooled_stat(A, B)), "ci_low": np.expm1(np.percentile(boot, 2.5)),
            "ci_high": np.expm1(np.percentile(boot, 97.5)), "counties_worse": int((per_county > 0).sum()),
            "n_counties": A.shape[1]}


def se_matrix(ens, model, feature_set, period, horizon):
    s = ens[(ens.model == model) & (ens.feature_set == feature_set) & (ens.period == period) & (ens.horizon == horizon)]
    return s.assign(se=(s.pred - s.actual) ** 2).pivot(index="target_month", columns="county", values="se")


def paired(a, b):
    months, counties = a.index.intersection(b.index), a.columns.intersection(b.columns)
    return a.loc[months, counties].to_numpy(), b.loc[months, counties].to_numpy()


def ablation(root, full, rng, draws):
    rows = []
    for group in GROUPS:
        try:
            ablated = load(root, f"ablate_no_{group}")
        except SystemExit:
            print(f"ablation: no runs for {group} yet")
            continue
        for (model, fs) in SERIES:
            for period in PERIODS:
                for h in (1, 3, 6):
                    A, B = paired(se_matrix(ablated, model, fs, period, h), se_matrix(full, model, fs, period, h))
                    if A.size:
                        rows.append({"group": group, "model": model, "feature_set": fs, "period": period, "horizon": h,
                                     **pooled(A, B, rng, draws)})
    return pd.DataFrame(rows)


def permutation_importance(root, rng, draws):
    frames = []
    for folder in (root / "results" / "env").glob("env_main_tuned_pfi_*"):
        if (folder / "importance.csv").exists():
            frames.append(pd.read_csv(folder / "importance.csv"))
    if not frames:
        print("pfi: no importance.csv yet")
        return pd.DataFrame(), pd.DataFrame()
    imp = pd.concat(frames, ignore_index=True)
    print("pfi seeds:", sorted(imp.seed.unique()))
    # average over seeds per forecast, then compare squared errors with and without permutation
    key = ["model", "feature_set", "county", "horizon", "target_month", "period", "level", "group"]
    imp = imp.assign(base_se=(imp.predicted_rate - imp.actual_rate) ** 2).groupby(key)[["base_se", "permuted_sq_error"]].mean().reset_index()
    per_county = (imp.groupby(["model", "feature_set", "period", "horizon", "level", "group", "county"])
                     [["base_se", "permuted_sq_error"]].mean()
                     .assign(change=lambda d: np.sqrt(d.permuted_sq_error / d.base_se) - 1).reset_index())
    rows = []
    for (model, fs, period, h, level, group), g in imp.groupby(["model", "feature_set", "period", "horizon", "level", "group"]):
        A = g.pivot(index="target_month", columns="county", values="permuted_sq_error").to_numpy()
        B = g.pivot(index="target_month", columns="county", values="base_se").to_numpy()
        rows.append({"model": model, "feature_set": fs, "period": period, "horizon": h, "level": level, "group": group,
                     **pooled(A, B, rng, draws)})
    return pd.DataFrame(rows), per_county


def case_lag(root, full, rng, draws):
    rows = []
    for lag in (0, 3):
        try:
            lagged = load(root, f"env_main_tuned_caselag{lag}")
        except SystemExit:
            print(f"caselag: no runs for lag {lag} yet")
            continue
        for frame, label in ((lagged, lag),) + (((full, 1),) if lag == 0 else ()):
            ref = frame[frame.model == "seasonal_naive"]
            for (model, fs), _ in frame[frame.model != "seasonal_naive"].groupby(["model", "feature_set"]):
                for period in PERIODS:
                    for h in (1, 3, 6):
                        A, B = paired(se_matrix(frame, model, fs, period, h), se_matrix(ref, "seasonal_naive", "none", period, h))
                        if A.size:
                            r = pooled(A, B, rng, draws)
                            rows.append({"case_lag": label, "model": model, "feature_set": fs, "period": period, "horizon": h,
                                         "skill": -r["change"], "skill_ci_low": -r["ci_high"], "skill_ci_high": -r["ci_low"]})
                        if fs == "RE":
                            A, B = paired(se_matrix(frame, model, "RE", period, h), se_matrix(frame, model, "E", period, h))
                            if A.size:
                                r = pooled(A, B, rng, draws)
                                rows.append({"case_lag": label, "model": model, "feature_set": "RE vs E", "period": period,
                                             "horizon": h, "change": r["change"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
    return pd.DataFrame(rows)


def forest(ax, table, label_col, order, value="change", lo="ci_low", hi="ci_high"):
    """Horizontal dot-and-interval plot; one row per (label, series)."""
    ax.axvline(0, color=INK_2, lw=1, zorder=1)
    offsets = np.linspace(0.27, -0.27, len(SERIES))
    for offset, ((model, fs), (name, color)) in zip(offsets, SERIES.items()):
        t = table[(table.model == model) & (table.feature_set == fs)].set_index(label_col).reindex(order)
        y = np.arange(len(order))[::-1] + offset
        ax.hlines(y, 100 * t[lo], 100 * t[hi], color=color, lw=2, zorder=2)
        ax.scatter(100 * t[value], y, s=40, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3, label=name)
    ax.set_yticks(np.arange(len(order))[::-1], [GROUP_LABELS.get(g, g) for g in order])


def figures(abl, pfi, per_county, out):
    if len(abl):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
        for row, period in enumerate(PERIODS):
            for col, h in enumerate((1, 3, 6)):
                ax = axes[row, col]
                forest(ax, abl[(abl.period == period) & (abl.horizon == h)], "group", GROUPS)
                ax.set_title(f"{PERIODS[period]}, {h} month{'s' if h > 1 else ''} ahead", loc="left")
        for ax in axes[1]:
            ax.set_xlabel("RMSE change when the group is removed (%)\nhelps →")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.95))
        fig.suptitle("Group ablation: how much worse forecasts get without each environmental group "
                     "(geometric mean over 8 counties, 95% block-bootstrap CI)", color=INK, fontsize=12, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        fig.savefig(out / "fig4_group_ablation.png", dpi=200)
        plt.close(fig)
    if len(pfi):
        groups = pfi[pfi.level == "group"]
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
        order = ["case_history"] + GROUPS
        for row, period in enumerate(PERIODS):
            for col, h in enumerate((1, 3, 6)):
                ax = axes[row, col]
                forest(ax, groups[(groups.period == period) & (groups.horizon == h)], "group", order)
                ax.set_title(f"{PERIODS[period]}, {h} month{'s' if h > 1 else ''} ahead", loc="left")
        for ax in axes[1]:
            ax.set_xlabel("RMSE change when the group is permuted (%)\nmodel relies on it →")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.95))
        fig.suptitle("Permutation importance by group (no refitting; geometric mean over 8 counties, 95% CI)",
                     color=INK, fontsize=12, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        fig.savefig(out / "fig5_permutation_importance_groups.png", dpi=200)
        plt.close(fig)

        # variable-level importance per county: XGBoost RE, test period, mean over horizons
        v = per_county[(per_county.level == "variable") & (per_county.model == "xgboost") & (per_county.feature_set == "RE")]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
        grids = {period: v[v.period == period].groupby(["group", "county"])["change"].mean().unstack("county") * 100
                 for period in PERIODS}
        order = grids["test"].mean(axis=1).sort_values(ascending=False).index  # same row order and color scale in both panels
        limit = max(1.0, np.nanpercentile(np.abs(np.concatenate([g.to_numpy().ravel() for g in grids.values()])), 95))
        for ax, period in zip(axes, PERIODS):
            grid = grids[period].loc[order]
            image = ax.imshow(grid.to_numpy(), cmap="Blues", vmin=0, vmax=limit, aspect="auto")
            ax.set_xticks(range(grid.shape[1]), grid.columns, rotation=35, ha="right")
            ax.set_yticks(range(grid.shape[0]), grid.index)
            ax.grid(False)
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    value = grid.iat[i, j]
                    ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=8,
                            color=SURFACE if value > 0.6 * limit else INK)
            ax.set_title(PERIODS[period], loc="left")
            fig.colorbar(image, ax=ax, fraction=0.04, label="RMSE increase when permuted (%)")
        fig.suptitle("XGBoost RE: permutation importance of each variable by county (mean over 1/3/6-month horizons)",
                     color=INK, fontsize=12, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(out / "fig6_variable_importance_by_county.png", dpi=200)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()
    out = args.root / "results" / "env" / "importance"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    full = load(args.root, "env_main_tuned")
    abl = ablation(args.root, full, rng, args.draws)
    pfi, per_county = permutation_importance(args.root, rng, args.draws)
    lags = case_lag(args.root, full, rng, args.draws)
    for name, table in (("ablation", abl), ("pfi_pooled", pfi), ("pfi_by_county", per_county), ("case_lag", lags)):
        if len(table):
            table.to_csv(out / f"{name}.csv", index=False)
    figures(abl, pfi, per_county, out)
    pd.set_option("display.width", 250)

    def cell(r):
        return f"{100 * r.change:+.0f}% [{100 * r.ci_low:+.0f},{100 * r.ci_high:+.0f}]"

    if len(abl):
        print("\nABLATION: RMSE change when the group is removed (+ = group helped), pooled, 95% CI")
        t = abl.assign(cell=abl.apply(cell, axis=1))
        print(t.pivot_table(index=["period", "model", "feature_set", "group"], columns="horizon", values="cell", aggfunc="first").to_string())
    if len(pfi):
        print("\nPERMUTATION IMPORTANCE (groups): RMSE change when permuted, pooled, 95% CI")
        t = pfi[pfi.level == "group"].assign(cell=lambda d: d.apply(cell, axis=1))
        print(t.pivot_table(index=["period", "model", "feature_set", "group"], columns="horizon", values="cell", aggfunc="first").to_string())
        print("\nPERMUTATION IMPORTANCE (variables, XGBoost RE): pooled RMSE change")
        t = pfi[(pfi.level == "variable") & (pfi.model == "xgboost") & (pfi.feature_set == "RE")]
        print((100 * t.pivot_table(index=["period", "group"], columns="horizon", values="change")).round(1).to_string())
    if len(lags):
        print("\nCASE DELAY: pooled skill vs seasonal naive by case delay (months)")
        s = lags[lags.feature_set != "RE vs E"]
        print(s.pivot_table(index=["period", "model", "feature_set"], columns=["horizon", "case_lag"], values="skill").round(3).to_string())
        print("\nCASE DELAY: RE vs E pooled RMSE change by case delay")
        c = lags[lags.feature_set == "RE vs E"]
        print((100 * c.pivot_table(index=["period", "model"], columns=["horizon", "case_lag"], values="change")).round(0).to_string())


if __name__ == "__main__":
    main()
