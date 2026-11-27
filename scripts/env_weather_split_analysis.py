"""
Weather group split: ablation and permutation importance for humidity, wind (speed + run), wind events and
soil temperature. Descriptive follow-up to scripts/env_importance_analysis.py on the frozen env_main_tuned
models; nothing here feeds back into model or feature choice.

  ablation : ablate_no_<subgroup> runs (configs/ablation/no_{humidity,wind,wind_events,soil_temp}.json) vs
             env_main_tuned, same seeds. The whole-weather ablation (ablate_no_weather) is included for reference.
  pfi      : pfi_weather_split runs (configs/importance/pfi_weather_split.json). Same inputs and forecasts as
             env_main_tuned_pfi; only the importance groups differ.

Same pooled statistic and bootstrap as env_importance_analysis.py. Writes to results/env/importance/weather_split/.

Usage: python scripts/env_weather_split_analysis.py <project_root> [--draws 2000]
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import env_importance_analysis as base  # noqa: E402
from env_plots import INK, load  # noqa: E402

SUBGROUPS = ["humidity", "wind", "wind_events", "soil_temp"]
LABELS = {"humidity": "Humidity", "wind": "Wind (speed, run)", "wind_events": "Wind events",
          "soil_temp": "Soil temperature", "weather": "Weather (all five)", "precipitation": "Precipitation",
          "air_quality": "Air quality", "wildfire": "Wildfire", "case_history": "Case history"}


def ablation(root, rng, draws):
    base.GROUPS = SUBGROUPS + ["weather"]
    return base.ablation(root, load(root, "env_main_tuned"), rng, draws)


def permutation_importance(root, rng, draws):
    original = Path.glob

    def glob(self, pattern):  # point the base loader at the split runs
        return original(self, "pfi_weather_split_*" if pattern == "env_main_tuned_pfi_*" else pattern)

    Path.glob = glob
    try:
        return base.permutation_importance(root, rng, draws)
    finally:
        Path.glob = original


def figure(table, order, xlabel, title, path):
    base.GROUP_LABELS.update(LABELS)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for row, period in enumerate(base.PERIODS):
        for col, h in enumerate((1, 3, 6)):
            ax = axes[row, col]
            base.forest(ax, table[(table.period == period) & (table.horizon == h)], "group", order)
            ax.set_title(f"{base.PERIODS[period]}, {h} month{'s' if h > 1 else ''} ahead", loc="left")
    for ax in axes[1]:
        ax.set_xlabel(xlabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.95))
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261110)
    args = parser.parse_args()
    out = args.root / "results" / "env" / "importance" / "weather_split"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    abl = ablation(args.root, rng, args.draws)
    pfi, per_county = permutation_importance(args.root, rng, args.draws)
    abl.to_csv(out / "ablation.csv", index=False)
    pfi.to_csv(out / "pfi_pooled.csv", index=False)
    per_county.to_csv(out / "pfi_by_county.csv", index=False)

    figure(abl, SUBGROUPS + ["weather"], "RMSE change when the subgroup is removed (%)\nhelps →",
           "Weather split, ablation (geometric mean over 8 counties, 95% block-bootstrap CI)",
           out / "fig_weather_split_ablation.png")
    groups = pfi[pfi.level == "group"]
    figure(groups, ["case_history", "precipitation"] + SUBGROUPS + ["air_quality", "wildfire"],
           "RMSE change when the group is permuted (%)\nmodel relies on it →",
           "Weather split, permutation importance (no refitting; geometric mean over 8 counties, 95% CI)",
           out / "fig_weather_split_pfi.png")

    def cell(r):
        return f"{100 * r.change:+.0f}% [{100 * r.ci_low:+.0f},{100 * r.ci_high:+.0f}]"

    pd.set_option("display.width", 250)
    for name, t in (("ABLATION (+ = subgroup helped)", abl),
                    ("PERMUTATION IMPORTANCE (+ = model relies on it)", groups[groups.group.isin(SUBGROUPS)])):
        print(f"\n{name}: pooled RMSE change, 95% CI")
        t = t.assign(cell=t.apply(cell, axis=1))
        print(t.pivot_table(index=["period", "model", "feature_set", "group"], columns="horizon",
                            values="cell", aggfunc="first").to_string())
    print("\nPFI by county, LSTM, mean over horizons (%):")
    v = per_county[(per_county.level == "group") & per_county.group.isin(SUBGROUPS) & (per_county.model == "lstm")]
    print((100 * v.groupby(["period", "feature_set", "group", "county"]).change.mean().unstack("county")).round(0).to_string())


if __name__ == "__main__":
    main()
