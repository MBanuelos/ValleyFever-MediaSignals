"""
Figures for the paper model set, built only from saved harness predictions (no model is refit).

  fig1_forecasts_h{H}.png    actual rate vs forecasts over time, one panel per county; test period shaded
  fig2_skill_by_horizon.png  median skill vs seasonal naive by horizon, validation vs test
  fig3_re_vs_e_by_county.png RE vs E relative RMSE change per county, test period

Seed-dependent models (XGBoost, LSTM) are averaged over seeds; deterministic models use one seed,
matching scripts/env_model_comparison.py.

Usage: python scripts/env_plots.py <project_root> [--name env_main_tuned] [--horizon 3] [--out results/env/figures]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SEEDED = {"xgboost", "lstm"}
KEY = ["county", "horizon", "target_month"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# colour = model (same as fig 3), line style = feature set (solid RE, dashed E), so RE vs E is read within a model
SERIES = {
    ("xgboost", "RE"): ("XGBoost RE", "#2a78d6", "-"),
    ("xgboost", "E"): ("XGBoost E", "#2a78d6", "--"),
    ("lstm", "RE"): ("LSTM RE", "#1baf7a", "-"),
    ("lstm", "E"): ("LSTM E", "#1baf7a", "--"),
    ("lr", "R"): ("AR(12)", "#eb6834", "-"),
}
FORECAST_SERIES = [("xgboost", "E"), ("lstm", "E"), ("lr", "R")]
CONTRAST_MODELS = {"xgboost": ("XGBoost", "#2a78d6"), "lr": ("LR", "#eb6834"), "lstm": ("LSTM", "#1baf7a")}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titlecolor": INK,
    "legend.frameon": False, "lines.linewidth": 2,
})


def load(root: Path, name: str) -> pd.DataFrame:
    """Seed-averaged forecasts for every (model, feature_set, county, horizon, target) of runs named `name`."""
    frames = []
    for folder in (root / "results" / "env").glob(f"{name}_*"):
        config = json.loads((folder / "config.json").read_text())
        if config["name"] == name:
            frames.append(pd.read_csv(folder / "predictions.csv").assign(seed=config["seed"]))
    if not frames:
        raise SystemExit(f"no runs named {name} under results/env/")
    p = pd.concat(frames, ignore_index=True)
    p = p[p["model"].isin(SEEDED) | (p["seed"] == p["seed"].min())]
    return (p.groupby(["model", "feature_set", "period", *KEY])
             .agg(pred=("predicted_rate", "mean"), actual=("actual_rate", "first")).reset_index()
             .dropna(subset=["pred", "actual"]))


def rmse(frame: pd.DataFrame) -> float:
    return float(np.sqrt(((frame.pred - frame.actual) ** 2).mean()))


def skill_table(ens: pd.DataFrame) -> pd.DataFrame:
    """Median (over counties) skill vs seasonal naive per model, feature set, period and horizon."""
    ref = ens[ens.model == "seasonal_naive"].set_index(KEY)["pred"].rename("ref")
    rows = []
    for (model, fs, period, county, h), g in ens.groupby(["model", "feature_set", "period", "county", "horizon"]):
        g = g.join(ref, on=KEY).dropna(subset=["ref"])
        rows.append({"model": model, "feature_set": fs, "period": period, "county": county, "horizon": h,
                     "skill": 1 - rmse(g) / float(np.sqrt(((g.ref - g.actual) ** 2).mean()))})
    return pd.DataFrame(rows).groupby(["model", "feature_set", "period", "horizon"])["skill"].median().reset_index()


def fig_forecasts(ens: pd.DataFrame, horizon: int, out: Path) -> None:
    sub = ens[ens.horizon == horizon]
    counties = sorted(sub.county.unique())
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True)
    for ax, county in zip(axes.flat, counties):
        c = sub[sub.county == county]
        actual = c.drop_duplicates("target_month").set_index("target_month")["actual"].sort_index()
        x = pd.PeriodIndex(actual.index, freq="M").to_timestamp()
        test_months = [t for t, per in c.drop_duplicates("target_month").set_index("target_month")["period"].sort_index().items()
                if per == "test"]
        if test_months:
            ax.axvspan(pd.Period(test_months[0], "M").to_timestamp(), x.max(), color="#f1f0ec", zorder=0, lw=0)
        ax.plot(x, actual.to_numpy(), color=INK, lw=2.2, label="Actual", zorder=3)
        for model, fs in FORECAST_SERIES:
            label, color, style = SERIES[(model, fs)]
            s = c[(c.model == model) & (c.feature_set == fs)].set_index("target_month")["pred"].sort_index()
            ax.plot(pd.PeriodIndex(s.index, freq="M").to_timestamp(), s.to_numpy(), color=color, ls=style, lw=1.6,
                    label=label)
        ax.set_title(county, loc="left")
        ax.set_ylim(bottom=0)
    for ax in axes[:, 0]:
        ax.set_ylabel("Cases per 100k")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 0.965), handlelength=4)
    fig.suptitle(f"Monthly Valley Fever rate: actual vs {horizon}-month-ahead forecasts "
                 "(shaded = held-out test period 2021–2024)", color=INK, fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / f"fig1_forecasts_h{horizon}.png", dpi=200)
    plt.close(fig)


def fig_skill(skill: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, period in zip(axes, ("validation", "test")):
        ax.axhline(0, color=INK_2, lw=1, zorder=1)
        for (model, fs), (label, color, style) in SERIES.items():
            s = skill[(skill.model == model) & (skill.feature_set == fs) & (skill.period == period)].sort_values("horizon")
            ax.plot(s.horizon, s.skill, color=color, ls=style, marker="o" if style == "-" else "s", ms=7, mec=SURFACE,
                    mew=2, label=label, zorder=3)
        ax.set_xticks([1, 3, 6], ["1 month", "3 months", "6 months"])
        ax.set_title(f"{period.capitalize()} ({'2017–2020' if period == 'validation' else '2021–2024'})", loc="left")
    axes[0].set_ylabel("Median skill vs seasonal naive\n(1 − RMSE / RMSE$_{naive}$; >0 = better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 0.93), handlelength=4)
    fig.suptitle("Forecast skill by horizon, median across 8 counties", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out / "fig2_skill_by_horizon.png", dpi=200)
    plt.close(fig)


def fig_re_vs_e(ens: pd.DataFrame, out: Path, period: str = "test") -> None:
    sub = ens[ens.period == period]
    rows = []
    for model in CONTRAST_MODELS:
        for (county, h), g in sub[sub.model == model].groupby(["county", "horizon"]):
            re_, e_ = g[g.feature_set == "RE"].set_index("target_month"), g[g.feature_set == "E"].set_index("target_month")
            shared = re_.index.intersection(e_.index)
            rows.append({"model": model, "county": county, "horizon": h,
                         "change": 100 * (rmse(re_.loc[shared]) / rmse(e_.loc[shared]) - 1)})
    table = pd.DataFrame(rows)
    counties = sorted(table.county.unique(), reverse=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), sharey=True, sharex=True)
    offsets = dict(zip(CONTRAST_MODELS, (0.22, 0, -0.22)))
    for ax, h in zip(axes, (1, 3, 6)):
        ax.axvline(0, color=INK_2, lw=1, zorder=1)
        for model, (label, color) in CONTRAST_MODELS.items():
            t = table[(table.model == model) & (table.horizon == h)].set_index("county").reindex(counties)
            ax.scatter(t.change, np.arange(len(counties)) + offsets[model], s=46, color=color,
                       edgecolor=SURFACE, linewidth=1.5, label=label, zorder=3)
        ax.set_yticks(range(len(counties)), counties)
        ax.set_title(f"{h} month{'s' if h > 1 else ''} ahead", loc="left")
        ax.set_xlabel("RMSE change, RE vs E (%)\n← case history helps")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 0.93))
    fig.suptitle(f"Adding case history to environmental features, by county ({period} period)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out / "fig3_re_vs_e_by_county.png", dpi=200)
    plt.close(fig)
    table.to_csv(out / "fig3_re_vs_e_by_county.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--name", default="env_main_tuned")
    parser.add_argument("--horizon", type=int, nargs="+", default=[1, 3, 6])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or args.root / "results" / "env" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    ens = load(args.root, args.name)
    for h in args.horizon:
        fig_forecasts(ens, h, out)
    skill = skill_table(ens)
    skill.to_csv(out / "fig2_skill_by_horizon.csv", index=False)
    fig_skill(skill, out)
    fig_re_vs_e(ens, out)
    print("wrote", sorted(p.name for p in out.iterdir()))
