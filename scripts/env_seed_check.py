"""Compare XGBoost RE-vs-E and RE-vs-R differences against seed-to-seed variation (development period).

Usage: python scripts/env_seed_check.py <project_root>  (after running the harness with --models seasonal_naive xgboost --seed N)
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from src.env_forecast.metrics import diebold_mariano  # noqa: E402

runs = []
for folder in (root / "results" / "env").glob("env_baseline_dev_*"):
    config = json.loads((folder / "config.json").read_text())
    if config["models"] == ["seasonal_naive", "xgboost"]:
        frame = pd.read_csv(folder / "predictions.csv")
        runs.append(frame[frame["model"] == "xgboost"].assign(seed=config["seed"]))
p = pd.concat(runs)
seeds = sorted(p["seed"].unique())
print("seeds:", seeds, "| rows per seed:", p.groupby("seed").size().unique())
keys = ["county", "horizon"]

rmse = p.groupby(["seed", "county", "horizon", "feature_set"]).apply(
    lambda g: np.sqrt(np.mean((g.predicted_rate - g.actual_rate) ** 2)), include_groups=False).unstack("feature_set")

# noise band: |RMSE_X(s)/RMSE_X(s') - 1| over seed pairs and the three feature sets, 95th percentile per county x horizon
noise = []
for (county, h), g in rmse.groupby(level=["county", "horizon"]):
    g = g.droplevel(["county", "horizon"])
    ratios = [abs(g.loc[a, fs] / g.loc[b, fs] - 1) for a, b in itertools.combinations(seeds, 2) for fs in ("E", "R", "RE")]
    noise.append({"county": county, "horizon": h, "seed_noise_p95": np.percentile(ratios, 95), "seed_noise_max": max(ratios)})
noise = pd.DataFrame(noise).set_index(keys)

rows = []
ensemble = p.groupby(["county", "horizon", "target_month", "feature_set"]).agg(pred=("predicted_rate", "mean"), actual=("actual_rate", "first")).reset_index()
for a, b in (("RE", "E"), ("RE", "R")):
    change = (rmse[a] / rmse[b] - 1).unstack("seed")
    for (county, h), values in change.iterrows():
        values = values.to_numpy()
        ea = ensemble[(ensemble.county == county) & (ensemble.horizon == h) & (ensemble.feature_set == a)].sort_values("target_month")
        eb = ensemble[(ensemble.county == county) & (ensemble.horizon == h) & (ensemble.feature_set == b)].sort_values("target_month")
        la, lb = (ea.pred.to_numpy() - ea.actual.to_numpy()) ** 2, (eb.pred.to_numpy() - eb.actual.to_numpy()) ** 2
        stat, pval = diebold_mariano(la, lb, int(h))
        band = noise.loc[(county, h), "seed_noise_p95"]
        rows.append({
            "contrast": f"{a}_vs_{b}", "county": county, "horizon": h,
            "mean_rel_change": values.mean(), "seed_sd": values.std(ddof=1), "min": values.min(), "max": values.max(),
            "seeds_same_sign": int(max((values < 0).sum(), (values > 0).sum())),
            "seed_noise_p95": band, "exceeds_noise": abs(values.mean()) > band and (np.sign(values) == np.sign(values.mean())).all(),
            "ensemble_rel_change": np.sqrt(la.mean()) / np.sqrt(lb.mean()) - 1, "ensemble_dm_p": pval,
        })
result = pd.DataFrame(rows)
out = root / "results" / "env" / "xgboost_seed_check"
out.mkdir(parents=True, exist_ok=True)
result.to_csv(out / "seed_check_contrasts.csv", index=False)
rmse.to_csv(out / "seed_check_rmse.csv")
pd.set_option("display.width", 220)
for contrast, g in result.groupby("contrast"):
    print(f"\n=== {contrast} (negative = RE better) ===")
    print(g.drop(columns="contrast").round(3).to_string(index=False))
    s = g.groupby("horizon").agg(
        RE_better_mean=("mean_rel_change", lambda x: (x < 0).sum()), all5_agree=("seeds_same_sign", lambda x: (x == len(seeds)).sum()),
        exceeds_noise_RE_better=("exceeds_noise", lambda x: (x & (g.loc[x.index, "mean_rel_change"] < 0)).sum()),
        exceeds_noise_RE_worse=("exceeds_noise", lambda x: (x & (g.loc[x.index, "mean_rel_change"] > 0)).sum()),
        ens_p05_RE_better=("ensemble_dm_p", lambda x: ((x < 0.05) & (g.loc[x.index, "ensemble_rel_change"] < 0)).sum()),
        median_change=("mean_rel_change", "median"), median_seed_sd=("seed_sd", "median"))
    print(s.round(3).to_string())
print("\nseed noise band (p95 |RMSE ratio - 1| between seeds) median by horizon:", noise.groupby("horizon")["seed_noise_p95"].median().round(3).to_dict())
