"""
Does adding fungicide/rodenticide features improve development-period forecasts?

Compares each pesticide variant (stale36, ets_fill, oracle1) against the no-pesticide run on identical
targets: seed-averaged RMSE change and Diebold-Mariano test per model x feature set x county x horizon,
plus whether every individual seed agrees on the direction.

Usage: python scripts/env_pesticide_check.py <project_root>
(after running configs/pesticide/*.json with --development-only for seeds 42-46)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from src.env_forecast.metrics import diebold_mariano  # noqa: E402

KEYS = ["model", "feature_set", "county", "horizon", "target_month"]
frames = []
for folder in (root / "results" / "env").glob("pest_*_dev_*"):
    config = json.loads((folder / "config.json").read_text())
    variant = config["name"].removeprefix("pest_").removesuffix("_dev")
    frames.append(pd.read_csv(folder / "predictions.csv").assign(variant=variant, seed=config["seed"]))
p = pd.concat(frames, ignore_index=True)
p = p[p["model"].isin(["ridge", "xgboost"]) & p["feature_set"].isin(["E", "RE"])]
print(p.groupby("variant")["seed"].unique().apply(sorted).to_dict())

ensemble = p.groupby(["variant", *KEYS]).agg(pred=("predicted_rate", "mean"), actual=("actual_rate", "first")).reset_index()
by_seed = p.assign(sq=(p.predicted_rate - p.actual_rate) ** 2).groupby(["variant", "seed", *KEYS[:-1]])["sq"].mean().pow(0.5)

rows = []
base = ensemble[ensemble.variant == "none"].set_index(KEYS)
for variant in [v for v in ("stale36", "ets_fill", "oracle1") if v in set(ensemble.variant)]:
    test = ensemble[ensemble.variant == variant].set_index(KEYS)
    joined = test.join(base, rsuffix="_none", how="inner").reset_index()
    for (model, fs, county, h), g in joined.groupby(KEYS[:-1]):
        g = g.sort_values("target_month")
        la, lb = (g.pred - g.actual) ** 2, (g.pred_none - g.actual) ** 2
        stat, pval = diebold_mariano(la.to_numpy(), lb.to_numpy(), int(h))
        seeds = by_seed.loc[variant].xs((model, fs, county, h), level=["model", "feature_set", "county", "horizon"])
        seeds_none = by_seed.loc["none"].xs((model, fs, county, h), level=["model", "feature_set", "county", "horizon"])
        per_seed = (seeds / seeds_none.reindex(seeds.index) - 1).to_numpy()
        change = np.sqrt(la.mean()) / np.sqrt(lb.mean()) - 1
        rows.append({"variant": variant, "model": model, "feature_set": fs, "county": county, "horizon": h,
                     "n": len(g), "rmse_with": np.sqrt(la.mean()), "rmse_without": np.sqrt(lb.mean()),
                     "rel_change": change, "dm_p": pval,
                     "seeds_agree": bool((np.sign(per_seed) == np.sign(change)).all()) if model == "xgboost" else True})
result = pd.DataFrame(rows)

out = root / "results" / "env" / "pesticide_check"
out.mkdir(parents=True, exist_ok=True)
result.to_csv(out / "pesticide_variant_vs_none.csv", index=False)

summary = result.groupby(["model", "feature_set", "variant", "horizon"]).apply(lambda g: pd.Series({
    "better": int((g.rel_change < 0).sum()),
    "better_all_seeds": int(((g.rel_change < 0) & g.seeds_agree).sum()),
    "sig_better": int(((g.rel_change < 0) & (g.dm_p < 0.05)).sum()),
    "sig_worse": int(((g.rel_change > 0) & (g.dm_p < 0.05)).sum()),
    "median_change": g.rel_change.median(),
}), include_groups=False)
summary.to_csv(out / "pesticide_summary.csv")
pd.set_option("display.width", 200)
print("\nWith-pesticide vs no-pesticide (negative change = pesticides help), counts out of 8 counties\n")
print(summary.round(3).to_string())
