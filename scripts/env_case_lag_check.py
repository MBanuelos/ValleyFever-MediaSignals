"""Development-period sensitivity of skill and RE-vs-E / RE-vs-R contrasts to the case-rate availability lag.

Usage: python scripts/env_case_lag_check.py <project_root>  (after running the harness with --development-only --case-lag 0/1/3)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
runs = {}
for folder in (root / "results" / "env").glob("env_baseline*dev_*"):
    config = json.loads((folder / "config.json").read_text())
    full = ["seasonal_naive", "persistence", "climatology", "linear", "ridge", "sarima", "xgboost"]
    if config["models"] == full and config["seed"] == 42:
        runs[config["rate"]["availability_lag"]] = folder
print({lag: folder.name for lag, folder in sorted(runs.items())})

metrics = pd.concat([pd.read_csv(f / "metrics.csv").assign(case_lag=lag) for lag, f in runs.items()])
contrasts = pd.concat([pd.read_csv(f / "contrasts.csv").assign(case_lag=lag) for lag, f in runs.items()])
predictions = pd.concat([pd.read_csv(f / "predictions.csv").assign(case_lag=lag) for lag, f in runs.items()])
dev = metrics[metrics.period == "development"]
c = contrasts[contrasts.period == "development"].reset_index(drop=True)

snaive = predictions[predictions.model == "seasonal_naive"].pivot_table(index=["county", "horizon", "target_month"], columns="case_lag", values="predicted_rate")
print("seasonal naive identical across lags:", bool(np.allclose(snaive.sub(snaive.iloc[:, 0], axis=0).fillna(0), 0)))

pd.set_option("display.width", 200)
skill = dev.pivot_table(index=["model", "feature_set"], columns=["horizon", "case_lag"], values="skill_vs_reference", aggfunc="median")
print("\nMedian skill vs seasonal naive across 8 counties (columns: horizon, case lag)\n", skill.round(3).to_string())

positive = dev.assign(pos=dev.skill_vs_reference > 0).pivot_table(index=["model", "feature_set"], columns=["horizon", "case_lag"], values="pos", aggfunc="sum")
print("\nCounties (of 8) with positive skill\n", positive.astype(int).to_string())

summary = c.groupby(["model", "contrast", "horizon", "case_lag"]).agg(
    RE_better=("relative_rmse_change", lambda x: int((x < 0).sum())),
    sig_RE_better=("relative_rmse_change", lambda x: int(((x < 0) & (c.loc[x.index, "dm_p_value"] < 0.05)).sum())),
    sig_RE_worse=("relative_rmse_change", lambda x: int(((x > 0) & (c.loc[x.index, "dm_p_value"] < 0.05)).sum())),
    median_change=("relative_rmse_change", "median"))
for model in ("xgboost", "ridge"):
    print(f"\n=== {model}: contrasts by horizon x case lag (negative change = RE better) ===")
    print(summary.loc[model].unstack("case_lag").round(3).to_string())

e_vs_clim = dev[dev.feature_set.isin(["E"]) | (dev.model == "climatology")]
print("\nE-only vs climatology (both use no recent case data at forecast time): median RMSE ratio E/climatology by horizon, lag")
rm = dev.pivot_table(index=["county", "horizon", "case_lag"], columns=["model", "feature_set"], values="rmse")
for model in ("xgboost", "ridge"):
    ratio = (rm[(model, "E")] / rm[("climatology", "none")]).groupby(["horizon", "case_lag"]).agg(["median", lambda x: int((x < 1).sum())])
    ratio.columns = ["median_ratio", "counties_E_better"]
    print(model, "\n", ratio.round(3).unstack("case_lag").to_string())

out = root / "results" / "env" / "case_lag_sensitivity"
out.mkdir(parents=True, exist_ok=True)
summary.reset_index().to_csv(out / "contrast_summary_by_case_lag.csv", index=False)
c.to_csv(out / "contrasts_by_case_lag.csv", index=False)
dev.to_csv(out / "metrics_by_case_lag.csv", index=False)
