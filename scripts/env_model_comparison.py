"""
Development-period comparison of the paper model set (configs/env_main.json).

Seed-dependent models (XGBoost, LSTM) are averaged over seeds; deterministic ones are taken as is.
Reports: (1) skill vs seasonal naive, (2) within-family input contrasts (RE vs E, RE vs R; SARIMAX vs
SARIMA), (3) nonlinear test models vs linear baselines on the same inputs. All comparisons are paired
on identical target months with a Diebold-Mariano test; seed agreement is checked for XGBoost/LSTM.

Usage: python scripts/env_model_comparison.py <project_root> [config_name ...]   (default: env_main_dev)
With several config names, a model's rows from a later name replace that model's rows from earlier names
(e.g. `env_main_dev env_main_tuned_dev` keeps LR/XGBoost from env_main and tuned SARIMA(X)/LSTM).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
names = sys.argv[2:] or ["env_main_dev"]
sys.path.insert(0, str(root))
from src.env_forecast.metrics import diebold_mariano  # noqa: E402

p = pd.DataFrame()
for name in names:
    frames = []
    for folder in (root / "results" / "env").glob(f"{name}_*"):
        config = json.loads((folder / "config.json").read_text())
        if config["name"] == name:
            frames.append(pd.read_csv(folder / "predictions.csv").assign(seed=config["seed"], source=name))
    if not frames:
        sys.exit(f"no runs named {name}")
    new = pd.concat(frames, ignore_index=True)
    p = pd.concat([p[~p["model"].isin(new["model"].unique())] if len(p) else p, new], ignore_index=True)
print("source per model:", p.groupby("model")["source"].first().to_dict())
SEEDED = {"xgboost", "lstm"}
p = p[p["model"].isin(SEEDED) | (p["seed"] == p["seed"].min())]
print("seeds per model:", p.groupby("model")["seed"].unique().apply(sorted).to_dict())

KEY = ["county", "horizon", "target_month"]
ens = p.groupby(["model", "feature_set", *KEY]).agg(pred=("predicted_rate", "mean"), actual=("actual_rate", "first")).reset_index()
ens = ens.dropna(subset=["pred", "actual"])


def series(model, fs, frame=ens):
    return frame[(frame.model == model) & (frame.feature_set == fs)].set_index(KEY)


def compare(a, b, label):
    """Pair (model, feature_set) a against b per county x horizon on shared targets."""
    joined = series(*a)[["pred", "actual"]].join(series(*b)[["pred"]], rsuffix="_b", how="inner").reset_index()
    rows = []
    for (county, h), g in joined.groupby(["county", "horizon"]):
        g = g.sort_values("target_month")
        la, lb = (g.pred - g.actual) ** 2, (g.pred_b - g.actual) ** 2
        _, pval = diebold_mariano(la.to_numpy(), lb.to_numpy(), int(h))
        change = np.sqrt(la.mean()) / np.sqrt(lb.mean()) - 1
        agree = True
        if a[0] in SEEDED or b[0] in SEEDED:
            signs = []
            for seed in sorted(p.seed.unique()):
                ps = p[p.seed == seed]
                sa = ps[(ps.model == a[0]) & (ps.feature_set == a[1]) & (ps.county == county) & (ps.horizon == h)]
                sb = p[(p.model == b[0]) & (p.feature_set == b[1]) & (p.county == county) & (p.horizon == h)]
                sb = sb[sb.seed == seed] if b[0] in SEEDED else sb
                if len(sa) == 0 or len(sb) == 0:
                    continue
                m = sa.set_index("target_month")[["predicted_rate", "actual_rate"]].join(sb.set_index("target_month")[["predicted_rate"]], rsuffix="_b").dropna()
                ra = np.sqrt(((m.predicted_rate - m.actual_rate) ** 2).mean())
                rb = np.sqrt(((m.predicted_rate_b - m.actual_rate) ** 2).mean())
                signs.append(np.sign(ra / rb - 1))
            agree = bool(signs) and all(s == np.sign(change) for s in signs)
        rows.append({"comparison": label, "county": county, "horizon": h, "n": len(g), "rmse_a": np.sqrt(la.mean()),
                     "rmse_b": np.sqrt(lb.mean()), "rel_change": change, "dm_p": pval, "seeds_agree": agree})
    return pd.DataFrame(rows)


def summarize(table):
    return table.groupby(["comparison", "horizon"]).apply(lambda g: pd.Series({
        "a_better": int((g.rel_change < 0).sum()),
        "a_better_all_seeds": int(((g.rel_change < 0) & g.seeds_agree).sum()),
        "sig_a_better": int(((g.rel_change < 0) & (g.dm_p < 0.05)).sum()),
        "sig_a_worse": int(((g.rel_change > 0) & (g.dm_p < 0.05)).sum()),
        "median_change": g.rel_change.median(),
    }), include_groups=False)


out = root / "results" / "env" / ("main_comparison" if names == ["env_main_dev"] else "main_comparison_" + "+".join(names))
out.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 220)

# (1) skill vs seasonal naive
ref = series("seasonal_naive", "none")[["pred"]].rename(columns={"pred": "ref"})
skill_rows = []
for (model, fs), g in ens.groupby(["model", "feature_set"]):
    j = g.set_index(KEY).join(ref, how="inner").reset_index()
    for (county, h), c in j.groupby(["county", "horizon"]):
        rm, rr = np.sqrt(((c.pred - c.actual) ** 2).mean()), np.sqrt(((c.ref - c.actual) ** 2).mean())
        skill_rows.append({"model": model, "feature_set": fs, "county": county, "horizon": h, "rmse": rm, "skill": 1 - rm / rr})
skill = pd.DataFrame(skill_rows)
skill.to_csv(out / "skill_vs_seasonal_naive.csv", index=False)
print("\n(1) Median skill vs seasonal naive across counties [counties with positive skill]")
med = skill.pivot_table(index=["model", "feature_set"], columns="horizon", values="skill", aggfunc="median")
pos = skill.assign(p=skill.skill > 0).pivot_table(index=["model", "feature_set"], columns="horizon", values="p", aggfunc="sum")
print((med.round(3).astype(str) + " [" + pos.astype(int).astype(str) + "]").to_string())

# (2) within-family input contrasts and (3) nonlinear vs linear baselines
contrasts = [(("lr", "RE"), ("lr", "E"), "LR: RE vs E"), (("xgboost", "RE"), ("xgboost", "E"), "XGB: RE vs E"),
             (("lstm", "RE"), ("lstm", "E"), "LSTM: RE vs E"), (("lr", "RE"), ("lr", "R"), "LR: RE vs R"),
             (("xgboost", "RE"), ("xgboost", "R"), "XGB: RE vs R"), (("lstm", "RE"), ("lstm", "R"), "LSTM: RE vs R"),
             (("sarimax", "RE"), ("sarima", "R"), "SARIMAX vs SARIMA")]
versus = [((m, fs), ("lr", fs), f"{m.upper()}-{fs} vs LR-{fs}") for m in ("xgboost", "lstm") for fs in ("R", "E", "RE")]
versus += [((m, "RE"), ("sarimax", "RE"), f"{m.upper()}-RE vs SARIMAX") for m in ("xgboost", "lstm")]
versus += [((m, "RE"), ("lr", "R"), f"{m.upper()}-RE vs AR (LR-R)") for m in ("xgboost", "lstm")]
for title, specs, file in (("(2) Within-family input contrasts", contrasts, "input_contrasts"),
                           ("(3) Nonlinear test models vs linear baselines, same inputs", versus, "model_contrasts")):
    table = pd.concat([compare(a, b, label) for a, b, label in specs], ignore_index=True)
    table.to_csv(out / f"{file}.csv", index=False)
    print(f"\n{title} (negative change = first model/input set better; counts out of 8 counties)")
    print(summarize(table).round(3).unstack("horizon").to_string())
