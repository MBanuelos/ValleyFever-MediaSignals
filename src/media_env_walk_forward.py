"""
Media + environmental walk-forward forecasting for Valley Fever (county level).

Design (follows the conventions in ValleyFever-MediaSignals README / notebooks)
-------------------------------------------------------------------------------
* Target            : log-difference of log1p(rate_per_100k)  ("logdiff_rate").
* Validation        : walk-forward, ROLLING 120-month (10-year) training window,
                      step = 1 month, forecast 1, 3 and 6 months ahead.
* Forecast strategy : direct. For every forecast origin t and step s = 1..6 one
                      model is trained to predict month t+s from information
                      available at t (all features lagged by s months).
                      The horizon-h result averages errors over months t+1..t+h,
                      so the 1/3/6-month results share the same fitted models.
* No leakage        : features for target month i are taken from month i-s; the
                      training window only contains targets <= t; hyper-
                      parameters are tuned once, on the FIRST window only,
                      with TimeSeriesSplit; imputers/scalers are fit per window.
* Models            : seasonal-naive baseline, Ridge, Random Forest, XGBoost.
* Metrics           : RMSE and MAE (log-diff scale, the repo convention) and on
                      the rate_per_100k scale; % improvement vs seasonal-naive.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

SEED = 5
TRAIN_MONTHS = 120            # 10-year rolling window
HORIZONS = (1, 3, 6)
MAX_STEP = max(HORIZONS)
MEDIA_START = pd.Period("2011-01", "M")       # first month with media data
FIRST_TARGET = pd.Period("2011-02", "M")      # first month with a log-difference
MODELS = ("seasonal_naive", "ridge", "random_forest", "xgboost")

ENV_COLS = [
    "Precipitation", "Avg Rel Hum (%)", "Avg Wind Speed (mph)", "Wind Run (miles)",
    "windeventcount", "Avg Soil Temp (F)", "AQI_PM25", "AQI_PM10",
    "FIRE_Acres_Burned", "fungicide_lbs_prd_used", "rodent_lbs_prd_used",
]
# Same compact media set the repo's California-level notebooks use
ARTICLE_COLS = ["article_count", "mean_title_sentiment", "mean_title_risk", "mean_body_risk"]
TWITTER_COLS = ["avg_sentiment", "avg_risk", "mean_body_sentiment", "tweet_count", "avg_likes"]
MEDIA_COLS = ARTICLE_COLS + TWITTER_COLS
MEDIA_COUNT_COLS = ["article_count", "tweet_count", "avg_likes"]   # 0 when no coverage
LOG_COLS = MEDIA_COUNT_COLS + ["FIRE_Acres_Burned", "fungicide_lbs_prd_used", "rodent_lbs_prd_used"]

AR_GROUP = "AR: recent log-change"
SEASONAL_GROUP = "AR: same month last year"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def top_counties(data_dir, n=10):
    """Rank counties by total Valley Fever cases (2011-2024) and split top-n / remaining."""
    media = pd.read_csv(Path(data_dir) / "interim" / "Media_per_county.csv")
    totals = media.groupby("County")["num_cases"].sum().sort_values(ascending=False)
    return list(totals.index[:n]), list(totals.index[n:])


def load_county_panel(county, data_dir):
    """Monthly panel (2001-2024) for one county: environment + CA-wide media + target."""
    data_dir = Path(data_dir)
    env = pd.read_csv(data_dir / "processed" / "env_vf_aggregate" / "aggregate_all_counties_2001_2024.csv")
    env = env[env["County"] == county].copy()
    if env.empty:
        raise ValueError(f"County '{county}' not found in the aggregate environmental file.")
    env["date"] = pd.PeriodIndex(env["Year-Month"], freq="M")
    env = env.set_index("date").sort_index()

    media = pd.read_csv(data_dir / "interim" / "Media_per_county.csv")
    media = media[media["County"] == county].copy()
    media["date"] = pd.PeriodIndex(media["year_month"], freq="M")
    media = media.set_index("date")[MEDIA_COLS]

    panel = env[ENV_COLS + ["rate_per_100k"]].join(media)

    # 12 months in 2011-2017 have no media records: no coverage -> counts are 0,
    # sentiment/risk carry the last observed value forward (causal, no look-ahead).
    in_media_era = panel.index >= MEDIA_START
    for c in MEDIA_COUNT_COLS:
        panel.loc[in_media_era, c] = panel.loc[in_media_era, c].fillna(0.0)
    other = [c for c in MEDIA_COLS if c not in MEDIA_COUNT_COLS]
    panel.loc[in_media_era, other] = panel.loc[in_media_era, other].ffill()
    panel[ENV_COLS] = panel[ENV_COLS].ffill()          # e.g. 2024 pesticide gaps

    for c in LOG_COLS:
        panel[c] = np.log1p(panel[c].clip(lower=0))
    panel["log_rate"] = np.log1p(panel["rate_per_100k"])
    panel["logdiff_rate"] = panel["log_rate"].diff()
    return panel


def _step_features(panel, step, feature_set):
    cols = ENV_COLS + (MEDIA_COLS if feature_set == "media_env" else [])
    X = panel[cols].shift(step)
    y = panel["logdiff_rate"]
    X["ar_lag0"] = y.shift(step)
    X["ar_lag1"] = y.shift(step + 1)
    X["ar_lag2"] = y.shift(step + 2)
    X["seasonal_lag12"] = y.shift(12)
    return X


def _group_of(col):
    if col.startswith("ar_lag"):
        return AR_GROUP
    if col == "seasonal_lag12":
        return SEASONAL_GROUP
    return col


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def _make_model(name):
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    if name == "ridge":
        return make_pipeline(imp, StandardScaler(), Ridge())
    if name == "random_forest":
        return make_pipeline(imp, RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=1))
    if name == "xgboost":
        return make_pipeline(imp, XGBRegressor(
            n_estimators=100, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=1, verbosity=0))
    raise ValueError(name)


_GRIDS = {
    "ridge": {"ridge__alpha": [1.0, 10.0, 100.0, 1000.0]},
    "random_forest": {
        "randomforestregressor__max_depth": [3, 6, None],
        "randomforestregressor__min_samples_leaf": [2, 5],
        "randomforestregressor__max_features": [0.33, 1.0],
    },
    "xgboost": {
        "xgbregressor__max_depth": [2, 3],
        "xgbregressor__learning_rate": [0.05, 0.1],
    },
}


def _importance(name, fitted, columns):
    est = fitted[-1]
    if name == "ridge":
        raw = np.abs(est.coef_)
    else:
        raw = np.asarray(est.feature_importances_, dtype=float)
    total = raw.sum()
    raw = raw / total if total > 0 else raw
    out = {}
    for col, v in zip(columns, raw):
        g = _group_of(col)
        out[g] = out.get(g, 0.0) + float(v)
    return out


# --------------------------------------------------------------------------- #
# Walk-forward engine
# --------------------------------------------------------------------------- #
def run_county(county, data_dir, feature_set="media_env", models=MODELS, verbose=True):
    """
    Run the rolling-window walk-forward evaluation for one county.

    Returns dict with: metrics (DataFrame), forecasts (DataFrame, rate scale),
    importances (DataFrame), best_params, meta.
    """
    t_start = time.time()
    panel = load_county_panel(county, data_dir)
    n = len(panel)
    pos = {p: k for k, p in enumerate(panel.index)}
    y = panel["logdiff_rate"]
    first_origin = pos[FIRST_TARGET] + TRAIN_MONTHS - 1      # end of first 10-year window
    last = n - 1
    origins = list(range(first_origin, last))                # origin t needs t+1 <= last

    feats, valid = {}, {}
    for s in range(1, MAX_STEP + 1):
        F = _step_features(panel, s, feature_set)
        feats[s] = F
        # a training target i needs: it is a real log-diff, and its lag-s features
        # fall inside the media era so every row is built from real media data.
        ok = (np.arange(n) >= pos[MEDIA_START] + s) & (np.arange(n) >= pos[FIRST_TARGET]) & y.notna().to_numpy()
        valid[s] = ok

    # ---- tune once, on the first window only (step 1) -------------------------
    fitted_models = [m for m in models if m != "seasonal_naive"]
    best_params = {}
    w0 = np.arange(first_origin - TRAIN_MONTHS + 1, first_origin + 1)
    w0 = w0[valid[1][w0]]
    for m in fitted_models:
        gs = GridSearchCV(_make_model(m), _GRIDS[m], cv=TimeSeriesSplit(n_splits=5),
                          scoring="neg_mean_squared_error", n_jobs=1)
        gs.fit(feats[1].iloc[w0], y.iloc[w0])
        best_params[m] = gs.best_params_

    # ---- rolling-window loop ---------------------------------------------------
    records = []
    imp_sum = {m: {} for m in fitted_models}
    imp_cnt = {m: 0 for m in fitted_models}
    for k, t in enumerate(origins):
        max_s = min(MAX_STEP, last - t)
        for s in range(1, max_s + 1):
            F = feats[s]
            target_pos = t + s
            actual = float(y.iloc[target_pos])
            rows = np.arange(t - TRAIN_MONTHS + 1, t + 1)
            rows = rows[valid[s][rows]]
            Xtr, ytr = F.iloc[rows], y.iloc[rows]
            Xte = F.iloc[[target_pos]]
            base = dict(origin=panel.index[t], step=s, target=panel.index[target_pos], actual=actual)

            if "seasonal_naive" in models:
                records.append({**base, "model": "seasonal_naive", "pred": float(Xte["seasonal_lag12"].iloc[0])})
            for m in fitted_models:
                mod = _make_model(m).set_params(**best_params[m])
                mod.fit(Xtr, ytr)
                records.append({**base, "model": m, "pred": float(mod.predict(Xte)[0])})
                for g, v in _importance(m, mod, list(F.columns)).items():
                    imp_sum[m][g] = imp_sum[m].get(g, 0.0) + v
                imp_cnt[m] += 1
        if verbose and (k + 1) % 10 == 0:
            print(f"  {county} [{feature_set}] origin {k + 1}/{len(origins)}", flush=True)

    preds = pd.DataFrame(records)
    metrics, forecasts = _evaluate(preds, panel, last)
    metrics.insert(0, "county", county)
    metrics.insert(1, "feature_set", feature_set)
    forecasts.insert(0, "county", county)
    forecasts.insert(1, "feature_set", feature_set)

    imp_rows = []
    for m in fitted_models:
        for g, v in imp_sum[m].items():
            imp_rows.append(dict(county=county, feature_set=feature_set, model=m, feature=g,
                                 importance=v / max(imp_cnt[m], 1)))
    importances = pd.DataFrame(imp_rows)

    meta = dict(county=county, feature_set=feature_set, n_months=n,
                first_forecast_month=str(panel.index[first_origin + 1]),
                last_forecast_month=str(panel.index[last]),
                n_origins=len(origins), train_months=TRAIN_MONTHS,
                runtime_sec=round(time.time() - t_start, 1))
    return dict(metrics=metrics, forecasts=forecasts, importances=importances,
                best_params={m: {k: (None if v is None else v) for k, v in p.items()}
                             for m, p in best_params.items()},
                meta=meta)


def _rmse(e):
    return float(np.sqrt(np.mean(np.square(e)))) if len(e) else np.nan


def _evaluate(preds, panel, last):
    """Per-model, per-horizon metrics + rate-scale forecast series."""
    preds = preds.copy()
    preds["err"] = preds["pred"] - preds["actual"]
    origin_pos = preds["origin"].map({p: k for k, p in enumerate(panel.index)})
    preds["origin_pos"] = origin_pos

    # cumulative predicted log-change from the origin -> predicted rate per month
    preds = preds.sort_values(["model", "origin_pos", "step"])
    preds["cum_pred"] = preds.groupby(["model", "origin_pos"])["pred"].cumsum()
    lr = panel["log_rate"].to_numpy()
    rate = panel["rate_per_100k"].to_numpy()
    tgt_pos = preds["origin_pos"] + preds["step"]
    preds["pred_rate"] = np.expm1(lr[preds["origin_pos"].to_numpy()] + preds["cum_pred"].to_numpy())
    preds["actual_rate"] = rate[tgt_pos.to_numpy()]
    preds["rate_err"] = preds["pred_rate"] - preds["actual_rate"]

    rows = []
    for h in HORIZONS:
        sub = preds[(preds["step"] <= h) & (preds["origin_pos"] <= last - h)]
        for m, g in sub.groupby("model"):
            win = g.groupby("origin_pos")
            rows.append(dict(
                model=m, horizon=h,
                n_windows=int(win.ngroups), n_forecasts=int(len(g)),
                rmse=_rmse(g["err"]), mae=float(g["err"].abs().mean()),
                avg_window_rmse=float(win["err"].apply(_rmse).mean()),
                avg_window_mae=float(win["err"].apply(lambda e: e.abs().mean()).mean()),
                rate_rmse=_rmse(g["rate_err"]), rate_mae=float(g["rate_err"].abs().mean()),
            ))
    metrics = pd.DataFrame(rows)
    base = metrics[metrics["model"] == "seasonal_naive"].set_index("horizon")
    if not base.empty:
        for col in ("rmse", "mae"):
            b = metrics["horizon"].map(base[col])
            metrics[f"pct_improve_{col}"] = (b - metrics[col]) / b * 100.0
    metrics = metrics.sort_values(["horizon", "rmse"]).reset_index(drop=True)

    # rate-scale series for plotting: month t+h forecast issued h months earlier
    fc = preds.copy()
    fc["horizon"] = fc["step"]
    fc = fc[fc["horizon"].isin(HORIZONS) & (fc["origin_pos"] <= last - fc["horizon"])]
    forecasts = fc[["model", "horizon", "target", "actual_rate", "pred_rate"]].copy()
    forecasts["target"] = forecasts["target"].astype(str)
    return metrics, forecasts.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Best model + top features
# --------------------------------------------------------------------------- #
def summarize_best(result, n_features=3, selection="rmse"):
    """
    Pick the best model (lowest mean `selection` metric across horizons 1/3/6,
    log-diff scale) and its top features.

    Top features come from the best *fitted* model (importance averaged over every
    walk-forward fit); if seasonal-naive wins, the best fitted model is used.
    """
    metrics, imp = result["metrics"], result["importances"]
    score = metrics.groupby("model")[selection].mean().sort_values()
    best = score.index[0]
    fitted_score = score[[m for m in score.index if m != "seasonal_naive"]]
    feat_model = best if best != "seasonal_naive" else fitted_score.index[0]

    fi = (imp[imp["model"] == feat_model].sort_values("importance", ascending=False)
          .reset_index(drop=True))
    top = fi.head(n_features)
    media_names = set(MEDIA_COLS)
    top_media = fi[fi["feature"].isin(media_names)].head(1)

    per_h = metrics[metrics["model"] == best].set_index("horizon")
    base = metrics[metrics["model"] == "seasonal_naive"].set_index("horizon")
    out = dict(
        county=result["meta"]["county"],
        best_model=best,
        features_from_model=feat_model,
        top_features=list(top["feature"]),
        top_feature_importance=[round(float(v), 4) for v in top["importance"]],
        top_media_feature=(top_media["feature"].iloc[0] if len(top_media) else None),
        mean_rmse=float(score.iloc[0]),
        mean_rmse_seasonal_naive=float(base["rmse"].mean()) if not base.empty else np.nan,
    )
    for h in HORIZONS:
        out[f"rmse_{h}m"] = float(per_h.loc[h, "rmse"])
        out[f"mae_{h}m"] = float(per_h.loc[h, "mae"])
    return out


def best_model_and_top_features(county, data_dir, feature_set="media_env", n_features=3, verbose=False):
    """Run the walk-forward evaluation for ONE county and return its best model + top features."""
    res = run_county(county, data_dir, feature_set=feature_set, verbose=verbose)
    return summarize_best(res, n_features=n_features), res


def best_model_for_remaining_counties(data_dir, n_top=10, counties=None, feature_set="media_env",
                                      n_features=3, verbose=True):
    """
    For every county outside the top `n_top` (or an explicit list), run the walk-forward
    evaluation and return a table with the best model and top three features.
    """
    if counties is None:
        _, counties = top_counties(data_dir, n_top)
    rows, results = [], {}
    for c in counties:
        if verbose:
            print(f"[remaining] {c}", flush=True)
        summary, res = best_model_and_top_features(c, data_dir, feature_set, n_features, verbose=False)
        results[c] = res
        rows.append(summary)
    table = pd.DataFrame(rows)
    for i in range(n_features):
        table[f"feature_{i + 1}"] = table["top_features"].map(lambda f, i=i: f[i] if len(f) > i else None)
    return table, results


# --------------------------------------------------------------------------- #
# Batch run
# --------------------------------------------------------------------------- #
def main(data_dir, out_dir, n_top=10, only=None):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    top, rest = top_counties(data_dir, n_top)
    if only:
        top = [c for c in top if c in only]
        rest = [c for c in rest if c in only]

    def collect(counties, feature_set):
        ms, fs, imps, metas, params = [], [], [], [], {}
        for c in counties:
            print(f"== {c} ({feature_set})", flush=True)
            r = run_county(c, data_dir, feature_set=feature_set)
            ms.append(r["metrics"]); fs.append(r["forecasts"]); imps.append(r["importances"])
            metas.append(r["meta"]); params[c] = r["best_params"]
            print(f"   done in {r['meta']['runtime_sec']}s", flush=True)
        return (pd.concat(ms, ignore_index=True), pd.concat(fs, ignore_index=True),
                pd.concat(imps, ignore_index=True), metas, params)

    # 1) media + environment, top counties
    m, f, i, meta, params = collect(top, "media_env")
    m.to_csv(out_dir / "top10_metrics_media_env.csv", index=False)
    f.to_csv(out_dir / "top10_forecasts_media_env.csv", index=False)
    i.to_csv(out_dir / "top10_feature_importance_media_env.csv", index=False)
    json.dump(dict(meta=meta, best_params=params), open(out_dir / "top10_run_meta.json", "w"), indent=2, default=str)

    # 2) remaining counties -> best model + top 3 features
    rows = []
    for c in rest:
        print(f"== {c} (remaining)", flush=True)
        r = run_county(c, data_dir, feature_set="media_env")
        s = summarize_best(r)
        rows.append(s)
        r["metrics"].to_csv(out_dir / f"remaining_metrics_{c.replace(' ', '')}.csv", index=False)
        r["importances"].to_csv(out_dir / f"remaining_importance_{c.replace(' ', '')}.csv", index=False)
    table = pd.DataFrame(rows)
    for k in range(3):
        table[f"feature_{k + 1}"] = table["top_features"].map(lambda ff, k=k: ff[k] if len(ff) > k else None)
    table.to_csv(out_dir / "remaining_counties_best_model_top3.csv", index=False)

    # 3) ablation: environment only, top counties (does media add anything?)
    m2, f2, i2, _, _ = collect(top, "env_only")
    m2.to_csv(out_dir / "top10_metrics_env_only.csv", index=False)
    f2.to_csv(out_dir / "top10_forecasts_env_only.csv", index=False)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results/media_env_walk_forward")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these counties (testing)")
    a = ap.parse_args()
    main(a.data_dir, a.out_dir, only=a.only)
