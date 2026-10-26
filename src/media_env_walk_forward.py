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
* Feature sets      : env_only, media_only, media_env. Every set also contains the
                      county's own past case changes (autoregressive terms).
* Models            : seasonal-naive baseline, Ridge, Random Forest, XGBoost,
                      SARIMAX (exog lagged 6 months, top-5 exog chosen on the first
                      window) and a NumPy LSTM (12-feature-month look-back, 6-step output).
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
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from numpy_lstm import NumpyLSTM
except ImportError:  # imported as a package
    from .numpy_lstm import NumpyLSTM

warnings.filterwarnings("ignore")

SEED = 5
TRAIN_MONTHS = 120            # 10-year rolling window
HORIZONS = (1, 3, 6)
MAX_STEP = max(HORIZONS)
MEDIA_START = pd.Period("2011-01", "M")       # first month with media data
FIRST_TARGET = pd.Period("2011-02", "M")      # first month with a log-difference
MODELS = ("seasonal_naive", "ridge", "random_forest", "xgboost", "sarimax", "lstm")
FEATURE_SETS = ("env_only", "media_only", "media_env")
LSTM_LOOKBACK = 6
SARIMAX_K = 5

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


def _set_cols(feature_set):
    if feature_set == "env_only":
        return list(ENV_COLS)
    if feature_set == "media_only":
        return list(MEDIA_COLS)
    if feature_set == "media_env":
        return ENV_COLS + MEDIA_COLS
    raise ValueError(f"unknown feature_set {feature_set}")


def _step_features(panel, step, feature_set):
    cols = _set_cols(feature_set)
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
# SARIMAX and LSTM (multi-step models: one fit per forecast origin)
# --------------------------------------------------------------------------- #
_EXT_EXCLUDE = ("ar_lag", "seasonal_lag12")


def _external_cols(F):
    return [c for c in F.columns if not (c.startswith(_EXT_EXCLUDE[0]) or c == _EXT_EXCLUDE[1])]


def _sarimax_prepare(F6, y, rows0, k=SARIMAX_K):
    """First window only: pick the k external features most correlated with y, then a small AIC grid for the order."""
    ext = _external_cols(F6)
    X0 = F6.iloc[rows0][ext]
    y0 = y.iloc[rows0].to_numpy()
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    Xi = pd.DataFrame(imp.fit_transform(X0), columns=ext)
    corr = Xi.apply(lambda c: abs(np.corrcoef(c, y0)[0, 1]) if c.std() > 0 else 0.0)
    cols = list(corr.sort_values(ascending=False).head(k).index)
    Z = StandardScaler().fit_transform(Xi[cols])
    best, best_aic = (1, 0, 0, 0), np.inf
    for (p, q, P, Q) in [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (1, 1, 0, 0), (2, 0, 0, 0),
                         (0, 0, 1, 0), (1, 0, 1, 0), (0, 1, 1, 0), (1, 1, 1, 0), (1, 0, 0, 1)]:
        try:
            res = SARIMAX(y0, exog=Z, order=(p, 0, q), seasonal_order=(P, 0, Q, 12), trend="c",
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=60)
            if np.isfinite(res.aic) and res.aic < best_aic:
                best, best_aic = (p, q, P, Q), res.aic
        except Exception:
            continue
    p, q, P, Q = best
    return dict(cols=cols, order=(p, 0, q), seasonal=(P, 0, Q, 12))


def _sarimax_forecast(F6, y, rows, t, max_s, spec):
    """Fit on the current window (exog lagged 6 months, so all future exog is known at t) and forecast max_s steps."""
    cols = spec["cols"]
    Xtr = F6.iloc[rows][cols]
    ytr = y.iloc[rows].to_numpy()
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    Ztr = sc.transform(imp.transform(Xtr))
    Zf = sc.transform(imp.transform(F6.iloc[t + 1:t + 1 + max_s][cols]))
    try:
        res = SARIMAX(ytr, exog=Ztr, order=spec["order"], seasonal_order=spec["seasonal"], trend="c",
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=60)
        pred = np.asarray(res.forecast(steps=max_s, exog=Zf), dtype=float)
        names = list(res.param_names)
        coefs = {c: abs(float(res.params[names.index(f"x{i + 1}")])) for i, c in enumerate(cols)
                 if f"x{i + 1}" in names}
        ok = bool(np.all(np.isfinite(pred)))
    except Exception:
        ok, coefs = False, {}
    if not ok:
        return np.full(max_s, float(ytr.mean())), {}, False
    tot = sum(coefs.values())
    imp_out = {c: v / tot for c, v in coefs.items()} if tot > 0 else {}
    return pred, imp_out, True


def _lstm_forecast(A, y_arr, pos_media, t, max_s, window_start, seed):
    """
    One LSTM per origin. Inputs: previous LSTM_LOOKBACK months of features (+ the case log-change),
    outputs: the next MAX_STEP monthly log-changes. All scaling is fit on data up to t.
    """
    L = LSTM_LOOKBACK
    lo = max(window_start, pos_media + L - 1)
    idx = np.arange(lo, t - MAX_STEP + 1)                 # last month of each training sequence
    if len(idx) < 20:
        return np.full(max_s, np.nan)
    hist = A[max(pos_media, 0):t + 1]
    med = np.nanmedian(np.where(np.isfinite(hist), hist, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    B = np.where(np.isfinite(A), A, med)
    mu, sd = B[pos_media:t + 1].mean(0), B[pos_media:t + 1].std(0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    Z = (B - mu) / sd
    X = np.stack([Z[i - L + 1:i + 1] for i in idx])
    Y = np.stack([y_arr[i + 1:i + 1 + MAX_STEP] for i in idx])
    ym, ys = Y.mean(), Y.std() if Y.std() > 1e-9 else 1.0
    net = NumpyLSTM(Z.shape[1], MAX_STEP, hidden=24, seed=seed)
    net.fit(X, (Y - ym) / ys, epochs=60, lr=0.01, batch=16)
    out = net.predict(Z[t - L + 1:t + 1][None])[0] * ys + ym
    return out[:max_s]

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
    fitted_models = [m for m in models if m in _GRIDS]          # tuned, one fit per step
    best_params = {}
    w0 = np.arange(first_origin - TRAIN_MONTHS + 1, first_origin + 1)
    w0 = w0[valid[1][w0]]
    for m in fitted_models:
        gs = GridSearchCV(_make_model(m), _GRIDS[m], cv=TimeSeriesSplit(n_splits=5),
                          scoring="neg_mean_squared_error", n_jobs=1)
        gs.fit(feats[1].iloc[w0], y.iloc[w0])
        best_params[m] = gs.best_params_

    # ---- SARIMAX / LSTM set-up (first window only) ------------------------------
    sar_spec, sar_fail = None, 0
    if "sarimax" in models:
        rows0 = np.arange(first_origin - TRAIN_MONTHS + 1, first_origin + 1)
        rows0 = rows0[valid[6][rows0]]
        sar_spec = _sarimax_prepare(feats[6], y, rows0)
        best_params["sarimax"] = dict(order=sar_spec["order"], seasonal_order=sar_spec["seasonal"],
                                      exog=sar_spec["cols"])
    A = np.column_stack([panel[_set_cols(feature_set)].to_numpy(float), y.to_numpy(float)])
    y_arr = y.to_numpy(float)

    # ---- rolling-window loop ---------------------------------------------------
    records = []
    imp_models = fitted_models + (["sarimax"] if "sarimax" in models else [])
    imp_sum = {m: {} for m in imp_models}
    imp_cnt = {m: 0 for m in imp_models}
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
        # multi-step models: one fit per origin
        def _add(model, pr):
            for s_i in range(1, max_s + 1):
                records.append(dict(origin=panel.index[t], step=s_i, target=panel.index[t + s_i],
                                    actual=float(y.iloc[t + s_i]), model=model, pred=float(pr[s_i - 1])))
        if "sarimax" in models:
            rows6 = np.arange(t - TRAIN_MONTHS + 1, t + 1)
            rows6 = rows6[valid[6][rows6]]
            pr, imp_o, ok = _sarimax_forecast(feats[6], y, rows6, t, max_s, sar_spec)
            sar_fail += int(not ok)
            _add("sarimax", pr)
            if ok:
                for g, v in imp_o.items():
                    imp_sum["sarimax"][g] = imp_sum["sarimax"].get(g, 0.0) + v
                imp_cnt["sarimax"] += 1
        if "lstm" in models:
            pr = _lstm_forecast(A, y_arr, pos[MEDIA_START], t, max_s, t - TRAIN_MONTHS + 1, SEED + t)
            if np.all(np.isfinite(pr)):
                _add("lstm", pr)
        if verbose and (k + 1) % 10 == 0:
            print(f"  {county} [{feature_set}] origin {k + 1}/{len(origins)}", flush=True)

    preds = pd.DataFrame(records)
    metrics, forecasts = _evaluate(preds, panel, last)
    metrics.insert(0, "county", county)
    metrics.insert(1, "feature_set", feature_set)
    forecasts.insert(0, "county", county)
    forecasts.insert(1, "feature_set", feature_set)

    imp_rows = []
    for m in imp_models:
        for g, v in imp_sum[m].items():
            imp_rows.append(dict(county=county, feature_set=feature_set, model=m, feature=g,
                                 importance=v / max(imp_cnt[m], 1)))
    importances = pd.DataFrame(imp_rows)

    meta = dict(county=county, feature_set=feature_set, n_months=n,
                first_forecast_month=str(panel.index[first_origin + 1]),
                last_forecast_month=str(panel.index[last]),
                n_origins=len(origins), train_months=TRAIN_MONTHS,
                sarimax_fit_failures=sar_fail,
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

    Top features come from the best model that reports feature importance (Ridge, Random Forest,
    XGBoost, SARIMAX); if seasonal-naive or the LSTM wins, the best of those is used.
    """
    metrics, imp = result["metrics"], result["importances"]
    score = metrics.groupby("model")[selection].mean().sort_values()
    best = score.index[0]
    has_imp = set(imp["model"]) if len(imp) else set()
    fitted_score = score[[m for m in score.index if m in has_imp]]
    feat_model = best if best in has_imp else fitted_score.index[0]

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
# Batch run (results are saved per county / feature set so runs can be resumed and combined)
# --------------------------------------------------------------------------- #
def _stem(tag, feature_set, county):
    return f"{tag}__{feature_set}__{county.replace(' ', '')}"


def run_parts(data_dir, out_dir, scope="top", feature_sets=FEATURE_SETS, models=MODELS, tag="run", counties=None):
    """Run the walk-forward for a scope ('top', 'remaining', 'all') or an explicit county list; save one set of files per county."""
    top, rest = top_counties(data_dir)
    counties = list(counties) if counties else {"top": top, "remaining": rest, "all": top + rest}[scope]
    parts = Path(out_dir) / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    for fs in feature_sets:
        for c in counties:
            stem = parts / _stem(tag, fs, c)
            if Path(f"{stem}_metrics.csv").exists():
                continue                                     # resume
            print(f"== {c} ({fs}) models={list(models)}", flush=True)
            r = run_county(c, data_dir, feature_set=fs, models=models)
            r["metrics"].to_csv(f"{stem}_metrics.csv", index=False)
            r["forecasts"].to_csv(f"{stem}_forecasts.csv", index=False)
            r["importances"].to_csv(f"{stem}_importance.csv", index=False)
            json.dump(dict(meta=r["meta"], best_params=r["best_params"]), open(f"{stem}_meta.json", "w"),
                      indent=2, default=str)
            print(f"   done in {r['meta']['runtime_sec']}s", flush=True)


def _newest_first(paths):
    return sorted(paths, key=lambda p: Path(p).stat().st_mtime, reverse=True)


def _read_if(path):
    return pd.read_csv(path) if Path(path).exists() else None


def _merge(frames, key):
    """Concatenate (newest first) and keep the first row per key, so newer results override older ones."""
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(key, keep="first")


def combine(data_dir, out_dir):
    """
    Merge new runs (out_dir/parts) INTO the saved results in out_dir.

    Nothing already saved is dropped: rows are matched on (county, feature set, model, horizon, ...)
    and a newer run replaces an older row with the same key. Running `combine` with no new parts
    simply rewrites the same tables. To remove a model's results, delete its rows from the CSVs.
    """
    out_dir = Path(out_dir)
    parts = out_dir / "parts"
    top, rest = top_counties(data_dir)
    key_m = ["county", "feature_set", "model", "horizon"]
    key_f = key_m + ["target"]
    key_i = ["county", "feature_set", "model", "feature"]

    def new(suffix):
        return [pd.read_csv(p) for p in _newest_first(parts.glob(f"*_{suffix}.csv"))]

    M = _merge(new("metrics")
               + [_read_if(out_dir / f"top10_metrics_{fs}.csv") for fs in FEATURE_SETS]
               + [_read_if(out_dir / "remaining_metrics_media_env_all_models.csv")], key_m)
    F = _merge(new("forecasts") + [_read_if(out_dir / f"top10_forecasts_{fs}.csv") for fs in FEATURE_SETS], key_f)
    I = _merge(new("importance")
               + [_read_if(out_dir / f"top10_feature_importance_{fs}.csv") for fs in FEATURE_SETS]
               + [_read_if(out_dir / "remaining_feature_importance_media_env.csv")], key_i)

    # stable ordering keeps rebuilds identical and git diffs small
    M = M.sort_values(key_m).reset_index(drop=True)
    F = F.sort_values(key_f).reset_index(drop=True)
    I = I.sort_values(key_i).reset_index(drop=True)

    summaries = []
    for fs in FEATURE_SETS:
        Mfs, Ffs, Ifs = M[M.feature_set == fs], F[F.feature_set == fs], I[I.feature_set == fs]
        Mfs[Mfs.county.isin(top)].to_csv(out_dir / f"top10_metrics_{fs}.csv", index=False)
        Ffs[Ffs.county.isin(top)].to_csv(out_dir / f"top10_forecasts_{fs}.csv", index=False)
        Ifs[Ifs.county.isin(top)].to_csv(out_dir / f"top10_feature_importance_{fs}.csv", index=False)
        for c in top:
            if (Mfs.county == c).any():
                s = summarize_best(dict(metrics=Mfs[Mfs.county == c], importances=Ifs[Ifs.county == c],
                                        meta={"county": c}))
                s["feature_set"] = fs
                summaries.append(s)
    top_tab = pd.DataFrame(summaries)
    for k in range(3):
        top_tab[f"feature_{k + 1}"] = top_tab["top_features"].map(lambda f, k=k: f[k] if len(f) > k else None)
    top_tab.to_csv(out_dir / "top10_best_model_top3.csv", index=False)

    Mr = M[(M.feature_set == "media_env") & M.county.isin(rest)]
    Ir = I[(I.feature_set == "media_env") & I.county.isin(rest)]
    rem_rows = [summarize_best(dict(metrics=Mr[Mr.county == c], importances=Ir[Ir.county == c], meta={"county": c}))
                for c in rest if (Mr.county == c).any()]
    rem = pd.DataFrame(rem_rows)
    for k in range(3):
        rem[f"feature_{k + 1}"] = rem["top_features"].map(lambda f, k=k: f[k] if len(f) > k else None)
    cols = ["county", "best_model", "feature_1", "feature_2", "feature_3", "top_media_feature",
            "features_from_model", "mean_rmse", "mean_rmse_seasonal_naive",
            "rmse_1m", "mae_1m", "rmse_3m", "mae_3m", "rmse_6m", "mae_6m"]
    rem[cols].round(4).to_csv(out_dir / "remaining_counties_best_model_top3.csv", index=False)
    Mr.to_csv(out_dir / "remaining_metrics_media_env_all_models.csv", index=False)
    Ir.to_csv(out_dir / "remaining_feature_importance_media_env.csv", index=False)

    meta_path = out_dir / "run_meta.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    for f in sorted(parts.glob("*_meta.json")):
        meta[f.stem] = json.load(open(f))
    json.dump(meta, open(meta_path, "w"), indent=2, default=str)
    print(f"combined: {len(M)} metric rows, {len(F)} forecast rows, {len(top_tab)} top-10 rows, {len(rem)} remaining counties")
    return M, top_tab, rem


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Walk-forward forecasting (media + environment)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="fit models and save new results to <out-dir>/parts")
    r.add_argument("--data-dir", default="data")
    r.add_argument("--out-dir", default="results/media_env_walk_forward")
    r.add_argument("--scope", choices=["top", "remaining", "all"], default="all")
    r.add_argument("--counties", nargs="+", default=None, help="explicit county list (overrides --scope)")
    r.add_argument("--feature-sets", nargs="+", default=list(FEATURE_SETS))
    r.add_argument("--models", nargs="+", default=list(MODELS))
    r.add_argument("--tag", default="run", help="label for this run; use a new tag per run")
    c = sub.add_parser("combine", help="merge new runs into the saved results (nothing is dropped)")
    c.add_argument("--data-dir", default="data")
    c.add_argument("--out-dir", default="results/media_env_walk_forward")
    a = ap.parse_args()
    if a.cmd == "run":
        run_parts(a.data_dir, a.out_dir, a.scope, a.feature_sets, a.models, a.tag, a.counties)
    else:
        combine(a.data_dir, a.out_dir)
