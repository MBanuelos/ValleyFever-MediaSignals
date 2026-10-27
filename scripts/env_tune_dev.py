"""
Development-period tuning for the paper model set. Never touches confirmation targets.

  sarima-orders : choose SARIMA orders per county by mean AIC over training windows that end at
                  development-period origins (AIC uses training data only, no evaluation targets).
  lstm-grid     : screen LSTM settings on thinned development targets (every `--step`th month, one seed);
                  score = mean log(RMSE_lstm / RMSE_seasonal_naive) over counties x horizons x feature sets.

Usage:
  python scripts/env_tune_dev.py sarima-orders --config configs/env_main.json
  python scripts/env_tune_dev.py lstm-grid --config configs/env_main.json --stage 1
  python scripts/env_tune_dev.py lstm-grid --config configs/env_main.json --stage 2
"""
import argparse
import dataclasses
import itertools
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from statsmodels.tsa.statespace.sarimax import SARIMAX  # noqa: E402

from src.env_forecast import HarnessConfig, build_models, load_panel, mask_history, run_panel  # noqa: E402
from src.env_forecast.run import development_only  # noqa: E402

TUNING_DIR = ROOT / "results" / "env" / "tuning"
ORDER_GRID = [((p, 0, q), (P, 0, Q, 12)) for p, q, P, Q in itertools.product((0, 1, 2), (0, 1, 2), (0, 1), (0, 1))]
AIC_ORIGINS = ("2016-12", "2017-12", "2018-12", "2019-12")


def county_aic(county, frame, config):
    rows = []
    for origin in map(lambda o: pd.Period(o, "M"), AIC_ORIGINS):
        end = origin - config.rate.availability_lag
        values = mask_history(frame, origin, config)[config.target_col].loc[end - config.window_months + 1:end]
        endog = np.log1p(values.to_numpy(float))
        for order, seasonal in ORDER_GRID:
            try:
                with warnings.catch_warnings(), threadpool_limits(limits=1):
                    warnings.simplefilter("ignore")
                    aic = SARIMAX(endog, order=order, seasonal_order=seasonal, trend="c").fit(disp=False).aic
            except (ValueError, np.linalg.LinAlgError):
                aic = np.nan
            rows.append({"county": county, "origin": str(origin), "order": str(order), "seasonal_order": str(seasonal), "aic": aic})
    return rows


def sarima_orders(config):
    panel = load_panel(config, ROOT)
    rows = Parallel(n_jobs=len(panel))(delayed(county_aic)(c, f, config) for c, f in panel.items())
    table = pd.DataFrame([r for county_rows in rows for r in county_rows])
    mean_aic = table.groupby(["county", "order", "seasonal_order"])["aic"].agg(["mean", "count"]).reset_index()
    mean_aic = mean_aic[mean_aic["count"] == len(AIC_ORIGINS)]  # orders that fitted at every window
    best = mean_aic.loc[mean_aic.groupby("county")["mean"].idxmin()]
    chosen = {r.county: {"order": list(eval(r.order)), "seasonal_order": list(eval(r.seasonal_order))} for r in best.itertuples()}
    overall = mean_aic.groupby(["order", "seasonal_order"])["mean"].sum().idxmin()
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TUNING_DIR / "sarima_aic_by_window.csv", index=False)
    (TUNING_DIR / "sarima_orders.json").write_text(json.dumps({"county_orders": chosen, "global_best": overall,
                                                              "aic_origins": AIC_ORIGINS}, indent=2))
    current = mean_aic[(mean_aic.order == "(1, 0, 1)") & (mean_aic.seasonal_order == "(1, 0, 1, 12)")].set_index("county")["mean"]
    print(best.set_index("county")[["order", "seasonal_order", "mean"]].join(current.rename("aic_current_(1,0,1)(1,0,1)")).round(1).to_string())
    print("global best (sum of mean AIC):", overall)


LSTM_STAGE1 = [{"lookback": lb, "hidden_size": hs, "dropout": dp, "learning_rate": 1e-3}
               for lb, hs, dp in itertools.product((6, 12, 24), (8, 32), (0.1, 0.3))]


def score(predictions):
    ref = predictions[predictions.model == "seasonal_naive"].set_index(["county", "horizon", "target_month"])["predicted_rate"]
    lstm = predictions[predictions.model == "lstm"].join(ref.rename("ref"), on=["county", "horizon", "target_month"])
    rows = []
    for (fs, county, h), g in lstm.dropna(subset=["predicted_rate", "actual_rate", "ref"]).groupby(["feature_set", "county", "horizon"]):
        rows.append({"feature_set": fs, "county": county, "horizon": h,
                     "log_ratio": np.log(np.sqrt(((g.predicted_rate - g.actual_rate) ** 2).mean()) / np.sqrt(((g.ref - g.actual_rate) ** 2).mean()))})
    rows = pd.DataFrame(rows)
    return rows.log_ratio.mean(), rows.groupby("feature_set").log_ratio.mean().to_dict()


def lstm_grid(config, stage, step, seed):
    base = dataclasses.replace(development_only(config), models=("seasonal_naive", "lstm"), eval_step=step, seed=seed)
    current = dict(config.model_params.get("lstm", {}))
    summary_path = TUNING_DIR / "lstm_grid.csv"
    done = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    if stage == 1:
        candidates = LSTM_STAGE1
    else:
        top = done[done.stage == 1].nsmallest(3, "score")
        candidates = [{**json.loads(r), "learning_rate": 3e-3} for r in top["settings"]]
    panel = load_panel(base, ROOT)
    records = []
    for settings in candidates:
        params = {**current, **settings}
        cfg = dataclasses.replace(base, model_params={**config.model_params, "lstm": params})
        t0 = time.time()
        predictions = run_panel(panel, cfg, build_models(cfg), n_jobs=len(panel))
        total, by_set = score(predictions)
        records.append({"stage": stage, "settings": json.dumps(settings, sort_keys=True), "score": total,
                        **{f"score_{k}": v for k, v in by_set.items()}, "seconds": round(time.time() - t0)})
        print(records[-1], flush=True)
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([done, pd.DataFrame(records)], ignore_index=True).to_csv(summary_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["sarima-orders", "lstm-grid"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cfg = HarnessConfig.from_json(args.config)
    sarima_orders(cfg) if args.task == "sarima-orders" else lstm_grid(cfg, args.stage, args.step, args.seed)
