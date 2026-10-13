"""Build the self-contained dashboard HTML from the combined walk-forward result CSVs."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
import media_env_walk_forward as w  # noqa: E402

HERE = Path(__file__).parent
COLS = ["model", "horizon", "n_windows", "rmse", "mae", "rate_rmse", "rate_mae",
        "pct_improve_rmse", "pct_improve_mae"]


def build(data_dir, res_dir, template, out_html):
    res_dir = Path(res_dir)
    top, _ = w.top_counties(data_dir)
    media = pd.read_csv(Path(data_dir) / "interim" / "Media_per_county.csv")
    cases = media.groupby("County")["num_cases"].sum()

    per_set = {}
    for fs in w.FEATURE_SETS:
        per_set[fs] = dict(
            met=pd.read_csv(res_dir / f"top10_metrics_{fs}.csv"),
            fc=pd.read_csv(res_dir / f"top10_forecasts_{fs}.csv"),
            imp=pd.read_csv(res_dir / f"top10_feature_importance_{fs}.csv"))

    counties = []
    for name in top:
        sets = {}
        for fs, d in per_set.items():
            m = d["met"][d["met"].county == name]
            if m.empty:
                continue
            res = dict(metrics=m, importances=d["imp"][d["imp"].county == name], meta={"county": name})
            best = w.summarize_best(res, n_features=6)
            feats = []
            for f, v in zip(best["top_features"], best["top_feature_importance"]):
                kind = "media" if f in w.MEDIA_COLS else ("env" if f in w.ENV_COLS else "ar")
                feats.append(dict(name=f, imp=v, kind=kind))
            series = {}
            sub = d["fc"][d["fc"].county == name]
            for h, gh in sub.groupby("horizon"):
                series[str(h)] = {mod: [[r.target, round(r.actual_rate, 3), round(r.pred_rate, 3)]
                                        for r in gm.sort_values("target").itertuples()]
                                  for mod, gm in gh.groupby("model")}
            sets[fs] = dict(
                best=dict(model=best["best_model"], features_model=best["features_from_model"],
                          mean_rmse=best["mean_rmse"], base_mean_rmse=best["mean_rmse_seasonal_naive"]),
                metrics=m[COLS].round(5).to_dict("records"), features=feats, forecasts=series)
        counties.append(dict(name=name, cases=int(cases[name]), sets=sets))

    any_fc = per_set["media_env"]["fc"]
    payload = dict(counties=counties, meta=dict(
        first_forecast_month=str(any_fc["target"].min()),
        last_forecast_month=str(any_fc["target"].max()),
        n_counties_in_data=int(media["County"].nunique())))
    html = Path(template).read_text().replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")))
    Path(out_html).write_text(html)
    return payload


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2], HERE / "dashboard_template.html", sys.argv[3])
