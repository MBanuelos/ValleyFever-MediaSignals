"""Build the self-contained dashboard HTML from the walk-forward result CSVs."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
import media_env_walk_forward as w  # noqa: E402

HERE = Path(__file__).parent


def build(data_dir, res_dir, template, out_html):
    res_dir = Path(res_dir)
    top, _ = w.top_counties(data_dir)
    media = pd.read_csv(Path(data_dir) / "interim" / "Media_per_county.csv")
    cases = media.groupby("County")["num_cases"].sum()
    n_counties = media["County"].nunique()

    met = pd.read_csv(res_dir / "top10_metrics_media_env.csv")
    env = pd.read_csv(res_dir / "top10_metrics_env_only.csv")
    fc = pd.read_csv(res_dir / "top10_forecasts_media_env.csv")
    imp = pd.read_csv(res_dir / "top10_feature_importance_media_env.csv")
    meta = json.load(open(res_dir / "top10_run_meta.json"))["meta"]

    cols = ["model", "horizon", "n_windows", "rmse", "mae", "rate_rmse", "rate_mae",
            "pct_improve_rmse", "pct_improve_mae"]
    counties = []
    for name in top:
        res = dict(metrics=met[met.county == name], importances=imp[imp.county == name],
                   meta={"county": name})
        best = w.summarize_best(res, n_features=6)
        feats = []
        for f, v in zip(best["top_features"], best["top_feature_importance"]):
            kind = "media" if f in w.MEDIA_COLS else ("env" if f in w.ENV_COLS else "ar")
            feats.append(dict(name=f, imp=v, kind=kind))
        series = {}
        sub = fc[fc.county == name]
        for h, gh in sub.groupby("horizon"):
            series[str(h)] = {m: [[r.target, round(r.actual_rate, 4), round(r.pred_rate, 4)]
                                  for r in gm.sort_values("target").itertuples()]
                              for m, gm in gh.groupby("model")}
        counties.append(dict(
            name=name, cases=int(cases[name]),
            best=dict(model=best["best_model"], features_model=best["features_from_model"],
                      mean_rmse=best["mean_rmse"], base_mean_rmse=best["mean_rmse_seasonal_naive"]),
            metrics=met[met.county == name][cols].round(5).to_dict("records"),
            env_metrics=env[env.county == name][cols].round(5).to_dict("records"),
            features=feats, forecasts=series))

    payload = dict(counties=counties, meta=dict(
        first_forecast_month=meta[0]["first_forecast_month"],
        last_forecast_month=meta[0]["last_forecast_month"],
        n_counties_in_data=int(n_counties)))
    html = Path(template).read_text().replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")))
    Path(out_html).write_text(html)
    return payload


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2], HERE / "dashboard_template.html", sys.argv[3])
