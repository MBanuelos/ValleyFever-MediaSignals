"""
Build the self-contained dashboard.html from the saved walk-forward results.

Run from the repo root:   python build_dashboard.py
Options:                  python build_dashboard.py --data-dir data --results-dir results/media_env_walk_forward --out dashboard.html
Check without writing:    python build_dashboard.py --check      (exit code 1 if dashboard.html is out of date)

What appears on the page is driven by two things:
  * the CSVs in the results folder (counties, models and feature sets found there are shown automatically)
  * dashboard_config.json (display names, colors, intro text)
A model or feature set that is in the results but not in the config still appears, with an automatic
label and color, so adding one never requires editing the template.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "src"))
import media_env_walk_forward as w  # noqa: E402

COLS = ["model", "horizon", "n_windows", "rmse", "mae", "rate_rmse", "rate_mae",
        "pct_improve_rmse", "pct_improve_mae"]
PALETTE = ["#1F8FA3", "#7A6C3A", "#5B7F95", "#A0527C", "#4C8C6B", "#B5651D", "#6B5B95", "#2E7D8C"]


def _title(key):
    return key.replace("_", " ").title()


def render(data_dir, results_dir, template, config_path):
    """Return (html, payload) without writing anything."""
    res_dir = Path(results_dir)
    cfg = json.loads(Path(config_path).read_text()) if Path(config_path).exists() else {}
    cfg.setdefault("intro", "")
    cfg.setdefault("feature_sets", [])
    cfg.setdefault("models", {})
    cfg.setdefault("feature_labels", {})

    top, _ = w.top_counties(data_dir)
    media = pd.read_csv(Path(data_dir) / "interim" / "Media_per_county.csv")
    cases = media.groupby("County")["num_cases"].sum()

    # feature sets: configured ones first, then any others that have result files
    known = {s["key"]: s for s in cfg["feature_sets"]}
    found = sorted(p.name[len("top10_metrics_"):-4] for p in res_dir.glob("top10_metrics_*.csv"))
    set_keys = [k for k in known if k in found] + [k for k in found if k not in known]
    sets_cfg = [dict(key=k, title=known.get(k, {}).get("title", _title(k))) for k in set_keys]

    per_set = {}
    for fs in set_keys:
        per_set[fs] = dict(
            met=pd.read_csv(res_dir / f"top10_metrics_{fs}.csv"),
            fc=pd.read_csv(res_dir / f"top10_forecasts_{fs}.csv"),
            imp=pd.read_csv(res_dir / f"top10_feature_importance_{fs}.csv"))

    # models: configured ones first (in config order), then any others found in the results
    present = []
    for d in per_set.values():
        for m in d["met"]["model"].unique():
            if m not in present:
                present.append(m)
    ordered = [m for m in cfg["models"] if m in present] + [m for m in present if m not in cfg["models"]]
    models_cfg, extra = {}, 0
    for m in ordered:
        c = cfg["models"].get(m, {})
        if "color" not in c:
            c = dict(c, color=PALETTE[extra % len(PALETTE)]); extra += 1
        models_cfg[m] = dict(label=c.get("label", _title(m)), color=c["color"])

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
                          mean_rmse=round(best["mean_rmse"], 5), base_mean_rmse=round(best["mean_rmse_seasonal_naive"], 5)),
                metrics=m.sort_values(["model", "horizon"])[COLS].round(5).to_dict("records"), features=feats, forecasts=series)
        counties.append(dict(name=name, cases=int(cases[name]), sets=sets))

    any_fc = next(iter(per_set.values()))["fc"]
    payload = dict(
        counties=counties,
        config=dict(intro=cfg["intro"], feature_sets=sets_cfg, models=models_cfg,
                    feature_labels=cfg["feature_labels"]),
        meta=dict(first_forecast_month=str(any_fc["target"].min()),
                  last_forecast_month=str(any_fc["target"].max()),
                  n_counties_in_data=int(media["County"].nunique())))
    html = Path(template).read_text().replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")))
    return html, payload


def build(data_dir, results_dir, template, config_path, out_html):
    html, payload = render(data_dir, results_dir, template, config_path)
    Path(out_html).write_text(html)
    print(f"wrote {out_html}: {len(payload['counties'])} counties, {len(payload['config']['feature_sets'])} "
          f"feature sets, {len(payload['config']['models'])} models")
    return payload


_DATA_RE = re.compile(r"^const DATA = (\{.*\});$", re.M)


def _split(html):
    m = _DATA_RE.search(html)
    if not m:
        raise ValueError("could not find the embedded data in the dashboard HTML")
    return html[:m.start(1)] + "/*DATA*/" + html[m.end(1):], json.loads(m.group(1))


def _diffs(a, b, path="", tol=1e-6, out=None):
    """Collect differences between two JSON values; numbers match within a small tolerance."""
    out = [] if out is None else out
    if len(out) >= 8:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: present in only one version")
            else:
                _diffs(a[k], b[k], f"{path}/{k}", tol, out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                _diffs(x, y, f"{path}[{i}]", tol, out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if abs(a - b) > tol * max(1.0, abs(a), abs(b)):
            out.append(f"{path}: {a} vs {b}")
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


def check(data_dir, results_dir, template, config_path, html_path):
    """Rebuild in memory and compare with the committed dashboard. Returns 0 if up to date, else 1."""
    if not Path(html_path).exists():
        print(f"FAIL: {html_path} does not exist. Run: python build_dashboard.py")
        return 1
    fresh_html, _ = render(data_dir, results_dir, template, config_path)
    fresh_skel, fresh_data = _split(fresh_html)
    old_skel, old_data = _split(Path(html_path).read_text())
    problems = []
    if fresh_skel != old_skel:
        problems.append("page layout or text (dashboard_template.html / dashboard_config.json) differs from the committed page")
    problems += _diffs(old_data, fresh_data)
    if problems:
        print(f"FAIL: {html_path} is out of date with its inputs.")
        for p in problems:
            print("  -", p)
        print("Fix: run  python build_dashboard.py  and commit the updated dashboard.html")
        print("(If you changed result CSVs, also commit them with git add -f so the check can see them.)")
        return 1
    print(f"OK: {html_path} matches a fresh build.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results-dir", default="results/media_env_walk_forward")
    ap.add_argument("--template", default=str(HERE / "dashboard_template.html"))
    ap.add_argument("--config", default=str(HERE / "dashboard_config.json"))
    ap.add_argument("--out", default="dashboard.html")
    ap.add_argument("--check", action="store_true", help="do not write; fail if --out is out of date")
    a = ap.parse_args()
    if a.check:
        sys.exit(check(a.data_dir, a.results_dir, a.template, a.config, a.out))
    build(a.data_dir, a.results_dir, a.template, a.config, a.out)
