"""
Command-line entry point.

    python -m src.env_forecast.run --config configs/env_baseline.json [--counties Fresno Kern]
                                   [--development-only] [--n-jobs 4] [--models xgboost] [--seed 7]

Writes results/env/<experiment_id>/ with predictions.csv, metrics.csv, contrasts.csv,
config.json and provenance.json. The experiment id is the config name plus a hash of the
full configuration, so rerunning an unchanged config reproduces the same folder.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from .config import HarnessConfig
from .data import load_panel
from .harness import run_panel
from .metrics import paired_contrasts, summarize
from .models import build_models

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def provenance(config: HarnessConfig) -> dict:
    """Input checksum, code version and package versions needed to reproduce a run."""
    data_path = Path(config.data_path) if Path(config.data_path).is_absolute() else PROJECT_ROOT / config.data_path
    packages = {}
    for package in ("numpy", "pandas", "scikit-learn", "scipy", "statsmodels", "xgboost", "torch"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "experiment_id": config.experiment_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_path": config.data_path,
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--", "src/env_forecast", "configs")),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "command": " ".join(sys.argv),
    }


def development_only(config: HarnessConfig) -> HarnessConfig:
    """Restrict evaluation to targets before confirmation_start, so confirmation targets are never forecast."""
    if config.confirmation_start is None:
        raise ValueError("development_only needs confirmation_start in the config.")
    last_development = pd.Period(config.confirmation_start, "M") - 1
    return dataclasses.replace(config, name=f"{config.name}_dev", eval_end=str(last_development))


def run_experiment(config: HarnessConfig, output_root: str | Path | None = None, n_jobs: int = 1) -> Path:
    """Run a configured experiment end to end and save its outputs; returns the output folder."""
    panel = load_panel(config, PROJECT_ROOT)
    predictions = run_panel(panel, config, build_models(config), n_jobs=n_jobs)
    predictions.insert(0, "experiment_id", config.experiment_id)

    output_dir = Path(output_root or PROJECT_ROOT / config.output_dir) / config.experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    summarize(predictions, config.reference_model).to_csv(output_dir / "metrics.csv", index=False)
    paired_contrasts(predictions, config.contrasts).to_csv(output_dir / "contrasts.csv", index=False)
    (output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2, default=str), encoding="utf-8")
    (output_dir / "provenance.json").write_text(json.dumps(provenance(config), indent=2), encoding="utf-8")
    return output_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rolling-origin environmental Valley Fever forecasts.")
    parser.add_argument("--config", required=True, help="JSON experiment configuration.")
    parser.add_argument("--counties", nargs="+", help="Override the configured counties.")
    parser.add_argument("--output-root", help="Override the output folder (default: config output_dir).")
    parser.add_argument("--development-only", action="store_true", help="Forecast only targets before confirmation_start.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Counties run in parallel (-1 = all cores).")
    parser.add_argument("--models", nargs="+", help="Override the configured models.")
    parser.add_argument("--seed", type=int, help="Override the configured seed.")
    parser.add_argument("--case-lag", type=int, help="Override the case-rate availability lag (months), for sensitivity runs.")
    args = parser.parse_args(argv)

    config = HarnessConfig.from_json(args.config)
    if args.counties:
        config = dataclasses.replace(config, counties=tuple(args.counties))
    if args.models:
        config = dataclasses.replace(config, models=tuple(args.models))
    if args.seed is not None:
        config = dataclasses.replace(config, seed=args.seed)
    if args.case_lag is not None:
        config = dataclasses.replace(config, name=f"{config.name}_caselag{args.case_lag}",
                                     rate=dataclasses.replace(config.rate, availability_lag=args.case_lag))
    if args.development_only:
        config = development_only(config)
    output_dir = run_experiment(config, args.output_root, args.n_jobs)

    metrics = pd.read_csv(output_dir / "metrics.csv")
    overview = metrics[metrics["period"] == "development"].pivot_table(
        index=["model", "feature_set"], columns="horizon", values="skill_vs_reference", aggfunc="median"
    )
    print(f"\nSaved {output_dir}")
    print(f"Median development skill vs {config.reference_model} across counties:\n{overview.round(3)}")


if __name__ == "__main__":
    main()
