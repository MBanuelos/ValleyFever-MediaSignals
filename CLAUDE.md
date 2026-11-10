# CLAUDE.md — ValleyFever-MediaSignals

UChicago DSI Valley Fever project. This repo holds two strands: the **media-signal** work (teammates) and the **environmental forecasting** work (Austin). These notes are for Claude and anyone else working on the environmental side.

## Ownership

- **Environmental forecasting (Austin):**
  - `src/env_forecast/`
  - `notebooks/env_*.ipynb`
  - `src/county_env_data_processing.py`
  - `src/utils.py`
  - `scripts/env_*.py`
  - env configs in `configs/`
  - env tests in `tests/`
  - `paper/`
- **Media signals and dashboard (teammates):**
  - `src/fucntions.py` / `functions.py`
  - `src/numpy_lstm.py`
  - files with "media" in the name
  - `build_dashboard.py`, `dashboard*.html`, `dashboard_config.json`, `DASHBOARD.md`
- **Shared or unclear:** `src/data_validation.py`.
- Don't edit another person's files without asking them; integrate through new files or adapters instead.
- `main` is the PR base.

## Study design (fixed)

- **Task:** monthly Valley Fever rate (`rate_per_100k`), 8 California counties, direct forecasts at 1, 3 and 6 months ahead.
- **Evaluation:** rolling origin. Retrain at every forecast month on a 120-month window, with strict information cutoffs:
  - case rates are available after 1 month;
  - environmental data after 1–2 months, per group.
  - Preprocessing (scaling, imputation) is fit on training rows only.
- **Periods:**
  - **Validation 2017–2020:** all tuning and feature decisions.
  - **Test 2021–2024:** scored once after freezing.
  - The config field is `test_start` (the old name `confirmation_start` is still accepted).
- **Frozen configuration:**
  - `configs/env_main_tuned.json`, experiment ID `env_main_tuned_753eb46c`, seeds 42–44.
  - The freeze record is `results/env/test_freeze.json`.
  - **Do not re-tune models or features on 2021–2024.** Any new analysis of the test period is descriptive and must not feed back into model choice.
- **Model set (from the PI):**
  - Baselines: AR (= LR on case history, AR(12)) and SARIMAX.
  - Test models: XGBoost and LSTM, each with feature sets R (case history), E (environment only) and RE (both).
  - Seasonal naive and climatology stay as zero-cost reference benchmarks. Climatology is the fair benchmark for E.
  - Ridge/OLS are diagnostics, not paper models.
  - SARIMAX always includes case history, so it has no E variant.
- **Key contrast: RE vs E.** RE vs R is secondary. Forecasting without recent cases (E) matters because case-data availability is uncertain.

### Environmental variable groups (`env_main_tuned`)

| Group | Variables | Delay (months) | Features |
|---|---|---|---|
| precipitation | `Precipitation` | 1 | lags 0–2; 6- and 12-month means |
| weather | `Avg Rel Hum (%)`, `Avg Wind Speed (mph)`, `Wind Run (miles)`, `windeventcount`, `Avg Soil Temp (F)` | 1 | lags 0–2 |
| air_quality | `AQI_PM25`, `AQI_PM10` | 1 | lags 0–2 |
| wildfire | `FIRE_Acres_Burned` | 2 | lags 0–2 |

Pesticide (fungicide/rodenticide, CDPR PUR) data is excluded from the main model. It is released 24–35 months late and adds nothing to RE; see `paper/pesticide_test_2026-10-26.md`.

## Terminology

- Say **"environmental feature"**, not "covariate".
- For sets of variables, say **"variable groups"** or "groups of environmental variables", not "family". The config key is still `env_families`.
- **Skill** = 1 − RMSE / RMSE(seasonal naive); above 0 means better than "same month last year".
- **Pooled change** = geometric mean across counties of the RMSE ratio, minus 1.

## Running

Use the `.venv313` environment.

```bash
.venv313/Scripts/python.exe -m pytest tests -q                                                   # ~4 min, 69 tests
.venv313/Scripts/python.exe -m src.env_forecast.run --config configs/env_main_tuned.json --validation-only --n-jobs 8
.venv313/Scripts/python.exe -m src.env_forecast.run --config <config> --seed 43 --case-lag 3 --n-jobs 8
.venv313/Scripts/python.exe scripts/env_model_comparison.py . env_main_tuned --period test     # comparison tables
.venv313/Scripts/python.exe scripts/env_pooled_significance.py .                             # pooled bootstrap + Holm
.venv313/Scripts/python.exe scripts/env_importance_analysis.py .                             # ablation, PFI, case delay; figs 4-6
.venv313/Scripts/python.exe scripts/env_plots.py .                                           # figs 1-3
```

- Each run writes `results/env/<name>_<hash>/`, containing:
  - `predictions.csv`, `metrics.csv`, `contrasts.csv`, `config.json`, `provenance.json`;
  - `importance.csv` when `importance_draws > 0`.
- The hash covers the full config, so an unchanged config reproduces the same folder.
- A run takes about 11–12 minutes for 8 counties with XGBoost and the LSTM and `--n-jobs 8`. PFI runs take about 14 minutes per seed.
- Harness details are in `src/env_forecast/README.md`.

## Key layout

- `src/env_forecast/`:
  - `config.py`, `harness.py` (rolling loop), `features.py`, `models.py`, `lstm.py`, `importance.py` (permutation importance), `run.py` (CLI)
- `configs/`:
  - `env_main.json` → `env_main_tuned.json` (frozen)
  - `ablation/no_<group>.json`
  - `importance/env_main_tuned_pfi.json`
  - `pesticide/`
- Figures:
  - `results/env/figures/` (figs 1–3)
  - `results/env/importance/` (figs 4–6 and tables)
  - `results/env/pooled_significance/`
- `paper/`: dated result notes. The latest are `test_results_2026-10-27.md` and `importance_and_case_delay_2026-10-27.md`.

## Current findings (2026-10-27)

- **RE beats E** for XGBoost by about 20% pooled at every horizon in both periods (Holm-significant). It holds at case delays of 0, 1 and 3 months.
- **XGBoost RE** is the only model with positive median skill at every horizon in both periods (test: 0.22 / 0.14 / 0.19). Its edge over AR(12) and SARIMAX is not significant in the test period.
- **Environment helps mainly at 6 months.** Precipitation is the only group clearly important under both ablation (+12% error without it at h=6) and permutation importance.
- **Importance varies by county:** precipitation is strong in Fresno, San Diego, San Luis Obispo and Tulare, and near zero in Kern, Los Angeles and Orange. This is descriptive only; do not choose per-county features from test-period results.
- Everything is predictive, not causal.

## Open follow-ups

- Split the weather group into soil temperature, wind (speed, run, event count) and humidity, and rerun the ablation and PFI. This takes about 2.5 hours.
- Test a trend proxy and 24- and 36-month precipitation means to explain the pesticide effect (validation period only).
- Run the LSTM in the case-delay analysis.
- Reproducibility note: on 2026-10-27 the frozen run's `predictions.csv` was rewritten during a period relabel. About 17% of values changed by roughly 1e-15, from CSV rounding. Regenerate from the frozen config if byte-exact files are needed.
