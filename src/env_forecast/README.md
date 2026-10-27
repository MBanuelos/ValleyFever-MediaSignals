# `env_forecast` — rolling-origin harness for environmental forecasting

This is one evaluation framework shared by every environmental model (baselines, AR/linear, SARIMAX, XGBoost, LSTM, and anything added later).

It enforces three things:

- **Chronological information cutoffs.**
- **Training-only preprocessing.**
- **Identical target months across models, feature sets and horizons.**

The existing `notebooks/env_*.ipynb` files are untouched. They still run as before; this package is the replacement evaluation path for results meant for the paper.

## How a forecast is made

For every county, horizon `h` and target month `m` in `[eval_start, eval_end]`:

1. **Origin.** The forecast is made at `t = m - h`. Because evaluation is defined by target month, every horizon is scored on the same targets.
2. **Mask** (`data.mask_history`). Keep only rows `<= t`. Blank the case rate after `t - case_lag`, and blank each environmental column after `t - availability_lag` of its family. Models only ever see this masked frame, or features built from it.
3. **Design** (`features.build_design`). Each row is a *pseudo-origin* `s`.
   - Features read months `<= s - availability_lag`, so training rows obey the same publication delays as the forecast row.
   - The label is `y[s + h]`. Labels not yet published at `t` are NaN.
4. **Window** (`harness.training_rows`). The training rows are the `window_months` most recent *published* labels, `s + h ∈ [t - case_lag - W + 1, t - case_lag]`.
   - Label availability is what limits the longer horizons. With W = 120 and a 1-month case lag, every horizon has 120 training rows, ending at the last published month.
   - A runtime guard raises if any label falls after the cutoff.
5. **Task** (`tasks.ForecastTask`). The harness passes the model one object holding `history`, `X_train`, `y_train`, `X_forecast` (the row `s = t`), `label_months`, `origin`, `horizon`, `feature_set`, `config` and `seed`.
6. **Model** returns one rate-scale prediction. The harness records it with the actual value and the metadata.

```
panel ──mask_history(t)──▶ history ──build_design(h)──▶ Design(X, y)
                                                        │ training_rows(t, h, W)
                                                        ▼
                         ForecastTask ──▶ model.fit_predict(task) ──▶ prediction row
```

## Modules

| Module | Responsibility |
|---|---|
| `config.py` | `HarnessConfig`, `RateSpec`, `FeatureBlock` (an environmental family with its publication delay, lags and rolling means). Loaded from JSON; hashed into the experiment id. |
| `data.py` | `load_panel` / `panel_from_frame` (validation, optional *causal* forward fill), `mask_history`. |
| `features.py` | `build_design`: rate lags, same-month-last-year (only if published and not a duplicate lag), environmental lags and rolling means, target-month sin/cos. Defines feature sets `R`, `E`, `RE`. |
| `tasks.py` | `ForecastTask`: the only thing a model sees. |
| `models.py` | `ForecastModel` protocol; baselines (`seasonal_naive`, `persistence`, `climatology`); `TabularModel`; `Sarima` / `Sarimax`; the `MODEL_BUILDERS` registry (`linear`, `ridge`, `lr`, `sarima`, `sarimax`, `xgboost`, `lstm`). |
| `lstm.py` | `LSTMModel`: a direct network for each horizon. Inputs are monthly rows, each variable shifted by its own publication delay; scaling comes from training data only; early stopping, then a refit on all training sequences. Torch is imported only when an LSTM is requested. |
| `imputation.py` | Origin-consistent Holt–Winters forward fill for late-published families (`forecast_fill: "ets"`). |
| `harness.py` | `run_county` / `run_panel`: the loop over county × horizon × target × model × feature set. `n_jobs` runs counties in parallel with results identical to a serial run. |
| `metrics.py` | `summarize` (RMSE/MAE/bias, skill vs the reference on paired targets), `paired_contrasts` (feature set A vs B per model, Diebold–Mariano with HLN correction and Newey–West h−1 lags), `diebold_mariano`. |
| `run.py` | CLI. Writes `results/env/<experiment_id>/`. |

**Feature sets:**

| Set | Contents |
|---|---|
| `R` | Rate history + calendar |
| `E` | Environmental families + calendar |
| `RE` | Both |

Baselines run once with feature set `none`. SARIMA runs once and is labelled `R`, because it uses case history only. The default contrasts are `RE vs E` (primary) and `RE vs R`.

## How each model plugs in

Every model implements:

```python
class ForecastModel(Protocol):
    name: str
    uses_features: bool          # True -> run once per feature set; False -> once, labelled feature_set_label (default "none")
    def fit_predict(self, task: ForecastTask) -> float: ...   # rate-scale prediction for task.target_month
```

**Direct tabular models** (AR/linear, Ridge, XGBoost, LightGBM): wrap an estimator factory in `TabularModel`.

- Put all preprocessing *inside* the estimator, usually a `Pipeline`. It is then fitted on `X_train` only and merely applied to `X_forecast`.
- `log_target=True` trains on `log1p(rate)` and back-transforms with `expm1`.
- AR(12) is `linear` (or `ridge`) with feature set `R`.
- **`xgboost`** is registered this way (`xgboost_factory`):
  - Defaults: 300 trees, depth 2, learning rate 0.05, subsample and column sample 0.8.
  - Override them via `model_params.xgboost.params`.
  - No imputer or scaler: XGBoost handles NaN natively.
  - `n_jobs=1` and the task seed make fits deterministic.
  - No early stopping yet; if added, its validation split must come from the end of `X_train`.

**Sequence and state-space models** (SARIMAX, LSTM): use `task.history`, which is already masked at the origin.

- **`sarima`** (`Sarima`) works on case history only:
  - Fits on the last `window_months` published rates (ending at `origin - case_lag`), on the log1p scale by default.
  - Forecasts `h + case_lag` steps and returns the last step.
  - Order is fixed in the config: `(1,0,1)(1,0,1,12)` with a constant. It is not re-selected at each origin.
  - One fit serves all horizons. The cache is keyed by the exact training values, so a fit can never be reused for different data.
  - Fits run with one BLAS thread. Otherwise the likelihood optimum shifts with the thread count, so serial and parallel runs disagree.
  - Failed fits return NaN and are counted in `Sarima.failures`.
- **Environmental SARIMAX** (future work): environmental inputs must be lags ≥ (steps ahead + availability lag), so their future values are already in `history`. Never feed realised future values.
- **LSTM:** build sequences from `task.history` (or reuse `X_train` rows as the per-step feature vectors).
  - Fit scalers on the training portion only, and use early stopping on the last part of the window.
  - `task.seed` is set; run several seeds by varying `config.seed`.

Whatever a model does internally, it cannot see beyond the task, so the invariance tests cover it automatically.

## Outputs (`results/env/<experiment_id>/`)

| File | Contents |
|---|---|
| `predictions.csv` | One row per forecast: `experiment_id, county, model, feature_set, horizon, forecast_origin, target_month, period, seed, actual_rate, predicted_rate, train_start, train_end, n_train, rate_info_cutoff, env_info_cutoff` |
| `metrics.csv` | Per county × model × feature set × horizon × period (`development`, `confirmation`, `all`): `n, rmse, mae, bias, rmse_reference, skill_vs_reference` |
| `contrasts.csv` | Per model × contrast × county × horizon × period: `rmse_a, rmse_b, relative_rmse_change, dm_statistic, dm_p_value` |
| `config.json`, `provenance.json` | Full settings; data SHA-256, git commit, dirty flag, package versions |

`experiment_id = <name>_<config hash>`, so an unchanged config always writes to the same folder.

## Paper model set (`configs/env_main.json`)

| Role | Model | Inputs |
|---|---|---|
| Reference | `seasonal_naive` (skill = 0), `climatology` | — |
| Baseline | `lr`: ridge regression with the penalty chosen by TimeSeriesSplit CV inside each training window. With R it uses only the 12 rate lags, i.e. **AR(12)**. | R, E, RE |
| Baseline | `sarima` (R) and `sarimax` (RE): each regressor enters as x[m − (h + availability lag)], so no unpublished value is used; one fit per horizon | R / RE |
| Test | `xgboost`, `lstm` | R, E, RE |

Configs can inherit from one another with `"extends": "<relative path>"`; a `null` value removes the inherited key (e.g. `"ag_chemicals": null`).

## Running

From the project root, in the project venv (`.venv313`):

```bash
# tests (requires: pip install pytest; about 4 minutes)
.venv313/Scripts/python.exe -m pytest tests -q

# development period only (targets before confirmation_start); confirmation targets are never forecast
# about 8 minutes for 8 counties with all models and --n-jobs 8
.venv313/Scripts/python.exe -m src.env_forecast.run --config configs/env_baseline.json --development-only --n-jobs 8

# full run, including the confirmation period: only once configurations are frozen
.venv313/Scripts/python.exe -m src.env_forecast.run --config configs/env_baseline.json --n-jobs 8

# subset of counties
.venv313/Scripts/python.exe -m src.env_forecast.run --config configs/env_baseline.json --counties Fresno Kern
```

From a notebook:

```python
import sys; sys.path.append(str(project_root))
from src.env_forecast import HarnessConfig, load_panel, run_panel, build_models, summarize
config = HarnessConfig.from_json(project_root / "configs/env_baseline.json")
predictions = run_panel(load_panel(config, project_root), config, build_models(config))
```

## What the tests guarantee (`tests/test_env_alignment.py`, `tests/test_env_models.py`)

The invariance test below runs every registered model, including `sarima` and `xgboost`. `test_env_models.py` adds model-specific checks:

- Data older than the rolling window has no effect. For SARIMA this includes rates before the first training label.
- SARIMA equals a hand-made fit on the last W published rates, forecast h + case lag steps, for case lags 0, 1 and 3.
- The SARIMA cache never reuses a fit for different data.
- New models share the same target months; SARIMA is labelled `R`.
- XGBoost is reproducible and responds to the seed.
- Parallel runs match serial runs.



The synthetic panel stores, in every cell, the position of its own month, so every feature and label can be traced to its source month.

- Masking respects each family's delay.
- Every design cell reads exactly the intended month, for h ∈ {1, 3, 6, 12} and case lag ∈ {0, 1, 3}.
- Labels end at `origin - case_lag`, and the window length is honoured (60 and 120 months).
- A spy model run through the full loop never sees unpublished data, and target − origin = h.
- **Invariance:** perturbing everything unpublished at the origin (including the target) leaves every prediction unchanged, for all models and feature sets. A paired test checks that published data *does* move predictions, so the invariance test cannot pass vacuously.
- A known driver (rate = 10 + 2·x[m−3]) is recovered exactly when it is published by the origin, and not when it isn't.
- Baseline definitions, NaN labels and actuals, reproducibility, reference skill = 0, and the direction of the DM test.

Mutation check: deliberately introducing an off-by-one label window, an unmasked environmental delay, or an off-by-one rate lag makes 7, 2 and 13 tests fail respectively.

Analysis scripts (run after the matching harness runs; each writes under `results/env/`):

| Script | What it compares |
|---|---|
| `scripts/env_model_comparison.py` | Paper model set: skill, input contrasts, nonlinear vs linear |
| `scripts/env_case_lag_check.py` | Case-lag sensitivity |
| `scripts/env_seed_check.py` | XGBoost seed stability |
| `scripts/env_pesticide_check.py` | Pesticide variants |

## Known limitations

- Rolling means are computed from each window's own values (`features.window_mean`). pandas' running-sum rolling mean let rounding from older months reach later values, which moved SARIMAX fits.
- The LSTM uses raw shifted columns for every family. It does not use `forecast_fill`.
- Development-tuned settings are in `configs/env_main_tuned.json`: SARIMA(X) orders per county chosen by AIC, and LSTM settings chosen from a small grid (`scripts/env_tune_dev.py`; results in `paper/model_comparison_dev_2026-10-26.md` §5). Orders are fixed per county, not re-selected at each origin.

- Values are the *current vintage*. Revisions to past case counts are not modelled; availability lags approximate reporting delay only.
- Availability lags in `configs/env_baseline.json` are assumptions to verify (audit section D2).
- `expm1` back-transform with `log_target` gives a median-type forecast (biased low for skewed rates).
- No hyperparameter search yet. Tune only on `development` targets.
