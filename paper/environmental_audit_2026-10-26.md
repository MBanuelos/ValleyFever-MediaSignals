# Environmental Forecasting Audit — 2026-10-26

Scope: Austin's environmental forecasting component only. Teammate-owned code (`src/fucntions.py`, `src/numpy_lstm.py`, `src/media_env_walk_forward.py`, all `Media*`/`media*` notebooks, dashboard files) was read only where needed to understand shared interfaces, and was not modified.

Evidence labels used throughout:

- **[V]** Verified: I checked it directly in code, data, or by re-running it.
- **[R]** Reported: a notebook output or comment states it, but I did not re-run it.
- **[H]** Hypothesis: my inference, which still needs checking.

---

## A. Project reconstruction

### A.1 Code ownership map [V, from `git log`]

| File | Author(s) | Ownership |
|---|---|---|
| `src/county_env_data_processing.py` | Austin | **Yours** |
| `src/utils.py` | Austin | **Yours** (copied from class notebooks; matches `Desktop/context/scripts/utils.py`) |
| `notebooks/env_AR.ipynb`, `env_SARIMAX.ipynb`, `env_XGBoost.ipynb`, `env_LSTM.ipynb` | Austin | **Yours** |
| `src/data_validation.py` | Created by Austin; later edited by Zion (`0bb178b`, `7609c12`) | **Unclear.** Treat as read-only until the team agrees. |
| `src/fucntions.py`, `src/numpy_lstm.py`, `src/media_env_walk_forward.py`, `results/media_env_walk_forward/*` | Zion / Jules | Teammate (read-only) |
| `build_dashboard.py`, `dashboard*.html/json`, `DASHBOARD.md`, `.github/workflows/check-dashboard.yml` | Jules | Teammate (read-only) |

### A.2 What has been implemented

**Data pipeline (`src/county_env_data_processing.py`) [V]**

- `process_data()` loads `aggregate_all_counties_2001_2024.csv` and validates it (`validate_data`).
- It can trim each county to the first month with fire data (`trim_leading_fire_data`, which starts at 2006-08).
- It linearly interpolates internal gaps in every numeric feature (`interpolate_internal_missing`).
- It fills the missing 2024 fungicide/rodenticide values with a forecast from additive-seasonal ETS (`impute_trailing_ets`), then adds `log1p_rate_per_100k`.
- Outputs: `data/interim/top8_*` (no NaNs) and `top21_*` (still 1,269–3,147 NaNs, so **the 21-county files are not model-ready**).

**Current models (October 2026).** All four notebooks share one design:

- Per-county models.
- **One forecast origin (2024-06)**, expanding training window from 2006-08.
- Separate direct multi-output model per horizon (1, 3, 6).
- Each horizon's metric is the RMSE/MAE over lead months 1..h.

| Notebook | Model | Inputs | Notes |
|---|---|---|---|
| `env_AR` cell 2 `ar_model` | OLS on rate lags 0–11, multi-output | Rate only | Raw-rate target |
| `env_SARIMAX` cells 3–9 | `auto_arima` order selection, then statsmodels SARIMAX | Rate + 11 environmental features at the same month | log1p target; future environmental inputs = origin values repeated |
| `env_XGBoost` cells 5–7, 14 | `MultiOutputRegressor(XGBRegressor)` | Environmental lags 0–11 + month sin/cos (134 features, about 200 rows) | Environment-only run saved; gain importance |
| `env_LSTM` cells 3–17 | 1-layer LSTM, direct multi-output | Environment (+ optional rate) sequences | Train-only feature and target scalers; 400 fixed epochs |

**Summer notebooks (`Desktop/context/notebooks`) [V]**

- day3–6 and day8: SARIMAX and OLS, single 80/20 chronological split.
- day9: stationarity tests and transformation tests.
- day10–13: OLS on log1p-diff with selected lags. The notebook itself is titled "POSSIBLE_DATA_LEAKAGE".
- day11: ETS window study for pesticide imputation.
- day12: feature/lag screening using Le's `env_only_all_model_results.csv`, which is not in the repo, plus `TimeSeriesSplit` CV.
- day14–15 and day17: LSTM, Yeo-Johnson transforms, permutation importance (PFI) on the test split.
- day16: LightGBM with a small `TimeSeriesSplit` grid.
- All of them depend on Google Colab and Drive paths.

**Teammate walk-forward (`src/media_env_walk_forward.py`) [V, read-only]**

This is already a proper rolling-origin framework: 120-month rolling window, monthly origins, direct models per step, and imputation/scaling fitted inside each window. Its `env_only` set is **not environment-only**: it includes AR terms (`ar_lag0–2`, `seasonal_lag12`). Because it is tied to the media start date (2011), its evaluation window covers only 2021-02 to 2024-12 (47 origins). An environmental pipeline is not limited by that date and can evaluate from about 2017.

### A.3 Results obtained, and how far each one can be trusted

| Result | Status | Can I reproduce it? |
|---|---|---|
| AR / SARIMAX / XGBoost / LSTM RMSE tables at the 2024-06 origin (`results/{ar,sarimax,xgboost,lstm}/`, gitignored) | [R] in notebook outputs | **Yes, exactly.** Fresno AR h1 = 6.6115 and XGBoost h1 = 4.4308 both reproduced [V]. `env_AR` cell 1, however, points to `all_counties_processed_no_fire_2006_2024.csv`, which does not exist. It only ran because the file was later renamed to `top8_*`, so the notebook fails as written. |
| "Best LSTM: lookback 12, hidden 8, no rate, raw target" (`env_LSTM` cell 17) | [R] | Reproducible, but **chosen by looking at the test origin**. See B3. |
| Summer feature/lag rankings (day12) and LR/LSTM results (day10–17) | [R] | **No.** They need Colab/Drive files that are not in the repo (`monthly_dataset.csv`, `env_only_all_model_results.csv`), and several cells error (day10–13 cell 4 `KeyError`). |
| Teammate env+AR walk-forward metrics (`results/media_env_walk_forward/top10_metrics_env_only.csv`) | [R] | Not re-run. Methodologically the closest thing to valid, but it is not your result. |

**Confirmed findings so far: none that are publication-grade.** Every current result comes from 1–6 target months at a single origin.

One telling check [V]: for Fresno at the 2024-06 origin, **naive persistence has a 1-month error of −1.07, smaller than AR (−1.45), SARIMAX (−1.63), the "best" LSTM (−1.70) or XGBoost (−3.63).** No current notebook includes a persistence or seasonal-naive baseline.

---

## B. Code and methodology audit

Severity: 🔴 invalidates results · 🟠 biases results or makes them non-operational · 🟡 hygiene or reproducibility.

### Evaluation design

**B1 🔴 One forecast origin.**
- Where: all four `env_*` notebooks (`evaluation_date = "2024-06-01"`).
- Each county–horizon metric is computed from 1, 3 or 6 points. The h=1 "RMSE" is one absolute error.
- 2024 was an unusually high year (Fresno annual rate sum 99.4 vs 44–64 in 2020–2023; Kern 421.6; SLO 167.2) [V]. Every model underpredicts Fresno in 2024 [V].
- Rankings at this origin say nothing about general skill.
- Fix: rolling origins (Section D).

**B2 🟠 "Horizon" means the path average over 1..h, not the target month t+h.**
- Where: `env_AR` cell 2, `evaluate_sarimax_horizons` (cell 7), `run_xgboost_county` / `run_lstm_county` metric rows.
- The 6-month score mixes lead-1 to lead-6 errors. This understates how quickly skill decays and does not match your stated definition ("predict t+h").
- Fix: report the h-step-ahead error (target = t+h) as the primary measure. The path average can be a secondary, labelled metric. The teammate pipeline uses the path average too, so this affects comparability across the team.

**B3 🔴 Model configuration chosen on the test data.**
- Where: `env_LSTM` cells 12–17 try lookback 12/24 × log/raw × hidden 8/16 and label the winner "BEST MODEL SO FAR" using the same 2024-06 test targets.
- Same problem in the summer work:
  - day10–13 uses lags/features "selected from day 12".
  - day12 screened using everything except the last 12 months (through 2023-12, `test_months = 12`, cell 49).
  - day10–13 then tested on the last 20% of 2006-08..2023-12 (about 2020-06..2023-12), which is **inside** the day-12 development data.
- That is the leakage the notebook title suspected. It is real, though it comes from feature selection, not from the features themselves.

**B4 🟠 No baselines, and no rate-only vs environment-only vs combined comparison on equal terms.**
- AR uses rate only, XGBoost/LSTM were run environment-only, and SARIMAX is rate + environment.
- These are different input configurations evaluated on different models, so the core question (does the environment add skill beyond incidence history?) cannot be answered from the current runs.

**B5 🟡 Expanding window, not the agreed 120-month rolling window.** All `env_*` notebooks train on 2006-08..origin.

### Information availability and leakage

**B6 🟠 SARIMAX uses environmental inputs from the same month as the target, and stale ones at forecast time.**
- Where: `prepare_sarimax_data` (cell 3).
- The model learns y_t ~ env_t, the relationship within a month. At forecast time it is given env_origin for t+1..t+6.
- This does not leak future values (good), but it is misspecified: the coefficients describe a relationship that is never used the same way at prediction time.
- Fix: use only environmental lags ≥ h (direct SARIMAX per horizon), or keep SARIMA without environmental inputs as the statistical baseline.

**B7 🟠 Environmental lag 0 and rate lag 0 assume real-time data.**
- Where: XGBoost `environmental_lags=(0,…)` and `rate_lags` from 0; LSTM sequences end at the origin; AR uses lag 0.
- Publication delays differ a lot by source [H, verify each]:
  - Weather (CIMIS-type stations) and AQI: roughly 0–1 month.
  - CAL FIRE acreage: weeks to months.
  - **CDPR Pesticide Use Reports: typically released about 1.5–2 years after the use year.**
  - **CDPH Valley Fever case data:** reporting delays plus later revisions; I could not determine from the data whether `rate_per_100k` is by onset date or report date.
- As currently coded, pesticides are roughly 18–24 months more current than anyone would have them in practice.
- Fix: an explicit per-feature availability-lag table (D2) and a case-data lag sensitivity analysis.

**B8 🟠 2024 test-period pesticide values are imputed.**
- Where: `impute_trailing_ets`.
- The 2024 fungicide/rodenticide values were never observed. They are ETS forecasts fitted once on all of 2001–2023. At the 2024-06 origin, every model's pesticide inputs are synthetic.
- Not future-leaking, but the 2024 "pesticide effect" is partly an ETS seasonal profile.
- Combined with B7, pesticides should not appear in the operational configuration except at lags ≥ 24, or in a configuration explicitly labelled retrospective.

**B9 🟡 Global preprocessing done before any split.**
- Where: `process_data()`.
- `interpolate_internal_missing` uses the next observed value, and the ETS imputation is fitted once on the whole series. The footprint in the 8 counties is small (Tulare PM10: 3 interpolated months [V]; pesticides: 12 months × 8 counties).
- Under rolling origins it becomes a leak if internal gaps sit near an origin.
- Fix: move imputation inside the window (forward-fill or a window-fitted ETS) and keep `process_data()` for validation only.
- The summer notebooks (day15, day17) also fit `PowerTransformer` on full county data [V].

**B10 ✅ Things done correctly [V]**
- XGBoost and LSTM training targets stop at the origin (`training_end = origin − h`; `last_training_origin = forecast_position − horizon`).
- LSTM scalers are fitted on training rows only.
- SARIMAX's scaler is fitted on the training data only.
- LSTM output dropout is genuinely active (`DirectLSTM.output_dropout`), even though `nn.LSTM` inter-layer dropout is correctly disabled when `num_layers=1`. So the "dropout 0.4" setting does affect the model.

### Model implementation

**B11 🟠 Too many XGBoost features for the sample size.**
- 134 features on about 200 rows, with 500 trees and no early stopping or validation.
- The feature importances (cell 11) are gain importances from one fit. Highly correlated lags (precipitation lag 3 vs 4; wind speed vs wind run) split credit arbitrarily, so these are not usable as scientific evidence.

**B12 🟠 LSTM training is not tuned on any validation data.**
- Fixed 400 epochs, no validation split, no early stopping; one seed (42); about 190 training sequences per model.
- Seed-to-seed variance has not been measured, so it is unknown whether differences between configurations exceed seed noise [H: very likely they do not at n=6 targets].

**B13 🟡 Back-transform bias.**
- `expm1(mean of log1p)` estimates a median, not a mean, so predictions are biased low on the rate scale. This matters most for skewed, high-incidence counties.
- Fix: smearing or variance correction, or train on the raw target with an appropriate loss. Report the chosen approach.

**B14 🟡 SARIMAX `auto_arima` runs stepwise AIC selection at every origin.**
- Under rolling origins this costs about 30 s × origins × counties, and the selected orders will change from window to window (instability that should be reported).
- Option: select the order once per county on the development period only, then refit only the coefficients.

### Data

**B15 🟠 County eligibility, checked against the data [V]** (2006-08..2024-12 window):

| County | Issue | Recommendation |
|---|---|---|
| San Mateo | Precipitation and all weather variables missing 2006-08 to 2017-07 (132 months); PM10 missing 2008-07 onward | **Exclude.** The data supports the team's suggestion. |
| Alameda | Only PM10 is missing (2011-01 onward); everything else complete | **Do not exclude automatically.** Keep it, and drop PM10 for this county or in a sensitivity configuration without PM10. |
| Santa Clara | Weather missing 2008-06 to 2009-08 (15 months); PM10 missing 2023–24 | Keep. The weather gap falls in early training windows only, where within-window imputation is acceptable. |
| Merced | 13 missing target months (2006-08 to 2007-12, scattered); weather missing 2024-05 to 2024-12 | [H] The missing rates may be zero-case months stored as NaN. Check with the ValleyCast source before imputing anything. |
| Solano | PM10 missing 2020-01 to 2020-07 | Keep |

Low-incidence counties (Sacramento, Santa Clara, San Mateo, Solano, Alameda) have monthly means of 0.14–0.27 per 100k and 17–46% zero months [V]. Their monthly rates are effectively small counts, so RMSE there is close to Poisson noise and a single pooled RMSE will be dominated by Kern (mean 18.7). Use scale-free skill metrics (D5).

**B16 🟡 Reproducibility hygiene**
- `env_AR` loads a file that no longer exists (A.3).
- `data/interim/README.md` names files that don't exist (`all_counties_processed_*`), and the actual files use inconsistent spellings (`top21_..._process_...` vs `top8_..._processed_...`).
- The `_no_fire_2006` suffix actually means "trimmed to the start of fire data", the opposite of what it says.
- Every `env_*` notebook re-implements loading, feature building, plotting, and metrics (four copies), and `results/` is gitignored, so no result is versioned.
- `requirements.txt` is missing `pmdarima`, `torch`, `matplotlib` and `lightgbm`.
- No tests.

---

## C. Publication-readiness assessment

| Item | Verdict |
|---|---|
| Data aggregation (`env_vf_aggregate`) | Usable, after the case-definition, revision and Merced-NaN questions are documented. Provenance belongs in ValleyCast; record its commit/version in `data/manifests/`. |
| `process_data()` validation checks | Reusable. Imputation must move inside each training window. |
| All 2024-06 single-origin metrics (AR, SARIMAX, XGBoost, LSTM) | **Not publishable; re-evaluate.** Treat them as smoke tests. |
| The "best LSTM configuration" | **Not a finding.** It was selected on test data (B3). Use it only as a starting point for the grid. |
| Summer feature/lag rankings and PFI (day12, day15, day17) | **Not publishable.** Selection leakage, a single split, irreproducible inputs. Useful only to suggest candidate lags (e.g., precipitation 3–8 months, soil temperature) for development-period screening. |
| XGBoost gain importances | Not evidence. Replace with grouped PFI or ablation on held-out origins. |
| ETS pesticide imputation study (day11) | Methodologically fine as a sensitivity piece, but likely moot if pesticides go in at lag ≥ 24 (B7). |

**Evidence still missing for the paper:**

1. Multi-origin, out-of-sample errors on identical target months, compared with persistence, seasonal-naive and climatology baselines.
2. A paired rate-only vs rate+environment comparison with temporal-dependence-aware uncertainty.
3. Horizon decay curves.
4. Stability across sub-periods (pre-2020, 2020–21, 2022–24) and across seeds.
5. Grouped importance or ablation by environmental family.
6. Documented data availability lags.

---

## D. Recommended experimental methodology

### D1. Units, periods and splits

- **Panel:** county × month, 2006-08..2024-12 (bounded by fire data). This gives 221 months for the 8 core counties.
- **Common target months:** 2017-01..2024-12 (96 per county). The origin is t = target − h, and the first origin needs 120 months of history, so the earliest is 2016-07. Every model, configuration and horizon is scored on **the same 96 target months**, which makes horizons comparable too.
- **Two periods:**
  - **Development** (targets 2017-01..2020-12): choose lags, hyperparameters, feature families and "best model" here.
  - **Confirmation** (targets 2021-01..2024-12): run once with frozen configurations; these are the numbers reported in the paper.
  - This is the cheapest defensible guard against B3. An alternative is nested selection inside each window (last 24 months as validation), which is more expensive and not obviously better at this sample size.
- **Training window:** rolling, 120 target months ending at the origin, i.e. rows with s+h ∈ (t−120, t]. An expanding window is run once as a labelled sensitivity analysis.

### D2. Information cutoffs (direct forecasting)

At origin t, predict y(t+h) using only x(t−k) with k ≥ L_f for each feature f. Proposed L_f values are **assumptions to verify per source**:

| Family | Features | Proposed availability lag L_f |
|---|---|---|
| Case history | `rate_per_100k` | 1 month in the primary analysis; sensitivity 0 and 3 |
| Weather | precipitation, humidity, wind speed, wind run, wind events, soil temperature | 1 |
| Air quality | PM2.5, PM10 | 1 |
| Wildfire | `FIRE_Acres_Burned` | 2 |
| Agricultural chemicals | fungicide, rodenticide | **24**, or excluded from the operational config. A separate "retrospective (oracle-availability)" config may use L=1, labelled explicitly. |
| Calendar | sin/cos of the **target** month | 0 (known in advance) |

Each prediction is stored with an `info_cutoff` field per family so tests can assert it.

### D3. Input configurations (same models, same targets)

- **R**: rate history only (lags L_y..L_y+11, plus the same month last year).
- **E**: environmental features + calendar only, no rate history. Tests whether the environment alone carries signal.
- **R+E**: combined. **The key contrast is R+E vs R.**
- Environmental lags: a small, theory-guided set per family. For example, precipitation sums over 3–6, 7–12 and 13–24 months back; soil temperature and wind lags 1–3. Fixed in development, not searched in confirmation.

### D4. Models (manageable set)

| Model | Strategy | Notes |
|---|---|---|
| Persistence | — | ŷ(t+h) = y(t−L_y) |
| Seasonal naive | — | ŷ(t+h) = y(t+h−12) |
| Climatology | — | Mean of that calendar month over the training window. The natural baseline for configuration E. |
| AR / Ridge | Direct, one model per h | R, E, R+E. Ridge alpha chosen in development. |
| SARIMA (no environmental inputs) | Recursive (state-space) | Statistical reference. A direct SARIMAX with lagged environmental inputs is optional. |
| XGBoost | Direct, one model per h | Shallow trees, early stopping on the last 24 months of the training window. |
| LSTM | Direct, shared multi-output model or one per h (document which) | Early stopping on a within-window validation split; 5 seeds, averaged. |
| LightGBM | — | Only if XGBoost is clearly competitive. Otherwise redundant. |

### D5. Metrics and statistics

- RMSE and MAE on the rate scale. The h-step-ahead error is primary; the path average is secondary.
- Report **pooled over the 96 targets** for each county × horizon × model × configuration. Keep the mean of per-origin errors separate and labelled.
- **Skill vs seasonal naive:** 1 − RMSE_model / RMSE_snaive. This is scale-free, so counties can be aggregated without Kern dominating. Also report MASE.
- **Paired tests on identical targets:** Diebold–Mariano with HAC variance (Newey–West, lag h−1), using the Harvey–Leybourne–Newbold small-sample correction, per county × horizon. Add moving-block bootstrap 95% CIs (block length about 12) for the RMSE difference.
- Across counties: a count with a sign test of counties where R+E beats R, and the median skill difference with a bootstrap CI. Apply Holm correction where multiple tests are run.
- Stability: skill by sub-period and by calendar season; the share of total squared-error gain contributed by the top 5% of months (addresses Q9).

### D6. Preprocessing inside each window

- log1p is stateless, so it is fine anywhere.
- Imputation: forward-fill within the window, or a window-fitted ETS for pesticides. Never interpolate across the origin.
- Scaling: fit on the training rows of the window only.
- Feature selection and hyperparameters: development period only.
- Back-transform: smearing correction, or a raw-scale loss. Document which.

### D7. Output schema

`results/env/<experiment_id>/predictions.csv`, one row per forecast:

```
experiment_id, county, model, config (R|E|R+E|...), horizon, forecast_origin, target_month,
period (development|confirmation), seed, actual_rate, predicted_rate, train_start, train_end,
n_train, case_lag, env_lags_json
```

Plus:

- `metrics.csv` (county × model × config × horizon × period, pooled and per-origin aggregates)
- `config.yaml` (full settings and seeds)
- `provenance.json` (input file SHA-256, git commit, package versions)

Teammates can read these files without any change to their code.

### D8. Pipeline and tests

Suggested modules, all new files owned by you:

- `src/env/`: `config.py`, `data.py`, `features.py`, `splits.py`, `models.py`, `evaluate.py`, `run.py`
- CLI: `python -m src.env.run --config configs/env_baseline.yaml --counties Fresno Kern`
- Notebooks only load `results/env/...` for figures.

`tests/test_env_alignment.py` uses a synthetic series where y_t = month index:

- Assert that every feature date ≤ origin − L_f.
- Assert that target_month = origin + h.
- Assert that the training windows contain no target after the origin.
- Assert that the scaler and imputer statistics change when only post-origin data is perturbed. They must **not** change.

---

## E. Research roadmap

| # | Task | Priority | Effort | Depends on | Essential? |
|---|---|---|---|---|---|
| 1 | Verify data availability lags (PUR, CAL FIRE, AQI, CDPH case timing/revisions); resolve the Merced NaN = 0 question; record ValleyCast version | P1 | S (reading and emails) | — | Essential |
| 2 | Build `src/env` rolling-origin harness + alignment tests + output schema | P1 | M (2–4 days) | 1 (lags can start as config defaults) | Essential |
| 3 | Baselines (persistence, seasonal naive, climatology) + AR/Ridge in R/E/R+E across the 8 counties, development period | P2 | S | 2 | Essential |
| 4 | Add SARIMA, XGBoost, LSTM (5 seeds) to the harness | P2 | M | 3 | Essential |
| 5 | Freeze configurations in development, then run confirmation once; DM tests and bootstrap CIs | P2 | S | 4 | Essential |
| 6 | Expand to eligible counties (B15 table); apply the low-incidence handling | P2 | S (compute) | 5 | Essential |
| 7 | Grouped permutation importance and family-ablation (drop weather / AQI / fire / chemicals) across origins and seeds | P3 | M | 5 | Essential (Q4, Q5) |
| 8 | Sensitivity: expanding window, case lag 0/1/3, pesticide L=1 retrospective config, imputation variant | P3 | S each | 5 | Essential (Q8), keep small |
| 9 | Error analysis: high-incidence months, inland vs coastal and high- vs low-incidence groups | P3 | S | 6 | Essential |
| 10 | Targeted tuning of the one or two best models in development (LSTM early stopping, Huber loss, gradient clipping) | P4 | M | 5 | Optional |
| 11 | Pooled/global model across counties (shared tree model with county features) | P4 | M | 6 | Optional, scientifically interesting (Section F) |
| 12 | Figures, tables, methods/results/limitations text | P5 | M | 5–9 | Essential |

Deprioritised: more LSTM grid search, LightGBM in addition to XGBoost, Yeo-Johnson feature transforms (tree models don't need them; Ridge can use log1p on skewed families).

---

## F. Potential findings worth testing

**F1. Environmental features add little skill beyond incidence history at 1 month, but more at 3–6 months.**
- Rationale [H]: at short horizons, autocorrelation dominates. At longer horizons, the antecedent precipitation → growth → dry/wind dispersal pathway (the "grow and blow" hypothesis) may add information.
- Evidence needed: skill(R+E) − skill(R) per county × horizon in confirmation, with DM/bootstrap CIs. A horizon-interaction pattern (gain increasing with h) across most endemic counties.
- Rejected if: gains are null or negative at all horizons, or if they appear only in development.

**F2. The predictive value of the environment depends on geography: it is concentrated in the endemic San Joaquin Valley and Central Coast counties (Kern, Tulare, Fresno, SLO) and absent in coastal/low-incidence counties.**
- Evidence needed: skill gain by county group, plus family-ablation by group.
- Caveat: low-incidence counties have a high noise floor, so a null result there may reflect statistical power rather than biology. Report the minimum detectable effect.

**F3. Antecedent precipitation (wet season 1–2 years earlier) is the most robust environmental predictor, while fire, fungicide and rodenticide effects are unstable.**
- Evidence needed: grouped PFI and ablation repeated across origins, seeds and sub-periods. Report rank stability (e.g., the fraction of origin blocks in which a family ranks in the top 2).
- Pesticides will be tested at realistic lag ≥ 24, so a null result for them is plausible and worth reporting honestly.

**F4. Complex models do not consistently beat Ridge/AR.**
- Evidence needed: paired DM tests of XGBoost/LSTM vs Ridge on identical targets, with seed variance for the LSTM.
- With about 120 training rows per window, I expect regularised linear models to be competitive [H].
- This is a publishable negative result if shown rigorously.

**F5 (optional). A model pooled across counties beats per-county models for low-incidence counties** by borrowing strength. Requires roadmap task 11.

---

## G. Immediate next steps

1. **Build the rolling-origin harness with the synthetic alignment tests first** (roadmap 2). Every later result depends on it. Start with persistence, seasonal naive, climatology and Ridge in R/E/R+E: cheap models that are enough to show the pipeline works.
2. **Pin down the information-availability assumptions** (roadmap 1). Above all: PUR release lag, CDPH case dating and revisions, and the Merced missing-rate question. Put the answers in `configs/env_baseline.yaml` and `data/manifests/`.
3. **Run the development-period baseline comparison on the 8 counties and look at it before adding XGBoost or LSTM.** If R+E does not beat R with Ridge, that changes the paper's emphasis (toward F2/F4) before compute is spent on tuning neural networks.
