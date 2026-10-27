# Paper Model Set — Development-Period Comparison (2026-10-26)

**Setup**
- Config: `configs/env_main.json`, run with `--development-only`.
- Targets 2017-01 to 2020-12 (48 per county), 8 counties, case lag 1, no pesticides.
- XGBoost and LSTM: predictions averaged over seeds 42–44. LR, SARIMA and SARIMAX are deterministic.
- Reproduce with `scripts/env_model_comparison.py`, which writes `results/env/main_comparison/`.

Model roles:

| Role | Models |
|---|---|
| Baselines | LR (ridge, penalty chosen by in-window time-series CV; LR-R = AR(12)), SARIMA (R) and SARIMAX (RE, regressors lagged h + publication delay) |
| Test models | XGBoost and LSTM, each with R / E / RE |
| References | Seasonal naive (skill = 0), climatology |

**These are development numbers. Nothing here is a confirmed finding,** and with 8 counties × 3 horizons and many comparisons, isolated p < 0.05 results are expected by chance.

## 1. Median skill vs seasonal naive

Format: median skill (number of counties out of 8 with positive skill).

| Model | h=1 | h=3 | h=6 |
|---|---|---|---|
| **XGBoost RE** | **0.24 (8)** | **0.10 (6)** | **0.15 (7)** |
| XGBoost R | 0.18 (8) | 0.03 (5) | −0.04 (3) |
| XGBoost E | 0.03 (5) | −0.07 (4) | −0.03 (4) |
| SARIMAX (RE) | 0.19 (8) | 0.06 (6) | −0.06 (2) |
| SARIMA (R) | 0.20 (8) | 0.01 (5) | −0.09 (2) |
| LR-R (AR(12)) | 0.14 (8) | 0.01 (5) | −0.06 (2) |
| LR-RE | 0.14 (7) | 0.01 (5) | −0.07 (3) |
| LR-E | −0.06 (3) | −0.23 (2) | −0.39 (0) |
| LSTM RE | 0.04 (5) | −0.08 (2) | −0.08 (2) |
| LSTM R | 0.06 (5) | −0.09 (3) | −0.10 (2) |
| LSTM E | 0.01 (5) | −0.07 (2) | −0.07 (2) |
| Climatology | −0.13 (2) | −0.13 (2) | −0.13 (2) |

## 2. Contrasts

Format: counties where the first model or input set is better (number significant, DM p < 0.05), then the median RMSE change; negative means the first is better.

| Contrast | h=1 | h=3 | h=6 |
|---|---|---|---|
| XGB: RE vs E | 8 (7) −22% | 7 (4) −13% | 6 (2) −13% |
| LR: RE vs E | 7 (6) −19% | 7 (3) −20% | 8 (2) −18% |
| LSTM: RE vs E | 4 (2) +0% | 2 (0) +2% | 3 (0) +3% |
| XGB: RE vs R | 6 (2) −10% | 7 (1) −11% | 7 (1) −13% |
| LR: RE vs R | 4 (0) −0% | 4 (0) −0% | 5 (0) −2% |
| SARIMAX vs SARIMA | 4 (1) −0% | 6 (1) −3% | 4 (0) +0% |
| XGB-RE vs AR (LR-R) | 7 (1) −12% | 5 (1) −14% | 7 (1) −16% |
| XGB-RE vs SARIMAX | 5 (0) −6% | 4 (1) −1% | 7 (2) −14% |
| XGB-RE vs LR-RE | 7 (1) −13% | 6 (1) −10% | 8 (1) −9% |
| XGB-E vs LR-E | 6 (2) −5% | 5 (0) −6% | 7 (1) −11% |
| LSTM-RE vs LR-RE | 1 (0) +21% | 3 (0) +14% | 3 (0) +12% |
| LSTM-R vs LR-R | 2 (0) +20% | 1 (0) +10% | 3 (0) +8% |

For XGBoost, all 3 seeds agree on the direction in nearly every county. For the LSTM they often do not: for example, in "LSTM: RE vs E" only 0–1 counties are better with all seeds agreeing.

## 3. What this suggests (hypotheses for confirmation)

1. **Case history adds value beyond environment (RE vs E)** for LR and XGBoost at every horizon, most clearly at 1 month.
2. **Environment adds value beyond case history only in XGBoost** (about −10% to −13%; 6–7 of 8 counties, rarely significant individually). For linear models it adds nothing: LR RE vs R ≈ 0, SARIMAX ≈ SARIMA. If confirmed, the environment signal appears to be **nonlinear or interaction-based**.
3. **XGBoost RE is the only configuration with positive median skill at all horizons.** At 3–6 months, every linear baseline is at or below seasonal naive.
4. **The LSTM is currently the weakest model and is unstable across seeds.** This reflects an untuned initial configuration (hidden size 8, about 120 training sequences per fit). It is not yet a fair verdict on LSTMs.

## 4. Before the confirmation run

1. ~~Tune the LSTM on the development period.~~ Done (section 5).
2. ~~Select the SARIMA/SARIMAX orders on the development period.~~ Done (section 5).
3. **Freeze every configuration (`configs/env_main_tuned.json`), then run the confirmation period (2021–2024) once.**

## 5. Development tuning (2026-10-27)

**SARIMA/SARIMAX orders** are chosen per county by the lowest mean AIC over 36 orders (p, q ∈ {0,1,2}; P, Q ∈ {0,1}; d = D = 0). The AIC is computed on the four training windows ending 2016-12, 2017-12, 2018-12 and 2019-12. AIC uses training data only, so this choice never looks at evaluation targets. SARIMAX reuses SARIMA's order. The chosen orders are in `results/env/tuning/sarima_orders.json`.

**LSTM grid** (`scripts/env_tune_dev.py lstm-grid`, `results/env/tuning/lstm_grid.csv`):

- **How it was scored.** Development targets thinned to every 3rd month, seed 42. Score = mean log(RMSE_LSTM / RMSE_seasonal naive) over counties × horizons × feature sets.
- **Search.** Stage 1 tried lookback {6, 12, 24} × hidden {8, 32} × dropout {0.1, 0.3} at lr 1e-3. Stage 2 reran the top 3 at lr 3e-3.
- **Results.**

  | Setting | Score |
  |---|---|
  | Original (12/8/0.2/1e-3) | about +0.14 |
  | Best at lr 1e-3 (24/32) | +0.03 |
  | **Chosen: lookback 24, hidden 32, dropout 0.3, lr 3e-3** | **−0.016** |

- **Dropout 0.1 vs 0.3** differed by 0.002, below seed noise. The small network was underfitting.
- **Caveat.** The grid's targets are a subset of the development targets, so the tuned LSTM's development skill below is mildly optimistic. The confirmation period is the unbiased test.

**Tuned comparison.**

- **Setup.** LR, XGBoost and climatology are unchanged from `env_main_dev`. SARIMA, SARIMAX and the LSTM come from `env_main_tuned_dev`. XGBoost and LSTM are averaged over seeds 42–44.
- **Command.** `python scripts/env_model_comparison.py . env_main_dev env_main_tuned_dev`, with output in `results/env/main_comparison_env_main_dev+env_main_tuned_dev/`.
- **Median skill vs seasonal naive** (number of counties out of 8 with positive skill in parentheses):

| Model | h = 1 | h = 3 | h = 6 |
|---|---|---|---|
| XGBoost RE | 0.24 (8) | 0.10 (6) | 0.15 (7) |
| LSTM RE (tuned) | 0.21 (7) | −0.02 (4) | −0.01 (3) |
| SARIMAX (tuned) | 0.21 (8) | 0.06 (6) | −0.03 (3) |
| LSTM R (tuned) | 0.20 (8) | −0.02 (3) | −0.04 (4) |
| SARIMA (tuned) | 0.19 (8) | 0.02 (4) | −0.09 (2) |
| LR R (AR(12)) | 0.14 (8) | 0.01 (5) | −0.06 (2) |
| LSTM E (tuned) | 0.14 (5) | −0.02 (4) | −0.06 (2) |
| XGBoost E | 0.03 (5) | −0.07 (4) | −0.03 (4) |

**Contrasts** (median relative RMSE change; counts out of 8 counties):

- **SARIMAX vs SARIMA** (tuned orders): −2% / −4% / −4%. SARIMAX is better in 5–7 counties, but significantly so in only 1. Lagged environmental features now add a little to the linear state-space model.
- **LSTM RE vs E:** −9% / −3% / −4%, better in 7 counties at every horizon, significant in 1–3. This is the same direction as LR and XGBoost.
- **LSTM RE vs R:** +3% / +3% / +1%. Environment does not help the LSTM.
- **LSTM E vs LR E:** −13% / −13% / −16%, better in 6 counties. Among the environment-only models, the LSTM now beats LR (as does XGBoost, −5% / −6% / −11%).
- **XGBoost RE vs LSTM RE:** XGBoost is still clearly better at 3 and 6 months. LSTM RE vs SARIMAX is +4% / +10% / −5%, with no county significantly better.
- **Seed instability.** Per-seed LSTM RE skill at h = 3 ranges from −0.12 to −0.02. The 3-seed average is noticeably better than single seeds; a larger seed ensemble would likely help.

**Updated reading of section 3.**

- Points 1–3 stand.
- Point 4 changes: tuning makes the LSTM competitive with SARIMA/SARIMAX at 1 month and the best environment-only model after XGBoost. It still does not extract value from environment on top of case history. **XGBoost RE remains the only model with positive median skill at every horizon.**
