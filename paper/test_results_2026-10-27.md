# Test Period Results: Held-Out Targets 2021–2024 (2026-10-27)

- **Configuration.** `configs/env_main_tuned.json`, frozen before this run. The freeze record is `results/env/test_freeze.json`: commit, config hash, and SHA-256 of the configs and harness code.
- **Runs.**
  - One run with seed 42 for all models.
  - Seeds 43 and 44 for XGBoost and the LSTM only. The other models are deterministic.
- **What was scored.**
  - Targets 2017-01..2024-12 were forecast in one pass.
  - The validation period (2017–2020) reproduces the earlier validation-only runs exactly.
  - The test period was run once, and nothing was changed afterwards.
- **Command.**

  ```
  python scripts/env_model_comparison.py . env_main_tuned --period test
  ```

  Output: `results/env/main_comparison_env_main_tuned_test/`.

## 1. Median skill vs seasonal naive

Number of counties (out of 8) with positive skill in parentheses.

| Model | h=1 test | h=3 test | h=6 test | h=1 val. | h=3 val. | h=6 val. |
|---|---|---|---|---|---|---|
| **XGBoost RE** | **0.22 (8)** | **0.14 (7)** | **0.19 (8)** | 0.24 | 0.10 | 0.15 |
| SARIMAX | 0.21 (7) | 0.13 (7) | 0.01 (4) | 0.21 | 0.06 | −0.03 |
| SARIMA | 0.23 (8) | 0.06 (5) | −0.06 (3) | 0.19 | 0.02 | −0.09 |
| LSTM RE | 0.19 (8) | 0.10 (6) | 0.02 (5) | 0.21 | −0.02 | −0.01 |
| LSTM R | 0.19 (8) | 0.10 (6) | 0.08 (6) | 0.20 | −0.02 | −0.04 |
| **LSTM E** | 0.17 (7) | **0.15 (6)** | 0.12 (5) | 0.14 | −0.02 | −0.06 |
| LR R = AR(12) | 0.19 (8) | 0.10 (5) | 0.07 (7) | 0.14 | 0.01 | −0.06 |
| LR RE | 0.08 (6) | 0.04 (5) | 0.01 (4) | 0.14 | 0.01 | −0.07 |
| XGBoost E | 0.06 (5) | 0.03 (5) | 0.06 (5) | 0.03 | −0.07 | −0.03 |
| LR E | −0.15 (2) | −0.11 (3) | −0.20 (3) | −0.06 | −0.23 | −0.39 |
| Climatology | −0.14 (2) | −0.14 (2) | −0.14 (2) | −0.13 | −0.13 | −0.13 |

## 2. Contrasts

Median relative RMSE change across counties (negative means the first option is better). The fractions are the number of counties where the first option is better (DM p < 0.05 count in parentheses).

| Contrast | h=1 | h=3 | h=6 | Validation (h=1/3/6) |
|---|---|---|---|---|
| **XGB: RE vs E** | −13%, 8/8 (4) | −10%, 7/8 (3) | −11%, 6/8 (4) | −22 / −13 / −13% |
| LR: RE vs E | −23%, 8/8 (3) | −9%, 7/8 (2) | −1%, 5/8 (1) | −19 / −20 / −18% |
| LSTM: RE vs E | −7%, 6/8 (4) | −5%, 5/8 (3) | −11%, 5/8 (0) | −9 / −3 / −4% |
| **XGB: RE vs R** | −2%, 5/8 (1) | −10%, 6/8 (0) | −16%, 6/8 (2) | −10 / −11 / −13% |
| LR: RE vs R | +15%, 3/8 | +2%, 3/8 | +8%, 4/8 | 0 / 0 / −2% |
| LSTM: RE vs R | +3%, 3/8 | −1%, 4/8 | −9%, 5/8 | +3 / +3 / +1% |
| SARIMAX vs SARIMA | +1%, 3/8 | −2%, 4/8 | −1%, 5/8 | −2 / −4 / −4% |
| XGB-RE vs AR(12) | +4%, 4/8 | 0%, 4/8 | −5%, 6/8 | −12 / −14 / −16% |
| XGB-RE vs SARIMAX | −3%, 5/8 | +6%, 3/8 | −19%, 7/8 (1) | −4 / −4 / −16% |
| LSTM-E vs LR-E | −25%, 7/8 | −23%, 8/8 | −19%, 6/8 | −13 / −13 / −16% |

## 3. What held up and what did not

1. **Held up: case history improves on environment alone (RE vs E).**
   - This is true for every model family at every horizon.
   - For XGBoost it holds in 6–8 counties, with 3–4 significant per horizon.
   - This is the key contrast, and it replicates.
2. **Held up: XGBoost RE is the only model with positive median skill at every horizon in both periods.**
   - Test skill: 0.22 / 0.14 / 0.19, positive in 7–8 of 8 counties.
   - Its advantage is largest at 6 months. There it beats SARIMAX by 19% (7 of 8 counties) and AR(12) by 5%.
3. **Held up, mainly at longer horizons: environment adds value beyond case history only in XGBoost.**
   - RE vs R is −2% at 1 month, −10% at 3 months and −16% at 6 months.
   - For the linear models it adds nothing or hurts: LR RE vs R is +15% at h = 1, and SARIMAX ≈ SARIMA.
   - The nonlinear/interaction interpretation stands.
4. **Did not hold up: XGBoost's lead over simple AR at short horizons.**
   - AR(12) was much stronger in 2021–2024 (0.19 / 0.10 / 0.07) than in 2017–2020.
   - At h = 1–3, XGBoost RE ≈ AR(12) (+4% and 0%). XGBoost's advantage remains only at h = 6.
5. **New: the tuned LSTM is the strongest environment-only model in test.**
   - LSTM E skill is 0.17 / 0.15 / 0.12 vs seasonal naive. It is 19–25% better than LR E and well above XGBoost E.
   - In validation it was near zero. This may reflect the 2021–2024 dynamics (see caveats), so it should not be over-read.
   - It is relevant to the goal of forecasting without case data.

## 4. Caveats

- **The 2021–2024 period is unusual.** It includes COVID-era changes to testing and reporting, and high-incidence years. Rank changes between periods (AR stronger; LSTM-E stronger) may reflect the period, not the models.
- **Small samples.** There are 8 counties and 48 target months per county per horizon. Few county-level DM tests are significant, so the claims rest on the consistency of direction across counties and periods.
- **The LSTM's validation skill was mildly optimistic** because tuning used validation targets. Its test numbers are unbiased.
- **No multiple-comparison correction** has been applied to the county-level DM counts.
