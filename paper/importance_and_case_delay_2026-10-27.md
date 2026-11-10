# Pooled Significance, Group Ablation, Permutation Importance and Case-Delay Sensitivity (2026-10-27)

All of these analyses are about the frozen configuration `configs/env_main_tuned.json`. Nothing in this note was used to change the models, features or settings; the test-period results in `paper/test_results_2026-10-27.md` stand as they are.

- **Pooled change.** Geometric mean across the 8 counties of the RMSE ratio, minus 1. For a contrast "A vs B", negative means A has lower error.
- **Uncertainty.** 95% circular moving-block bootstrap, block 6 months, 4000 draws for the contrasts and 2000 for the ablations. The same resampled months are used in every county, so correlation between counties is kept.
- **Multiple testing.** Holm adjustment over the 36 contrast × horizon tests in each period. The ablation intervals are not adjusted (see §2).
- **Seeds.** XGBoost and LSTM errors are averaged over seeds 42–44.

Reproduce:

```
python scripts/env_pooled_significance.py .            # §1
python scripts/env_importance_analysis.py .            # §2–§4, figures 4–6
```

Outputs: `results/env/pooled_significance/`, `results/env/importance/`.

## 1. Pooled significance of the main contrasts

| Contrast | Test h=1 | Test h=3 | Test h=6 | Validation h=1 / 3 / 6 |
|---|---|---|---|---|
| **XGBoost: RE vs E** | **−20% [−30, −11]\*** | **−18% [−24, −12]\*** | **−19% [−27, −13]\*** | −26\* / −16\* / −15%\* |
| LR: RE vs E | −33% [−41, −15]\* | −10% [−29, +1] | −25% [−30, −12]\* | −21\* / −19\* / −19%\* |
| LSTM: RE vs E | −7% [−10, −2] | −4% [−8, +5] | −8% [−14, −2] | −9 / −4 / −6% |
| XGBoost: RE vs R | −4% [−10, +6] | −9% [−14, +3] | −17% [−25, −5] | −6 / −6 / −13%\* |
| XGBoost RE vs AR(12) | −2% [−6, +13] | −1% [−11, +22] | −6% [−16, +11] | −10 / −8 / −14% |
| XGBoost RE vs SARIMAX | −2% [−9, +10] | 0% [−7, +12] | −19% [−27, −1] | −4 / −3 / −14% |
| SARIMAX vs SARIMA | +3% [−4, +13] | −7% [−13, +8] | −2% [−7, +5] | −2 / −6 / −4% |
| LSTM E vs LR E | −44% [−60, −11]\* | −23% [−37, −13]\* | −51% [−60, −5]\* | −15\* / −18 / −24% |

\* Holm-adjusted p < 0.05. Brackets: 95% bootstrap interval. The block-12 run (`..._block12.csv`) gives the same conclusions.

- **The key contrast holds with proper uncertainty.** XGBoost RE beats XGBoost E by about 20% at every horizon in both periods, and survives the multiple-testing correction everywhere.
- **Environment on top of case history (RE vs R) for XGBoost** has an interval excluding zero only at 6 months (both periods), and does not survive Holm in the test period. It is a long-horizon effect, not a general one.
- **XGBoost RE vs the simple baselines** (AR(12), SARIMAX) is not significant at any horizon in the test period. At h=6 vs SARIMAX the interval excludes zero but Holm does not.
- **Environment-only deep learning beats environment-only linear regression** (LSTM E vs LR E) significantly at all test horizons: nonlinearity matters when only environmental features are used.

## 2. Group ablation

Each run removes one group of environmental variables from `env_main_tuned`, retrains XGBoost and the LSTM (E and RE, seeds 42–44, targets 2017–2024) and compares with the full model. Positive = forecasts get worse without the group, i.e. the group helped. Configs: `configs/ablation/no_<group>.json`. Figure: `results/env/importance/fig4_group_ablation.png`.

| Group | Variables |
|---|---|
| Precipitation | Precipitation (lags 0–2, 6- and 12-month means) |
| Weather | Relative humidity, average wind speed, wind run, wind event count, soil temperature (lags 0–2) |
| Air quality | AQI PM2.5, AQI PM10 (lags 0–2) |
| Wildfire | Acres burned (lags 0–2) |

Test period (2021–2024), RMSE change when the group is removed:

| Model | Group | h=1 | h=3 | h=6 |
|---|---|---|---|---|
| XGBoost RE | precipitation | +0% [−9, +6] | +4% [−5, +13] | **+12% [+4, +24]** |
| XGBoost RE | air quality | +1% [−1, +2] | +1% [−2, +3] | +2% [+1, +3] |
| XGBoost RE | weather | −1% [−3, +1] | +1% [−1, +3] | +2% [−1, +6] |
| XGBoost RE | wildfire | +3% [−2, +5] | +2% [0, +4] | −1% [−3, +1] |
| XGBoost E | air quality | +5% [+3, +8] | +2% [−2, +6] | +4% [+1, +6] |
| LSTM RE | precipitation | +5% [0, +7] | +5% [−2, +10] | +8% [0, +17] |
| LSTM RE | weather | −2% [−8, +2] | −3% [−8, +2] | −6% [−9, −1] |
| LSTM E | precipitation | +5% [+1, +12] | +7% [+4, +12] | +4% [−1, +8] |

Full table for both periods: `results/env/importance/ablation.csv`.

- **Precipitation is the group that matters most in the test period.** Removing it makes XGBoost RE 12% worse at 6 months, and it helps the LSTM at most horizons. In the validation period the same effects point the same way for XGBoost but are smaller, and their intervals include zero, so the size of this effect is not stable between periods.
- **Air quality gives a small, consistent gain to XGBoost** (about +1–6%, several intervals excluding zero in both periods), mostly for the environment-only model.
- **Weather and wildfire show no reliable contribution.** For the LSTM RE at 6 months in the test period, removing weather *improves* forecasts by 6%, which suggests the LSTM overfits those five variables.
- **No single group explains the RE vs E gap.** Removing any one group changes error by at most ~12%, while removing case history changes it by ~20%.

Caveats:
- 32 intervals per period are not adjusted for multiple testing, so one or two intervals excluding zero are expected by chance. Only precipitation for XGBoost RE at h=6 (+12%) is large enough to be clearly beyond that.
- Groups share information (e.g. humidity and soil temperature with precipitation), so removing one group can be partly made up by the others. Ablation measures what is unique to a group, not everything it carries.

## 3. Permutation importance

How the method works:
- At every forecast origin, after the model is trained, the inputs of one group (or one variable) are replaced with values from 20 random training months and the model predicts again. There is no retraining.
- For the LSTM, the whole 24-month input sequence of those columns is replaced.
- Substitute values come only from the training window, so no unpublished data is used. Forecasts are identical with and without this step; both properties are tested in `tests/test_env_importance.py`.
- Positive = forecasts get worse when the group is scrambled, i.e. the trained model relies on it.

Run: `configs/importance/env_main_tuned_pfi.json`, seeds 42–44. Figures: `fig5_permutation_importance_groups.png` (groups) and `fig6_variable_importance_by_county.png` (XGBoost RE, each variable by county). Tables: `pfi_pooled.csv`, `pfi_by_county.csv`.

Test period, RMSE change when the group is permuted:

| Model | Case history | Precipitation | Weather | Air quality | Wildfire |
|---|---|---|---|---|---|
| XGBoost RE, h=1 / 3 / 6 | **+58 / +37 / +41%** | +9 / +14 / **+25%** | +1 / +4 / +2% | +2 / +4 / +3% | +10 / +2 / 0% |
| XGBoost E | — | **+18 / +16 / +30%** | +5 / +8 / +6% | +7 / +8 / +9% | +14 / +3 / +2% |
| LSTM RE | **+27 / +19 / +14%** | +3 / +5 / +10% | +9 / +9 / +7% | +4 / +6 / +4% | −1 / −3 / −5% |
| LSTM E | — | +8 / +12 / +11% | **+20 / +18 / +14%** | **+20 / +21 / +14%** | +1 / 0 / −3% |
| LR E | — | +9 / +12 / +14% | +5 / +4 / +4% | +3 / +3 / +3% | −29 / −5 / −46% |

Validation (2017–2020) shows the same ranking. Full tables with intervals: `pfi_pooled.csv`.

Findings:
- **Case history is by far the most relied-on input** for both RE models (XGBoost +27 to +62%, LSTM +14 to +28%, both periods). This matches the RE vs E contrast.
- **Precipitation is the most relied-on environmental variable for XGBoost**, and its importance grows with horizon (test: +9% → +25% for RE). This agrees with the ablation, where precipitation was the only group with a clear unique effect at 6 months.
- **The LSTM relies on different inputs:** weather and air quality rather than precipitation, especially in the E-only model. The ablation showed that removing weather does not hurt (and at h=6 helps) the LSTM, so these variables are used but are replaceable by other inputs, or are overfit.
- **Wildfire has a negative score for linear regression** (scrambling it *reduces* error by up to 46%). Extreme fire years fall outside the training range and a linear model extrapolates them into bad forecasts; tree models do not extrapolate, so XGBoost is not affected. This partly explains why LR E is the worst model.
- **Wildfire matters to XGBoost at 1 month in the test period** (+10% RE, +14% E) but not in validation, and the ablation shows almost no unique effect. It is driven by a few counties (fig 6: Fresno, San Luis Obispo, Tulare), likely by the large 2020–2021 fire seasons.

### Importance differs by county (fig 6, XGBoost RE)

- **Precipitation matters in Fresno (+49% validation / +48% test), San Diego, San Luis Obispo and Tulare in both periods**, and in Ventura in the test period.
- **It matters almost nothing in Kern, Los Angeles and Orange.** Kern, the highest-incidence county, gets close to no value from any environmental variable; its forecasts come from case history.
- **The other variables are near zero almost everywhere** (mostly within ±3%). Soil temperature and PM2.5 reach 5–7% in only a few county-period cells, and not consistently across periods.
- This is direct evidence for the idea that useful environmental features vary by county. It is descriptive, not a tested county-specific model: choosing features per county from these results and scoring on the same years would be test-set selection.

### PFI vs ablation

The two methods answer different questions and should not be expected to agree exactly:
- **PFI:** how much the trained model *uses* an input.
- **Ablation:** how much accuracy is *lost* when a retrained model never sees it.

A variable can score high on PFI and zero on ablation when other variables carry the same information (LSTM weather). PFI can also overstate correlated variables, because scrambling one creates unrealistic combinations with the others. **Precipitation is the only environmental group that is clearly positive under both methods**, so it is the safest environmental result to report.

## 4. Case-delay sensitivity

The main runs assume case rates are known 1 month after the fact. These runs repeat the main models with a delay of 0 months (cases known immediately) and 3 months (a realistic reporting delay). The LSTM was not rerun. Table: `results/env/importance/case_lag.csv`.

Pooled skill vs seasonal naive, test period:

| Model | h=1 delay 0 / 1 / 3 | h=3 delay 0 / 1 / 3 | h=6 delay 0 / 1 / 3 |
|---|---|---|---|
| **XGBoost RE** | 0.36 / 0.22 / 0.13 | 0.14 / 0.12 / 0.12 | 0.15 / 0.17 / 0.14 |
| XGBoost R | 0.33 / 0.18 / 0.03 | 0.09 / 0.03 / −0.02 | −0.02 / −0.01 / −0.02 |
| XGBoost E | 0.12 / 0.03 / −0.06 | −0.06 / −0.07 / −0.09 | −0.02 / −0.03 / −0.04 |
| SARIMAX | 0.35 / 0.20 / 0.10 | 0.17 / 0.12 / 0.08 | 0.00 / −0.03 / −0.06 |
| SARIMA | 0.39 / 0.23 / 0.05 | 0.11 / 0.05 / −0.03 | −0.03 / −0.06 / −0.10 |
| LR R = AR(12) | 0.37 / 0.21 / 0.11 | 0.13 / 0.11 / 0.10 | 0.10 / 0.11 / 0.09 |

XGBoost RE vs E (pooled RMSE change), test: −28 / −20 / −18% at h=1, −18 / −18 / −19% at h=3, −17 / −19 / −17% at h=6 for delays 0 / 1 / 3. Validation is similar.

- **The key result does not depend on the delay assumption.** RE beats E by 17–28% for XGBoost at every delay, horizon and period.
- **Case-only models lose most of their skill as the delay grows**, especially at 1 month ahead. With a 3-month delay, SARIMA and XGBoost R have almost no skill left.
- **XGBoost RE keeps its skill at 3 and 6 months for every delay** and is the best model at h=6 for every delay. The more delayed the case data, the more the environmental features matter relative to case history.
- **E-only models also change with the delay** because their training labels are case rates, and with a longer delay the most recent months cannot be used for training.
- **Sanity check.** For case-only models only the total lead (horizon + delay) should matter, and it does: e.g. SARIMA has the same skill at delay 1, h=3 and delay 3, h=1 (0.051). Climatology is identical across delays and horizons by construction (it always averages the same 10 previous same-month values).
