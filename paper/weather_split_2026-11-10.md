# Weather Group Split: Ablation and Permutation Importance (2026-11-10)

Follow-up to `paper/importance_and_case_delay_2026-10-27.md`. There, the LSTM relied on the weather group under permutation importance, but removing weather did not hurt it (and at 6 months in the test period improved LSTM RE by 6%). This note splits weather to see which part is responsible.

Everything here is about the frozen configuration `configs/env_main_tuned.json`. It is descriptive and was not used to change models or features.

## Subgroups

The split is based on correlations among the weather variables over 2001–2020 (no test-period data):

| Subgroup | Variables | Why |
|---|---|---|
| humidity | `Avg Rel Hum (%)` | separate process |
| wind | `Avg Wind Speed (mph)`, `Wind Run (miles)` | near-duplicates: r = 0.81–0.98 in every county |
| wind events | `windeventcount` | weakly related to average wind in 5 of 8 counties (r −0.06 to 0.47); zero in 81–95% of months in Orange, San Diego, San Luis Obispo and Tulare |
| soil temperature | `Avg Soil Temp (F)` | separate process; strongly negatively correlated with humidity (r ≈ −0.9) in Fresno, Kern and Tulare, so in those counties each can partly stand in for the other |

Grouping only defines what is removed or permuted together; the models still see every variable's lags as separate inputs.

## Runs

- Ablation: `configs/ablation/no_{humidity,wind,wind_events,soil_temp}.json`, XGBoost and LSTM, E and RE, seeds 42–44, targets 2017–2024. Remaining weather variables keep their original order.
- Permutation importance: `configs/importance/pfi_weather_split.json` (built on the helper `_weather_split_base.json`), seeds 42–44, 20 draws. Its inputs and column order are identical to `env_main_tuned_pfi`, and its forecasts match that run exactly (max difference 0.0 for every model and seed), so only the importance groups differ.
- Analysis: `python scripts/env_weather_split_analysis.py .` → `results/env/importance/weather_split/` (`ablation.csv`, `pfi_pooled.csv`, `pfi_by_county.csv`, `fig_weather_split_ablation.png`, `fig_weather_split_pfi.png`).
- Same pooled statistic and 95% block bootstrap (block 6, 2000 draws) as the earlier note. No multiple-testing adjustment.
- Compute: 15 runs, 4 at a time with 4 counties each, about 2 h 40 min on the laptop.

## 1. Permutation importance: what the LSTM uses

Test period, pooled RMSE change when the subgroup is permuted (h = 1 / 3 / 6):

| Model | Humidity | Wind | Wind events | Soil temp. |
|---|---|---|---|---|
| LSTM E | +5 / +5 / +6% | **+7 / +7 / +5%** | +2 / +1 / +1% | +1 / +1 / +1% |
| LSTM RE | +1 / +1 / +4% | +5 / +5 / −1% | +1 / +1 / +2% | +1 / +1 / +1% |
| XGBoost E | +1 / +2 / 0% | +3 / 0 / +3% | 0 / −1 / 0% | −1 / +5 / +2% |
| XGBoost RE | 0 / 0 / 0% | 0 / −1 / 0% | 0 / 0 / 0% | 0 / +4 / 0% |

- **The LSTM's reliance on weather comes from wind (speed + run) and humidity.** Both are positive with intervals excluding zero for LSTM E at most horizons in both periods (validation: wind +9 / +6 / +4%, humidity +5 / +3 / +5%).
- **Wind events and soil temperature are barely used** by any model (mostly within ±2%).
- **XGBoost hardly uses any weather subgroup**, consistent with the earlier finding that it relies on case history and precipitation.
- **By county (LSTM E):** wind reliance is concentrated in Orange, San Diego and Tulare in the test period (+11 to +13%) and in Los Angeles in validation (+32%). Humidity is strongest in Los Angeles (+17% test, +26% validation). Kern and Fresno use little of either.

## 2. Ablation: what is lost without each subgroup

- **No subgroup is clearly needed.** Removing any single subgroup changes pooled error by less than ±5% in almost every cell, and nearly all intervals include zero. Most intervals that exclude zero are +1 to +4%, and with 48 intervals per period, a few are expected by chance.
- **The one larger effect:** removing wind events *improves* LSTM E by 9% at 6 months in the test period ([−12, −4]). It is +1% in validation, so it does not replicate across periods and should not be over-read.
- **The earlier whole-weather result (−6% for LSTM RE at h = 6, test) is not explained by any single subgroup.** Individually, humidity is +5% and the others are near zero. The gain from dropping all five appears only when they are removed together, which suggests the LSTM is somewhat overfit to the weather inputs as a block, not harmed by one specific variable.

## 3. Interpretation

- **Used but replaceable.** The LSTM leans on wind and humidity when it has them (permutation importance), but a retrained LSTM without them does about as well (ablation). Other inputs carry overlapping information. Air quality is a plausible substitute, since PM10 reflects wind-blown dust, but this was not tested.
- **Wind events add nothing reliable.** They are mostly zero in half the counties, models barely use them, and their only sizeable ablation effect does not replicate. Checking how `windeventcount` is constructed in the data processing is worthwhile before using it in any future model.
- **No change to the main conclusions.** Precipitation remains the only environmental group clearly important under both methods, and the weather group's contribution remains small. Do not drop or reselect weather variables based on these results, since that would be choosing features using the test period.

## Caveats

- Correlated subgroups mask each other in ablation, particularly humidity and soil temperature in the inland counties (r ≈ −0.9).
- Permutation importance can overstate correlated inputs because scrambling one creates unrealistic combinations with the others.
- 8 counties and 48 target months per county per horizon; about 2–3 of the 48 ablation intervals per period are expected to exclude zero by chance.
