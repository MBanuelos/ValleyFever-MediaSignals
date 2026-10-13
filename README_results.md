# Media + environment walk-forward forecasts

See `DASHBOARD.md` for how to use and update the dashboard.

## Commands
```bash
pip install -r requirements.txt
python src/media_env_walk_forward.py run --tag first                  # fit everything (about 2 hours on one core)
python src/media_env_walk_forward.py combine                          # merge new runs into the saved results
python build_dashboard.py                                             # rebuild dashboard.html
```
`run` accepts `--scope top|remaining|all`, `--counties ...`, `--feature-sets ...`, `--models ...` and `--tag`. Use a new tag for each run: finished county runs with the same tag are skipped, so an interrupted run can be resumed. `combine` adds new results to the saved ones and never drops saved rows.

## Remaining-county function
```python
from src.media_env_walk_forward import best_model_for_remaining_counties, best_model_and_top_features
table, results = best_model_for_remaining_counties("data")
summary, res   = best_model_and_top_features("Monterey", "data")
```

## Setup
- Rolling 120-month training window, 1-month step, direct 1 to 6 month forecasts (2021-02 to 2024-12).
- Target: month-to-month change in log1p(rate per 100k). Features are lagged by the forecast step.
- Feature sets: environment, media, media + environment. All include the county's own recent case changes.
- Models: seasonal naive, Ridge, Random Forest, XGBoost, SARIMAX, LSTM.
  - Ridge, Random Forest and XGBoost are tuned once on the first window (TimeSeriesSplit).
  - SARIMAX uses the five outside predictors most correlated with cases on the first window (lagged 6 months, so future values are known) and an order picked by AIC on that window.
  - The LSTM is written in NumPy (no PyTorch needed): one hidden layer of 24 units, 6-month look-back, predicts all six steps at once. Gradients were checked numerically. Hyperparameters are fixed, not tuned, and each LSTM is a single seeded run.
- Best model = lowest RMSE averaged over the 1, 3 and 6 month horizons.
- The seasonal-naive baseline repeats last year's change. A simple "no change from last month" forecast is a stricter baseline, and at 1 month ahead it matches or beats the models in several counties.
