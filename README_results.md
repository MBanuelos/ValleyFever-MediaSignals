# Media + environment walk-forward forecasts

## Run
```bash
pip install xgboost statsmodels scikit-learn pandas numpy
python src/media_env_walk_forward.py run --data-dir data --out-dir results/media_env_walk_forward        # everything (about 2 hours on one core)
python src/media_env_walk_forward.py combine --data-dir data --out-dir results/media_env_walk_forward    # merge parts into final tables
python build_dashboard.py data results/media_env_walk_forward dashboard.html
```
`run` accepts `--scope top|remaining|all`, `--feature-sets env_only media_only media_env`, `--models ...` and `--tag`; finished county runs are skipped on re-run.

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
  - The LSTM is written in NumPy (no PyTorch needed): one hidden layer of 24 units, 6-month look-back, predicts all six steps at once. Gradients were checked numerically. Hyperparameters are fixed, not tuned. Each LSTM is a single run (fixed seed), so its scores vary with the seed.
- Best model = lowest RMSE averaged over the 1, 3 and 6 month horizons.
