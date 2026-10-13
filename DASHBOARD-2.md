# Valley Fever forecast dashboard

An interactive page showing walk-forward forecasts for the 10 California counties with the most Valley Fever cases (among the 21 counties in this project's data).

## Open it

- **Online:** https://mbanuelos.github.io/ValleyFever-MediaSignals/ (works once GitHub Pages is turned on for this repo; see the setup note at the bottom).
- **On your computer:** download or clone the repo and double-click `dashboard.html`. It is one self-contained file and needs no install and no internet connection. (Viewing the file on github.com shows its source code, not the page.)

## How to use it

**Main dashboard (top of the page)**

- The table ranks the 10 counties by total cases. For each feature set (Environment, Media, Media + environment) it shows the best model and its average RMSE, the forecast error. Lower is better.
- A dot marks each county's lowest error across the three feature sets. The sentence above the table counts how many counties each feature set wins.
- Click a row, or use the **County** dropdown under the table, to choose a county.

**The three dropdowns: Environment, Media, Media + environment**

- Click a heading to open or close it. Each heading's summary line gives that county's best model and average RMSE for that feature set.
- **Horizon buttons** switch between forecasts made 1, 3 and 6 months ahead.
- **Model toggles** show or hide each model: Seasonal naive, Ridge, Random forest, XGBoost, SARIMAX, LSTM. The best model is tagged "best". At least one model always stays on.
- **Chart:** cases per 100,000 per month, 2021 to 2024. The black line is observed. Each colored line is that model's forecast, made the chosen number of months earlier. Hover over the chart, or click it and use the left and right arrow keys, to read exact values for a month.
- **What drives the forecasts:** the top features of the best model that reports importance, colored by news and Twitter, environment, or past case changes.
- **All models:** RMSE and MAE at 1, 3 and 6 months, plus improvement over the seasonal-naive baseline (which repeats last year's value).

## What the numbers mean

- RMSE and MAE are averaged over every forecast month in every rolling window, on the log-change scale the project uses (month-to-month change in the log of cases per 100,000). The summary text under each county also gives the error in cases per 100,000.
- Models are trained on a rolling 10-year window and forecast 2021-02 to 2024-12.
- Media data are California-wide, so every county sees the same media signal.
- Results are predictive and associational, not causal.

## Where the data comes from

Every chart is drawn from files in `results/media_env_walk_forward/`:

- `top10_forecasts_{env_only,media_only,media_env}.csv`: observed and predicted rates by county, model, horizon and month
- `top10_metrics_{env_only,media_only,media_env}.csv`: RMSE, MAE and improvement over baseline
- `top10_feature_importance_{env_only,media_only,media_env}.csv`: feature importance

## Rebuild it

```bash
pip install xgboost statsmodels
python build_dashboard.py data results/media_env_walk_forward dashboard.html
```

To rerun the models themselves, see `README_results.md`.

## Extending the dashboard

`dashboard.html` is generated, so never edit it by hand. The next rebuild overwrites it. Change the code that builds it, then rebuild.

How the pieces fit together:

- `src/media_env_walk_forward.py` runs the models and writes the results CSVs.
- `build_dashboard.py` reads those CSVs and injects the data into `dashboard_template.html`.
- `dashboard_template.html` holds all the layout and interactive code (plain HTML and JavaScript, no framework).
- The output is `dashboard.html`.

Where to make each kind of change:

| To add | Change |
|:--|:--|
| A new model | Add its name to `MODELS` and its fitting code to `run_county` in `src/media_env_walk_forward.py`. In the template, add it to `MODEL_ORDER`, `MODEL_NAME` and `MODEL_COLOR`. |
| A baseline (for example "no change from last month") | Same as a new model. It needs no fitting, so it runs quickly. |
| A new feature set | Add it to `FEATURE_SETS` and `_set_cols` in the module, and add an entry to `SETS` in the template. `build_dashboard.py` picks it up automatically. |
| New data columns | Add them to `ENV_COLS` or `MEDIA_COLS`, and add a readable name to `LABELS` in the template. |
| A new chart, table or control | Edit `dashboard_template.html`. The data is available as `DATA.counties[i].sets[key]`, with `metrics`, `forecasts`, `features` and `best`. |
| A new field in the data | Add it to the payload in `build_dashboard.py`, then use it in the template. |

After a change, rerun what it affects and rebuild. Use a new `--tag` for each run, since finished county runs are skipped:

```bash
python src/media_env_walk_forward.py run --data-dir data --out-dir results/media_env_walk_forward --tag mytag
python src/media_env_walk_forward.py combine --data-dir data --out-dir results/media_env_walk_forward
python build_dashboard.py data results/media_env_walk_forward dashboard.html
open dashboard.html
```

If you only change the look and not the results, you only need the last two commands.

Things to watch:

- The dashboard covers the top 10 counties only. The "top10" file names and the `top_counties` call (default 10) are written into the code, so changing that takes some edits.
- `results/` is ignored by git. Commit changed CSVs with `git add -f` and name each file instead of adding the whole folder.
- Make changes on a branch and open a pull request so teammates can review them.
- After rebuilding, open the page and click through a couple of counties and all three feature-set sections to check nothing broke.

## Setup note: turning on the online link

In the GitHub repo: **Settings**, then **Pages**, then under "Build and deployment" choose **Deploy from a branch**, branch `main`, folder `/ (root)`, and click **Save**. After a minute or two the page is live. To make the link open the dashboard directly, add a copy of `dashboard.html` named `index.html` to the repo's top folder.
