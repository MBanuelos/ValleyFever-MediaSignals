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

## Updating the dashboard

`dashboard.html` is generated. Never edit it by hand, because the next build overwrites it. Change the inputs, then rebuild. Building only reads the saved results and takes a few seconds. Fitting models is the slow part, and it is only needed when the numbers themselves change.

```bash
pip install -r requirements.txt     # once
python build_dashboard.py           # rebuild dashboard.html from the saved results
open dashboard.html                 # check it
```

### What lives where

| File | What it controls |
|:--|:--|
| `dashboard_config.json` | Display names, colors, intro text, and which feature sets are shown and in what order |
| `dashboard_template.html` | Layout, charts, tables, controls and the footer notes |
| `build_dashboard.py` | Reads the results CSVs and fills the template |
| `src/media_env_walk_forward.py` | The models and the walk-forward run (what produces the numbers) |
| `results/media_env_walk_forward/*.csv` | The saved results the dashboard is built from |

The dashboard shows whatever is in the results folder. A model or feature set that is in the CSVs but not in `dashboard_config.json` still appears, with an automatic label and color.

### Common changes

| I want to... | Do this | Time |
|:--|:--|:--|
| Change wording, labels, colors or the intro text | Edit `dashboard_config.json`, then `python build_dashboard.py` | seconds |
| Change layout, add a chart or control, or edit the footer notes | Edit `dashboard_template.html`, then `python build_dashboard.py`. The data is available as `DATA.counties[i].sets[key]` with `metrics`, `forecasts`, `features` and `best`. | seconds |
| Add a model | See "Adding a model" below | seconds to about 20 minutes |
| Add a feature set | See "Adding a feature set" below | about 25 minutes |
| Add a field to the data | Add it in the payload in `build_dashboard.py`, then use it in the template | seconds |
| Use new months of case or media data | Reprocess the data, then rerun everything with the `run` command | hours |

### Adding a model

1. In `src/media_env_walk_forward.py`, add the model's name to `MODELS` and its fitting code to `run_county`. A model that needs no fitting, such as a no-change baseline, is a few lines.
2. Run only that model. Include `seasonal_naive` so the "vs baseline" column can be computed, and use a new `--tag` for every run:

```bash
python src/media_env_walk_forward.py run --models seasonal_naive MYMODEL --tag add-mymodel
```

   To try it on one county first, add `--counties Kern --feature-sets media_env`.
3. Merge the new results into the saved ones. Nothing already saved is dropped:

```bash
python src/media_env_walk_forward.py combine
```

4. Rebuild and check: `python build_dashboard.py`, then `open dashboard.html`.
5. Optional: add `"mymodel": {"label": "My model", "color": "#123456"}` to `dashboard_config.json` for a nicer name and color.

### Adding a feature set

1. Add its key to `FEATURE_SETS` and its columns to `_set_cols` in `src/media_env_walk_forward.py`.
2. Run it for the dashboard's counties, with every model: `python src/media_env_walk_forward.py run --scope top --feature-sets MYSET --tag add-myset`.
3. Run `combine`, then `python build_dashboard.py`. A new dropdown appears automatically. Add its title to `feature_sets` in `dashboard_config.json`.

### Safe workflow for any change

1. Work on a branch: `git checkout -b my-change`.
2. Make the change, run, `combine` and build as above.
3. Open `dashboard.html`, click through a few counties and every dropdown, and check that the numbers look sensible. Run `python build_dashboard.py --check` to confirm `dashboard.html` matches its inputs.
4. Commit the changed files. `results/` is ignored by git, so add result files by name with `git add -f`. Don't add the `results/media_env_walk_forward/parts/` folder, which only holds work in progress.
5. Open a pull request so a teammate can review.

### Automatic check on pull requests

A GitHub workflow (`.github/workflows/check-dashboard.yml`) rebuilds the dashboard in memory on every pull request that touches the dashboard, the code, or the results. It then compares the rebuild with the committed `dashboard.html` and fails if they differ. It only checks. It never changes or commits anything.

If it fails, the committed `dashboard.html` doesn't match the files it's built from. Fix it like this:

```bash
python build_dashboard.py
git add dashboard.html
git commit -m "Rebuild dashboard"
git push
```

If you changed a results CSV, commit it too with `git add -f <file>`. The check only sees committed files, and a rebuilt page that disagrees with the committed CSVs fails. You can run the same check on your own computer with `python build_dashboard.py --check`. Differences in the last few decimals between computers are ignored.

To see the result on GitHub, open the pull request and look at the "Checks" section, or the **Actions** tab. A failed check doesn't block merging unless you turn on branch protection and mark it required. If you do, note that the check only runs when a pull request changes the files listed in the workflow, so a pull request that touches none of them gets no status.

### Good to know

- `combine` merges new runs into the saved tables. A newer run replaces older rows with the same county, feature set, model and horizon. Old rows are never deleted, so if you remove a model, delete its rows from the CSVs by hand.
- Running `combine` with no new runs rewrites the same tables, so it's safe to run twice.
- Re-running a model reproduces the same numbers, because the random seeds are fixed.
- The dashboard covers the 10 counties with the most cases. The "top10" file names and the `top_counties` call in the code assume 10, so changing that takes some edits.

## Setup note: turning on the online link

In the GitHub repo: **Settings**, then **Pages**, then under "Build and deployment" choose **Deploy from a branch**, branch `main`, folder `/ (root)`, and click **Save**. After a minute or two the page is live. To make the link open the dashboard directly, add a copy of `dashboard.html` named `index.html` to the repo's top folder.
