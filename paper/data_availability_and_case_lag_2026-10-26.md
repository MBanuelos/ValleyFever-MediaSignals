# Data Availability and Case-Lag Sensitivity — 2026-10-26

This note covers two things for the environmental forecasting component:

- What is known about when each input becomes available to a forecaster.
- How the development-period (2017–2020) results change when the case-rate availability lag varies.

Evidence labels: **[V]** verified from a source or run; **[R]** reported by a source but not checked against our data; **[H]** hypothesis.

## 1. Publication delays by source

| Input | Origin in our data (ValleyCast README [R]) | What is published, and when | Implication for the forecast cutoff |
|---|---|---|---|
| Case rate (`rate_per_100k`) | Monthly county counts; 2000–2015 from the `valleyfevercasedata` repository, later years presumably CDPH. **The date basis (onset vs report) is not documented** in ValleyCast or in the source repo. | CDPH publishes (a) a **preliminary monthly dashboard**: suspect + probable + confirmed cases "reported so far in the current year", subject to change; and (b) **final year-end reports**: confirmed cases with county and monthly breakdowns [V, CDPH data page]. CDPH counts by **estimated illness onset date**, falling back to the earliest of diagnosis, report, specimen or death date [R]. Counts for recent months rise as late reports arrive [R]. Finalised county data can arrive six months or more after year-end (Austin's evidence); preliminary data within about one month. | No single fixed lag is correct. A real-time forecaster sees **incomplete, provisional** recent months under a different case definition, while our series is the final vintage. **Treat the lag as an uncertain scenario parameter and test it.** |
| Fungicide / rodenticide (CDPR PUR) | Monthly county totals, 2000–2023 | Annual release. The 2023 data files are dated June 2025, and the public announcement was September 2025 [V, CDPR]. Data for year Y therefore appear about 17–21 months after the end of Y. | A month m of year Y is available around June of Y+2, which is a lag of **18 months (December) to 29 months (January)**. The current fixed lag of 24 is **too optimistic for January–May** values. Use 30 (conservative) or a release-calendar rule. |
| Wildfire acreage (`FIRE_Acres_Burned`) | "CAL FIRE wildland incident reports (2007–2018 and 2019–current)" [R] | Incident reports are near-real-time, with acreage updated during each incident. The FRAP perimeter database is released once a year in spring with the previous year's fires (e.g., Firep25_1, April 2026) [V]. | If the column comes from incident reports, a lag of 1–2 months is reasonable (current: 2). If it comes from FRAP perimeters, the lag is 4–16 months. **Check the ValleyCast processing notebook.** |
| PM2.5 / PM10 AQI | EPA "Download Daily Data" [R] | Recent days come from AirNow (preliminary, available the next day) and are replaced by AQS data. AQS certification is due **May 1** of the following year [V, EPA]. | A lag of 1 month is realistic using preliminary values. Revisions after certification are usually small [H]. |
| Weather (precipitation, humidity, wind, soil temperature) | CIMIS stations [R] | Hourly and daily data are available in near-real-time and quality-flagged ("Y" = estimated, "M" = missing) [R]. | A lag of 1 month is realistic. Small QC revisions are possible. |

### 1b. Pesticide source: CalPIP (checked 2026-10-26)

The pesticide columns reportedly come from CalPIP, the California Pesticide Information Portal, which is CDPR's query front end to PUR data.

| Data year | Evidence of availability | Delay after year-end |
|---|---|---|
| 2001 | Press release with *preliminary* figures, 2002-10-22 [R] | about 10 months |
| 2007 | Press release, 2008-12-17 [R] | about 12 months |
| 2013 (presumed) | "Annual Pesticide Use Report Released", 2015-05-05 [R] | about 16 months |
| 2021 | File-site files dated 2023-04-13 [V] | about 15–16 months |
| 2022 | File-site files dated 2024-07-10 [V] | about 18–19 months |
| 2023 | Data files dated 2025-06-04, press release 2025-09-02, file-site zip dated 2025-09-17 [V]; CalPIP "Version 2025.01 (2023 PUR Data Update)" [V] | about 17–21 months |
| 2024 | **Not in CalPIP (newest year 2023) and not on the file site as of 2026-10-26** [V] | more than 22 months, still pending |

Notes:

- File timestamps are last-modified dates. 2019 and 2020 were re-uploaded on 2022-08-22, so they are upper bounds on release dates, not exact ones.
- CalPIP refreshes earlier years with corrections and late reports (2018–2022 were refreshed in the 2023 update), and "the most current three years … are refreshed at least once a year" [V]. Our series is therefore a revised vintage too.
- The release delay has **grown over time**, from about 10–12 months in the 2000s to about 16–21 months recently, with 2024 not out after 22 months.

**Rule to use:** year Y becomes available at the end of year Y+2 (conservative primary analysis). That is a delay of 24 months for December values and 35 months for January. A sensitivity scenario uses June of Y+2 (18–29 months). The current fixed 24-month lag is too optimistic for every month except December.

Sources:
- [CalPIP home](https://calpip.cdpr.ca.gov/main.cfm)
- [CalPIP year selector](https://calpip.cdpr.ca.gov/year.cfm)
- [CDPR PUR file site](https://files.cdpr.ca.gov/pub/outgoing/pur/data/)
- [2023 PUR press release](https://www.cdpr.ca.gov/2025/09/02/department-of-pesticide-regulation-releases-annual-pesticide-use-report/)
- [PUR references and definitions](https://www.cdpr.ca.gov:/docs/pur/pur_references_definitions.pdf)
- [2008 release (archived)](https://webarchive.library.unt.edu/web/20090116064141/http://www.cdpr.ca.gov/docs/pressrls/2008/081217.htm)
- [CDPR press archive](https://www.cdpr.ca.gov/category/press-release/page/15)
- [2002 release coverage](https://beyondpesticides.org/archive/daily-news-archive/2002/file540)

**Not modelled anywhere yet:** data *revisions*. Every run uses the current vintage, so a 1-month case lag still assumes the final, complete count for that month. This makes our case-history results optimistic relative to real time.

Sources:
- [CDPH Valley fever data publications](https://www.cdph.ca.gov/Programs/CID/DCDC/Pages/ValleyFeverDataPublications.aspx)
- [CDPH provisional dashboard](https://www.cdph.ca.gov/Programs/CID/DCDC/Pages/ValleyFeverProvisionalDashboard.aspx)
- [CDPH 2022 epidemiologic summary](https://www.cdph.ca.gov:443/Programs/CID/DCDC/CDPH%20Document%20Library/CocciEpiSummary2022.pdf)
- [Provisional monthly report (via Riverside Univ. Health)](https://www.RUHealth.org/sites/default/files/PH/Disease%20Control/docs/CocciinCAProvisionalMonthlyReport.pdf)
- [CDPR 2023 PUR release](https://www.cdpr.ca.gov/2025/09/02/department-of-pesticide-regulation-releases-annual-pesticide-use-report/)
- [CDPR PUR data files](https://files.cdpr.ca.gov/pub/outgoing/pur/data/2023_pur_report_textfiles/Pesticide_use_annual_report_2023_data_summary/)
- [FRAP fire perimeters](https://data.amerigeoss.org/dataset/california-fire-perimeters-all-5b4ee)
- [EPA Download Daily Data](https://epa.gov/outdoor-air-quality-data/download-daily-data)
- [EPA 2022 data certification Q&A](https://epa.gov/system/files/documents/2022-04/2022-data-certification-qa-508.pdf)
- [CIMIS data user guide (python-cimis)](https://python-cimis.readthedocs.io/en/latest/user_guide.html)
- [ValleyCast](https://github.com/MBanuelos/ValleyCast)
- [valleyfevercasedata](https://github.com/valleyfever/valleyfevercasedata)

## 2. Case-lag sensitivity (development period, 8 counties, seed 42)

**Setup:** `configs/env_baseline.json`, run with `--development-only --case-lag {0, 1, 3}`. Targets 2017-01 to 2020-12 (48 per county). The case lag controls two things: when rate features become available, *and* which training labels are usable. So E models are affected too, through their training window.

**Reproduce:**
- Runs: `results/env/env_baseline_caselag0_dev_51db30fa`, `env_baseline_dev_8b589bf2` (lag 1), `env_baseline_caselag3_dev_2981c37d`.
- Analysis: `scripts/env_case_lag_check.py` writes to `results/env/case_lag_sensitivity/`.

The seasonal-naive reference produces identical forecasts at all three lags [V], because last year's rate is always published when h + lag ≤ 12. Skill values are therefore comparable across lags.

### Median skill vs seasonal naive (8 counties)

| Model / inputs | h=1: L0 / L1 / L3 | h=3: L0 / L1 / L3 | h=6: L0 / L1 / L3 |
|---|---|---|---|
| Persistence | 0.40 / 0.17 / −0.19 | −0.04 / −0.19 / −0.33 | −0.33 / −0.33 / −0.23 |
| SARIMA (R) | 0.37 / 0.20 / 0.01 | 0.08 / 0.01 / −0.07 | −0.07 / −0.09 / −0.08 |
| XGBoost R | 0.27 / 0.18 / 0.02 | 0.10 / 0.02 / −0.03 | −0.03 / −0.03 / −0.04 |
| XGBoost E | 0.15 / 0.09 / 0.02 | 0.04 / 0.02 / −0.01 | 0.05 / 0.06 / 0.04 |
| **XGBoost RE** | **0.33 / 0.23 / 0.08** | **0.16 / 0.11 / 0.09** | **0.15 / 0.15 / 0.13** |
| Ridge RE | 0.34 / 0.14 / −0.03 | −0.06 / −0.15 / −0.18 | −0.07 / −0.07 / −0.08 |

### XGBoost contrasts

Negative means RE is better. Each cell gives the number of counties (of 8) where RE is better, with the number significant at DM p < 0.05 in brackets, then the median change.

| Contrast | h | Lag 0 | Lag 1 | Lag 3 |
|---|---|---|---|---|
| RE vs E | 1 | 8 (6) −24% | 8 (5) −22% | 7 (4) −11% |
| RE vs E | 3 | 7 (4) −15% | 7 (3) −10% | 7 (4) −10% |
| RE vs E | 6 | 6 (3) −11% | 6 (2) −10% | 6 (2) −9% |
| RE vs R | 1 | 8 (0) −7% | 6 (0) −9% | 7 (4) −11% |
| RE vs R | 3 | 7 (3) −8% | 7 (0) −13% | 6 (2) −13% |
| RE vs R | 6 | 7 (1) −18% | 7 (2) −13% | 8 (2) −8% |

For Ridge, RE beats R in only 1–5 of 8 counties at any horizon or lag. The linear model does not extract value from the environmental features, while XGBoost does.

### E-only vs climatology

Neither uses recent case data at forecast time. XGBoost E has lower RMSE than training-window climatology in 7–8 of 8 counties at every horizon and lag. The median RMSE ratio is 0.71–0.88.

## 3. What these runs support (development evidence only; not confirmed)

1. **Case-history models depend strongly on data timeliness at short horizons.** Going from lag 0 to lag 3 removes most of the 1-month skill of persistence, SARIMA and XGBoost R. At 6 months the lag barely matters.
2. **XGBoost RE is the most lag-robust configuration.** It keeps positive median skill at every horizon and lag, and positive skill in 6–7 of 8 counties in every cell.
3. **For RE vs E (the key contrast), case history adds the most when it is timely.** The 1-month advantage halves from lag 0/1 to lag 3. At 3–6 months it is stable at about 10%.
4. **For RE vs R, environment adds value only in the tree model.** The 1-month benefit grows as case data get staler, consistent with environment substituting for missing recent cases [H].
5. **E is not fully immune to case-reporting delay.** XGBoost E's 1-month skill falls from 0.15 to 0.02 as the lag grows, because its training labels end earlier. A plausible reason is the rising 2016–2020 incidence level [H]. Environment-only forecasting still needs case labels for training. It removes the need for case data *at forecast time*, not the need for timely labels. Its fair baseline is **climatology**, which also does not need recent cases, and E beats climatology in 7–8 of 8 counties.

## 4. Gaps and recommended changes

1. **Release-calendar availability.** Add a harness rule of the form "values for year Y become available in month M of year Y + k". This handles PUR (June of Y+2) and the annual final case-data scenario. Under annual final case data, seasonal naive is often unavailable, so climatology has to be the reference for the E-goal analysis.
2. **Run lag 6 and the annual-release scenario** to cover finalised surveillance data.
3. **Preliminary-data scenario.** Optionally degrade the most recent case months (for example, scale them by a completeness curve) to mimic provisional undercounting. This can only be done credibly with dashboard vintages; archived CDPH dashboard snapshots would allow it.
4. **Confirm with ValleyCast:** the case date basis (onset vs report), the case definition, and whether fire acreage comes from incident reports or FRAP perimeters.
5. **Pesticides:** before any importance analysis, use the release-calendar rule "year Y available at end of Y+2" (section 1b). If only a fixed lag is available, set `ag_chemicals.availability_lag` to 36, which is safe for every month.
