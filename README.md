# ValleyFever-MediaSignals

ValleyFever-MediaSignals is a research repository for analyzing news and social-media coverage of Valley Fever. The project uses natural language processing to study public-health messaging, awareness proxies, media framing, and the relationship between media activity and county-level Valley Fever case patterns.

The repository is designed to complement [ValleyCast](https://github.com/MBanuelos/ValleyCast), which provides environmental predictors and processed Valley Fever case-rate data. ValleyCast remains the environmental-data source; this repository owns media and social-media collection, NLP feature development, and media-aware analyses.

## Research goals

- Determine whether news and social-media features improve Valley Fever case-rate forecasting beyond environmental and seasonal baselines.
- Characterize how Valley Fever is discussed through topics such as risk, prevention, diagnosis, treatment, outbreaks, and uncertainty.
- Compare media patterns across California counties and evaluate transfer to Arizona counties when compatible case-rate and media coverage are available.
- Develop reproducible, validated measures of media attention and public-awareness proxies.

All findings should be interpreted as predictive or associational unless a future study provides a design that supports causal conclusions. Media activity may reflect disease incidence, seasonality, reporting delays, environmental conditions, or public-health campaigns.

## Planned repository structure

```text
ValleyFever-MediaSignals/
├── data/
│   ├── raw/          # Original news and social-media records
│   ├── interim/      # Cleaned, deduplicated, and validated records
│   ├── processed/    # Monthly county/source features and analysis tables
│   └── manifests/    # Data versions, coverage, checksums, and provenance
├── notebooks/        # Exploratory analysis and reproducible study notebooks
├── src/              # Reusable collection, NLP, validation, and modeling code
├── configs/          # Data sources, feature definitions, and experiment settings
├── results/          # Tables, figures, model outputs, and evaluation summaries
├── paper/            # Manuscripts, reviews, and supporting materials
└── AGENTS.md         # Working context and research conventions
```

## Data sources

Planned inputs include:

- Google News articles and metadata;
- Twitter/X or other social-media posts, where permitted and available;
- ValleyCast environmental predictors;
- county-level Valley Fever case rates and population denominators;
- geographic metadata for California and Arizona counties.

Raw data should be preserved separately from cleaned and derived data. Each dataset should document its source, collection period, query or hashtag rules, geographic coverage, missingness, deduplication method, and version or collection date.

## Common data conventions

Media records should be normalized around fields such as:

```text
item_id, source, published_at, text, title, url,
query_or_hashtag, location, and collection metadata
```

Case-rate data should use stable state and county identifiers with a monthly date field. Before analysis, validate date coverage, duplicate county-month rows, missing months, population denominators, and the definition of `VFRate`.

## Planned studies

### Forecasting with media and social-media features

Compare environmental-only, news-only, social-only, and combined models against seasonal-naive, autoregressive/SARIMAX, regularized regression, and tree-based baselines. Use rolling-origin or expanding-window validation and select lags and features using training data only.

### NLP analysis of Valley Fever awareness and framing

Create annotated datasets and validated measures for relevance, topic, geographic specificity, and framing. Use California counties for development and primary comparisons, then evaluate transfer to held-out California counties and Arizona counties when coverage permits.

## Reproducibility principles

- Keep raw, interim, processed, and result artifacts separate.
- Do not use held-out outcomes to select lags, features, hyperparameters, or geographic inclusion.
- Record random seeds, package versions, date ranges, feature definitions, and model configurations.
- Compare new NLP methods with transparent keyword or TF-IDF baselines.
- Report repeated-run variability and uncertainty, not only a single best score.
- Treat influential-point removal as a sensitivity analysis, not an automatic preprocessing step.
- Distinguish awareness proxies from direct measurements of individual awareness.
- Use associational language unless causal identification is justified.

## Getting started

This repository is being initialized. The first implementation steps are:

1. Define and document the news, social-media, and case-rate schemas.
2. Add data manifests and coverage-validation utilities.
3. Import a versioned ValleyCast county-month release through an adapter.
4. Add media collection and cleaning workflows.
5. Establish baseline monthly features and evaluation splits before adding advanced NLP models.

See [`AGENTS.md`](AGENTS.md) for the detailed research context, workflow, and two-paper roadmap.
