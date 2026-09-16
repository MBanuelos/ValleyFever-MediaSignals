# Monthly county environmental aggregates

This directory contains one monthly CSV per county that combines the
processed environmental predictors with the normalized Valley Fever
case rate.

## Sources

- `data/processed/soil_humidity_wind_precip/shwp_<County>_2001_2024.csv`
- `data/processed/air_quality/aqi_<County>_2001_2024.csv`
- `data/processed/fire/fire_<County>_2006_2024.csv`
- `data/processed/pesticide_rodentcide/pr_<County>_2001_2023.csv`
- `data/processed/vf/vf_<County>_2001_2024.csv`

## Merge rules

- Each file uses the monthly spine from `2001-01` through `2024-12`.
- Fire values are blank before `2006-08` because the processed fire
  series does not begin until then.
- Pesticide / rodenticide values are blank in 2024 because the source
  processed series ends in 2023.
- County names are kept in consistent title case in the file inventory
  below, while filenames keep the project's no-space county stems.

## Output files

| County | File |
| --- | --- |
| Fresno | aggregate_Fresno_2001_2024.csv |
| Kern | aggregate_Kern_2001_2024.csv |
| Los Angeles | aggregate_LosAngeles_2001_2024.csv |
| Orange | aggregate_Orange_2001_2024.csv |
| San Diego | aggregate_SanDiego_2001_2024.csv |
| San Luis Obispo | aggregate_SanLuisObispo_2001_2024.csv |
| Tulare | aggregate_Tulare_2001_2024.csv |
| Ventura | aggregate_Ventura_2001_2024.csv |

- Stacked panel: `aggregate_all_counties_2001_2024.csv`

## Columns

- `County` in the stacked panel file
- `Year-Month`
- `Precipitation`
- `Avg Rel Hum (%)`
- `Avg Wind Speed (mph)`
- `Wind Run (miles)`
- `windeventcount`
- `Avg Soil Temp (F)`
- `AQI_PM25`
- `AQI_PM10`
- `FIRE_Acres_Burned`
- `fungicide_lbs_prd_used`
- `rodent_lbs_prd_used`
- `rate_per_100k`
