# Monthly county environmental aggregates

This directory contains one monthly CSV per county that combines the
processed environmental predictors with the normalized Valley Fever
case rate.

## Sources

This data comes from the [ValleyCast](https://github.com/MBanuelos/ValleyCast) repository.

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
| Alameda | aggregate_Alameda_2001_2024.csv |
| Contra Costa | aggregate_ContraCosta_2001_2024.csv |
| Fresno | aggregate_Fresno_2001_2024.csv |
| Kern | aggregate_Kern_2001_2024.csv |
| Los Angeles | aggregate_LosAngeles_2001_2024.csv |
| Merced | aggregate_Merced_2001_2024.csv |
| Monterey | aggregate_Monterey_2001_2024.csv |
| Orange | aggregate_Orange_2001_2024.csv |
| Riverside | aggregate_Riverside_2001_2024.csv |
| Sacramento | aggregate_Sacramento_2001_2024.csv |
| San Bernardino | aggregate_SanBernardino_2001_2024.csv |
| San Diego | aggregate_SanDiego_2001_2024.csv |
| San Joaquin | aggregate_SanJoaquin_2001_2024.csv |
| San Luis Obispo | aggregate_SanLuisObispo_2001_2024.csv |
| San Mateo | aggregate_SanMateo_2001_2024.csv |
| Santa Barbara | aggregate_SantaBarbara_2001_2024.csv |
| Santa Clara | aggregate_SantaClara_2001_2024.csv |
| Solano | aggregate_Solano_2001_2024.csv |
| Stanislaus | aggregate_Stanislaus_2001_2024.csv |
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