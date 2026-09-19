"""
Preprocessing for monthly environmental county data
"""

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from data_validation import validate_data



# path for project root
project_root = Path(__file__).resolve().parents[1]
input_file = project_root/"data"/"processed"/"env_vf_aggregate"/"aggregate_all_counties_2001_2024.csv"
output_file = project_root/"data"/"interim"/"aggregate_all_counties_2001_2024_processed.csv"

plot_folder = project_root/"results"/"ets_forecasts"

county_col = "County"
date_col = "Year-Month"
target_col = "rate_per_100k"

environmental_columns = [
    "Precipitation",
    "Avg Rel Hum (%)",
    "Avg Wind Speed (mph)",
    "Wind Run (miles)",
    "windeventcount",
    "Avg Soil Temp (F)",
    "AQI_PM25",
    "AQI_PM10",
    "FIRE_Acres_Burned",
]

# columns that will have their missing trailing data imputed using ETS
# e.g. fungicide and rodenticide is missing last year
ets_columns = [
    "fungicide_lbs_prd_used",
    "rodent_lbs_prd_used",
]

required_columns = [
    county_col,
    date_col,
    target_col,
    *environmental_columns,
    *ets_columns
]


def print_missing_report(data, label):
    """Print the number of missing values in each affected column."""

    missing_counts = data.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    print(f"\nMissing values {label}:")

    if missing_counts.empty:
        print("None")
    else:
        print(missing_counts)


def trim_leading_fire_data(data):
    """
    Remove rows before fire data becomes available for each county.

    The entire row is removed so all features remain aligned by month.
    """

    data = data.copy()
    trimmed_counties = []

    for county, county_data in data.groupby(county_col, sort=False):
        fire_available = county_data["FIRE_Acres_Burned"].notna()

        if not fire_available.any():
            raise ValueError(f"{county} has no observed fire data.")

        # Position of the first row with observed fire data.
        first_fire_position = np.where(fire_available)[0][0]

        rows_removed = first_fire_position

        # Keep the first observed fire month and everything after it.
        county_data = county_data.iloc[first_fire_position:]

        trimmed_counties.append(county_data)

        print(
            f"Removed {rows_removed} leading rows for {county}. "
            f"Data now begins at "
            f"{county_data[date_col].iloc[0]:%Y-%m}."
        )

    data = pd.concat(trimmed_counties, ignore_index=True)

    return data


# some counties like Tulare have a couple missing values
# so we interpolate linearly (take average of prev and next vals)
def interpolate_internal_missing(data):
    """Linearly interpolate internal feature gaps within each county."""
    data = data.copy()

    # Do not interpolate the disease rate. It is the prediction target.
    feature_columns = [
        col for col in data.select_dtypes(include="number").columns
        if col != target_col
    ]

    data[feature_columns] = (
        data.groupby(county_col)[feature_columns]
        .transform(
            lambda county: county.interpolate(
                method="linear",
                limit_area="inside",
            )
        )
    )

    return data

# used to impute missing trailing year of fungicide/rodenticide data
def impute_trailing_ets(data, column):
    """Use additive seasonal ETS to fill missing values at the end of a county series."""
    data = data.copy()

    for county in data[county_col].unique():
        county_index = data.index[data[county_col] == county]
        county_values = data.loc[county_index, column]

        observed_positions = np.where(county_values.notna())[0]

        if len(observed_positions) == 0:
            print(f"Skipping {county} {column}: no observed values")
            continue

        first_observed_position = observed_positions[0]
        last_observed_position = observed_positions[-1]
        trailing_index = county_values.index[last_observed_position + 1 :]

        # this county has no trailing values to impute
        if len(trailing_index) == 0:
            continue

        training_data = county_values.iloc[
            first_observed_position : last_observed_position + 1
        ].reset_index(drop=True)

        if training_data.isna().any():
            raise ValueError(
                f"{county} {column} still has an internal gap after interpolation."
            )

        if len(training_data) < 24:
            raise ValueError(
                f"{county} {column} needs at least 24 observed months for ETS."
            )

        model = ExponentialSmoothing(
            training_data,
            trend=None,
            seasonal="add",
            seasonal_periods=12,
        )

        fitted_model = model.fit()
        forecast = fitted_model.forecast(len(trailing_index))

        # fungicide and rodenticide amounts cannot be negative
        forecast = np.clip(forecast, 0, None)
        data.loc[trailing_index, column] = forecast.to_numpy()

        print(f"Imputed {len(trailing_index)} {column} values for {county}")

    return data

# takes input and output file path as arguments 
def process_data(input_file=input_file, output_file=output_file):
    """
    Run the complete preprocessing pipeline
    Check and validate data (no duplicate or missing values)
    Interpolate inner missing values 
    ETS impute missing trailing values
    """
    data = pd.read_csv(input_file)

    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values([county_col, date_col]).reset_index(drop=True)

    # check columns, duplicate county month rows, and missing months
    validate_data(data, required_columns, county_col, date_col)
    print_missing_report(data, "before processing")

    # remove months before fire data is available
    data = trim_leading_fire_data(data)

    print_missing_report(data, "after trimming leading fire data")

    # fill inner gaps that have an observed value before and after them
    data = interpolate_internal_missing(data)

    # fill missing trailing data at end of fungicide/rodenticide
    for column in ets_columns:
        data = impute_trailing_ets(data, column)

    # rates should be positive
    if (data[target_col] < 0).any():
        raise ValueError(f"{target_col} contains negative values")

    # log1p rate column
    data["log1p_rate_per_100k"] = np.log1p(data[target_col])

    print_missing_report(data, "after processing")

    data.to_csv(output_file, index=False, date_format="%Y-%m")

    print(f"\nSaved processed data to {output_file}")

    return data

if __name__ == "__main__":
    processed_data = process_data()
    print("Processed file")