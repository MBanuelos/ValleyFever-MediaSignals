import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
import re
import ast
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def detransform(log_rate_start, logdiff_predictions):
    """
    Converts predicted log differences back into rates.
    
    Parameters:
    log_rate_start: starting log(rate+1)
    logdiff_predictions: predicted log differences
    
    Returns:
    reconstructed rates
    """
    
    log_rates = []
    
    current_log = log_rate_start
    
    for diff in logdiff_predictions:
        current_log += diff
        log_rates.append(current_log)
    
    rates = np.expm1(log_rates)
    
    return rates

def Evaluator(actual, prediction):
    # Convert to Series so we can align/drop NaNs
    actual = pd.Series(actual).reset_index(drop=True)
    prediction = pd.Series(prediction).reset_index(drop=True)

    # Combine into one dataframe
    results = pd.DataFrame({
        "actual": actual,
        "prediction": prediction
    })

    # Remove NaN rows
    results = results.dropna()

    # Calculate metrics
    mae = mean_absolute_error(
        results["actual"],
        results["prediction"]
    )

    rmse = np.sqrt(
        mean_squared_error(
            results["actual"],
            results["prediction"]
        )
    )

    r2 = r2_score(
        results["actual"],
        results["prediction"]
    )

    print(f"Observations used: {len(results)}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

def adf_test(series, name):

    result = adfuller(series)

    print("="*60)
    print(name)
    print(f"ADF Statistic : {result[0]:.4f}")
    print(f"P-value       : {result[1]:.4f}")

    if result[1] < 0.05:    
        print("Stationary")
    else:
        print("Non-stationary")

def check_required_columns(data, required_columns):
    """Check that all expected columns exist."""

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print("Required-column check passed.")


def check_duplicate_rows(data, county_col="County", date_col="Year-Month"):
    """Check that each county-month occurs only once."""

    duplicate_rows = data.duplicated(subset=[county_col, date_col], keep=False)

    if duplicate_rows.any():
        duplicates = data.loc[duplicate_rows, [county_col, date_col]]

        raise ValueError("Duplicate county-month rows found:\n"f"{duplicates}")

    print("Duplicate-row check passed.")


def check_monthly_continuity(data, county_col="County", date_col="Year-Month"):
    """Check that each county has one row for every month."""

    problems = []

    for county, county_data in data.groupby(county_col):
        actual_dates = pd.DatetimeIndex(county_data[date_col])

        expected_dates = pd.date_range(start=actual_dates.min(), end=actual_dates.max(), freq="MS")

        missing_dates = expected_dates.difference(actual_dates)

        for missing_date in missing_dates:
            problems.append(
                {
                    "County": county,
                    "Missing Month": missing_date,
                }
            )

    if problems:
        problems = pd.DataFrame(problems)

        raise ValueError(
            "Missing county-month rows found:\n"
            f"{problems}"
        )

    print("Monthly-continuity check passed.")

def validate_agg_data(data, required_columns, date_col="year_month"):
    """Run all validation checks."""

    check_required_columns(data, required_columns)
    check_duplicate_rows(data, date_col)
    check_monthly_continuity(data, date_col)