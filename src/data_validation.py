import pandas as pd


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


# pipeline for running all validation checks
# can be used for any sort of data
def validate_data(data, required_columns, county_col="County", date_col="Year-Month"):
    """Run all validation checks."""

    check_required_columns(data, required_columns)
    check_duplicate_rows(data, county_col, date_col)
    check_monthly_continuity(data, county_col, date_col)