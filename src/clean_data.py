"""
AIPI 510: Module 2 Project: Statistics
- src/clean_data.py
Hasan Al-Quaid
=================================================
Cleans the raw data in a few ways:
  1. Checks for missing, zero, or negative runtimes and wrong array sizes
  2. Flags outliers with the IQR rule (reported, not removed)
  3. Adds a log-transformed runtime column, in case runtimes are skewed

Output:
    data/processed/runtimes_clean.csv
"""
import os
import numpy as np
import pandas as pd

# global values
RAW_FILE = "data/raw/runtimes.csv"
CLEAN_FILE = "data/processed/runtimes_clean.csv"
ARRAY_SIZE = 10_000


def iqr_range(group):
    """
    Calculate the normal range for one algorithm's runtimes using the IQR rule, 
    with k being 1.5 as its conventional (Tukey's rule).

    Parameters:
        group (pd.Series): Runtimes for one algorithm

    Returns:
        tuple: (q1, q3, iqr, lower, upper)
    """
    q1 = group.quantile(0.25)
    q3 = group.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return q1, q3, iqr, lower, upper


def flag_outliers(group):
    """
    Flag outliers in one algorithm's runtimes using the IQR rule

    Parameters:
        group (pd.Series): Runtimes for one algorithm

    Returns:
        pd.Series: True where a runtime is outside the normal range
    """
    q1, q3, iqr, lower, upper = iqr_range(group)
    too_low = group < lower
    too_high = group > upper

    return too_low | too_high


def initial_check_data(runtimes_data):
    """
    Check the raw data for missing values, bad runtimes, and wrong array sizes

    Parameters:
        runtimes (pd.DataFrame): The raw experiment data

    Returns:
        bool: True if all checks passed, else False
    """
    print("-------------------------- Start: Data checks --------------------------")
    print(f"Rows: {len(runtimes_data)}  |  Columns: {list(runtimes_data.columns)}")
    print("\nRuns per algorithm:")
    print(runtimes_data["algorithm"].value_counts().to_string())

    missing = runtimes_data.isnull().sum().sum()
    zero_or_negative = (runtimes_data["runtime(ms)"] <= 0).sum()
    wrong_size = (runtimes_data["array_size"] != ARRAY_SIZE).sum()

    if missing == 0 and zero_or_negative == 0 and wrong_size == 0:
        all_passed = True
    else:
        all_passed = False

    print(f"\nMissing values:            {missing}")
    print(f"Zero or negative runtimes: {zero_or_negative}")
    print(f"Arrays not {ARRAY_SIZE} long:     {wrong_size}")
    print(f"All checks passed: {all_passed}")
    print("-------------------------- End: Data checks ----------------------------\n")

    return all_passed


def add_outlier_flags(runtimes_data):
    """
    Flag outliers for each algorithm with the IQR rule and print a summary

    Parameters:
        runtimes_data (pd.DataFrame): The experiment data

    Returns:
        pd.DataFrame: The data with an added 'is_outlier' column
    """
    print("-------------------------- Start: Outliers (IQR rule) ------------------")
    runtimes_data["is_outlier"] = runtimes_data.groupby("algorithm")["runtime(ms)"].transform(flag_outliers)

     # normal range for each algorithm
    print("Normal range for each algorithm (Q1 - 1.5 x IQR to Q3 + 1.5 x IQR):")
    for name, group in runtimes_data.groupby("algorithm")["runtime(ms)"]:
        q1, q3, iqr, lower, upper = iqr_range(group)
        print(f"\n{name}:")
        print(f"  Q1:           {q1:.2f} ms")
        print(f"  Q3:           {q3:.2f} ms")
        print(f"  IQR:          {iqr:.2f} ms")
        print(f"  Normal range: {lower:.2f} to {upper:.2f} ms")

    print("\nOutliers per algorithm (kept in the data, not removed):")
    print(runtimes_data.groupby("algorithm")["is_outlier"].sum().to_string())
    print(f"Total outliers: {runtimes_data['is_outlier'].sum()} of {len(runtimes_data)} runs")

    # list every outlier, sorted by algorithm and runtime
    print("\nAll outliers:")
    outliers = runtimes_data[runtimes_data["is_outlier"]].sort_values("run_id")
    print(outliers[["run_id", "algorithm", "runtime(ms)"]].to_string(index=False))
    print("-------------------------- End: Outliers -------------------------------\n")

    return runtimes_data


def add_log_runtime(runtimes_data):
    """
    Add a log-transformed runtime column, in case runtimes are skewed

    Parameters:
        runtimes_data (pd.DataFrame): The experiment data

    Returns:
        pd.DataFrame: The data with an added 'log_runtime' column
    """
    print("-------------------------- Start: Log transform ------------------------")
    runtimes_data["log_runtime"] = np.log(runtimes_data["runtime(ms)"])
    print("Added new column called 'log_runtime'")
    print(f"Rows: {len(runtimes_data)}  (no rows removed)")
    print("-------------------------- End: Log transform --------------------------\n")

    return runtimes_data


def main():
    # read in the data file
    runtimes_data = pd.read_csv(RAW_FILE)

    # do an initial data check for missing values or wrong data
    initial_check_data(runtimes_data)
    runtimes_data = add_outlier_flags(runtimes_data)
    runtimes_data = add_log_runtime(runtimes_data)

    # output the clean data 
    os.makedirs("data/processed", exist_ok=True)
    runtimes_data.to_csv(CLEAN_FILE, index=False)
    print(f"Saved {CLEAN_FILE}")


if __name__ == "__main__":
    main()
