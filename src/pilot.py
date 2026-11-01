"""
AIPI 510: Module 2 Project: Statistics
- src/pilot.py
Hasan Al-Quaid
=================================================
Pilot study: times each sorting algorithm on 20 random arrays.

The pilot data are only used to estimate the effect size for the power
analysis (power_analysis.py). They are not part of the final analysis.

Output:
    data/raw/pilot_runtimes.csv
"""

import gc
import os
import random
import time
import pandas as pd
from sorting_algorithms import ALGORITHMS

SEED = 2026
RUNS_PER_ALGORITHM = 20
WARMUP_RUNS = 3
ARRAY_SIZE = 10_000


def make_array(random_num_gen):
    """
    Generate a new list of 10,000 random integers between 0 and 1,000,000

    Parameters:
        random_num_gen (random.Random): The seeded random number generator to draw values from

    Returns:
        list: A list of ARRAY_SIZE random integers between 0 and 1,000,000
    """
    return [random_num_gen.randint(0, 1_000_000) for position in range(ARRAY_SIZE)]


def time_sort(sort_function, arr):
    """
    Time one sort

    Parameters:
        sort_function (function): The sorting algorithm to run
        arr (list): The list to sort

    Returns:
        tuple: (sorted list, runtime in milliseconds)
    """
    gc.collect()
    gc.disable()  # pause garbage collection so it doesn't add random delays
    start = time.perf_counter()
    result = sort_function(arr)
    end = time.perf_counter()
    gc.enable()
    return result, (end - start) * 1000


def main():
    random_num_gen = random.Random(SEED)

    # Warm-up runs (not recorded)
    for name, sort_function in ALGORITHMS.items():
        for warmup in range(WARMUP_RUNS):
            time_sort(sort_function, make_array(random_num_gen))

    # List every run, then shuffle the order
    schedule = []
    for name in ALGORITHMS:
        for run in range(RUNS_PER_ALGORITHM):
            schedule.append(name)
    random_num_gen.shuffle(schedule)

    # Run the pilot
    rows = []
    for run_id, name in enumerate(schedule, start=1):
        arr = make_array(random_num_gen)
        expected = sorted(arr)

        result, runtime_ms = time_sort(ALGORITHMS[name], arr)

        if result != expected:
            raise AssertionError(f"Run {run_id}: {name} sort returned the wrong result")

        rows.append({"run_id": run_id, "algorithm": name, "runtime(ms)": runtime_ms})

    pilot = pd.DataFrame(rows)

    print("\n-------------------------- Pilot results --------------------------")
    print(f"Runs per algorithm: {RUNS_PER_ALGORITHM}  |  Array size: {ARRAY_SIZE}  |  Seed: {SEED}")
    print("\n")
    print(pilot.groupby("algorithm")["runtime(ms)"].describe()[["count", "mean", "50%", "std"]].round(2))
    print("-------------------------------------------------------------------\n")

    os.makedirs("data/raw", exist_ok=True)
    pilot.to_csv("data/raw/pilot_runtimes.csv", index=False)
    print("Saved data/raw/pilot_runtimes.csv")


if __name__ == "__main__":
    main()