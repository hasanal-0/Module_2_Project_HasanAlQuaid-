"""
AIPI 510: Module 2 Project: Statistics
- src/experiment.py
Hasan Al-Quaid
=================================================
time each sorting algorithm on 53 random arrays

53 runs per algorithm comes from the power analysis (medium effect,
Cohen's f = 0.25, alpha = 0.05, power = 0.80). We will use a different seed
from the pilot so none of the pilot arrays are reused. prior seed was 2026, so anything can be used except that.

Output:
    data/raw/runtimes.csv
"""
import gc
import os
import random
import time
import pandas as pd
from sorting_algorithms import ALGORITHMS

# global values
SEED = 42
RUNS_PER_ALGORITHM = 53
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
    # stoping the garbage collection so it does not add any extra time
    gc.collect()
    gc.disable()  

    #time how long sorting takes
    start = time.perf_counter()
    result = sort_function(arr)
    end = time.perf_counter()

    # reanable the garbage collector
    gc.enable()

    # calculate and return the result of time in ms
    return result, (end - start) * 1000


def main():
    random_num_gen = random.Random(SEED)

    # Warm-up runs (not recorded)
    for name, sort_function in ALGORITHMS.items():
        for warmup in range(WARMUP_RUNS):
            time_sort(sort_function, make_array(random_num_gen))

    # List every run
    schedule = []
    for name in ALGORITHMS:
        for run in range(RUNS_PER_ALGORITHM):
            schedule.append(name)

    # order of algos before shuffling
    schedule_before_shuffle = schedule.copy()
    
    # shuffle the order of the list of algos
    random_num_gen.shuffle(schedule)


    # Run the experiment
    rows = []
    for run_id, name in enumerate(schedule, start=1):
        arr = make_array(random_num_gen)
        expected = sorted(arr)

        result, runtime_ms = time_sort(ALGORITHMS[name], arr)

        if result != expected:
            raise AssertionError(f"Run {run_id}: {name} sort returned the wrong result")

        rows.append({"run_id": run_id,
                    "algorithm_before_shuffle": schedule_before_shuffle[run_id - 1],
                     "algorithm": name,
                     "array_size": len(arr),
                     "runtime(ms)": round(runtime_ms, 2)})

    runtimes = pd.DataFrame(rows)

    print("-------------------------- Experiment results --------------------------")
    print(f"Runs per algorithm: {RUNS_PER_ALGORITHM}  |  Array size: {ARRAY_SIZE}  |  Seed: {SEED}\n")
    print(runtimes.groupby("algorithm")["runtime(ms)"].agg(["count", "mean", "median", "std"]).round(2))
    print("------------------------------------------------------------------------")

    os.makedirs("data/raw", exist_ok=True)
    runtimes.to_csv("data/raw/runtimes.csv", index=False)
    print("Saved data/raw/runtimes.csv")


if __name__ == "__main__":
    main()
