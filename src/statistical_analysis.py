"""
AIPI 510: Module 2 Project: Statistics
- src/analysis.py
Hasan Al-Quaid
=================================================
Statistical analysis:
  1. Descriptive statistics for each algorithm
  2. Assumption checks: Shapiro-Wilk, Q-Q plots, histograms, Levene's test
  3. Decision rule: log-transform if right-skewed, then confirm equal
     variances (Levene's test)
  4. One-way ANOVA (alpha = 0.05) and eta-squared
  5. Tukey's HSD
  6. Figures

Output:
    reports/descriptive_stats.csv
    reports/posthoc_results.csv
    reports/figures/*.png
"""
import itertools
import os
import matplotlib
matplotlib.use("Agg")  # save figures to files without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# global values
CLEAN_FILE = "data/processed/runtimes_clean.csv"
RESULTS_DIR = "reports"
FIGURES_DIR = "reports/figures"
ALPHA = 0.05

ALGORITHM_ORDER = ["merge", "quick", "heap"]
ALGORITHM_NAMES = {"merge": "Merge sort", "quick": "Quicksort", "heap": "Heapsort"}
ALGORITHM_COLORS = {"merge": "#2a78d6", "quick": "#eb6834", "heap": "#1baf7a"}


# ---------------------------------------------------------------- Descriptive statistics
def descriptive_stats(runtimes):
    """
    Print and save n, mean, median, SD, IQR, and skewness for each algorithm

    Parameters:
        runtimes (pd.DataFrame): The cleaned experiment data

    Returns:
        pd.DataFrame: One row of statistics per algorithm
    """
    rows = []
    for name in ALGORITHM_ORDER:
        values = runtimes.loc[runtimes["algorithm"] == name, "runtime(ms)"]
        rows.append({
            "algorithm": name,
            "n": len(values),
            "mean(ms)": values.mean(),
            "median(ms)": values.median(),
            "sd(ms)": values.std(),
            "iqr(ms)": values.quantile(0.75) - values.quantile(0.25),
            "skewness": values.skew(),
        })
    table = pd.DataFrame(rows).round(2)

    print("\n-------------------------- Step 1/6 - Start: Descriptive statistics --------------------------\n")
    print(table.to_string(index=False))
    print("\nSkewness:")
    print("  near 0   = symmetric")
    print("  positive = longer right tail (right-skewed)")
    print("  negative = longer left tail (left-skewed)")
    print("  Conclusion: All 3 values are right-skewed")
    print("\n-------------------------- Step 1/6 - End: Descriptive statistics ----------------------------\n")

    table.to_csv(f"{RESULTS_DIR}/descriptive_stats.csv", index=False)
    return table

# ---------------------------------------------------------------- Assumption checks
def check_normality(runtimes, column):
    """
    Run the Shapiro-Wilk test on each algorithm's values

    Parameters:
        runtimes (pd.DataFrame): The cleaned experiment data
        column (str): The column to test ('runtime(ms)' or 'log_runtime')

    Returns:
        dict: Shapiro-Wilk p-value for each algorithm
    """
    p_values = {}
    for name in ALGORITHM_ORDER:
        values = runtimes.loc[runtimes["algorithm"] == name, column]
        statistic, p_value = stats.shapiro(values)
        p_values[name] = p_value

        if p_value <= ALPHA:
            verdict = "Evidence the data is not normal."
        else:
            verdict = "Insufficient evidence the data is not normal."
        print(f"  {name:<6} W = {statistic:.4f}, p = {p_value:.4g}  ->  {verdict}")
        
    return p_values


def check_equal_variance(runtimes, column):
    """
    Run Levene's test to test for equal variances for the three algorithms

    Parameters:
        runtimes (pd.DataFrame): The cleaned experiment data
        column (str): The column to test ('runtime(ms)' or 'log_runtime')

    Returns:
        float: Levene's test p-value
    """
    merge_times = runtimes.loc[runtimes["algorithm"] == "merge", column]
    quick_times = runtimes.loc[runtimes["algorithm"] == "quick", column]
    heap_times  = runtimes.loc[runtimes["algorithm"] == "heap", column]
    statistic, p_value = stats.levene(merge_times, quick_times, heap_times)

    if p_value <= ALPHA:
        verdict = "Evidence that the variances differ."
    else:
        verdict = "Insufficient evidence that the variances differ."

    print(f"  Levene's test: W = {statistic:.4f}, p = {p_value:.4g}  ->  {verdict}")
    return p_value


def plot_distributions(runtimes, column, label, filename):
    """
    Create a histogram and a Q-Q plot for each of the three algorithm

    Parameters:
        runtimes (pd.DataFrame): The cleaned experiment data
        column (str): The column to plot
        label (str): Axis label for the values
        filename (str): Name of the image file to save in results/figures
    """
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))

    for position, name in enumerate(ALGORITHM_ORDER):
        values = runtimes.loc[runtimes["algorithm"] == name, column]
        color = ALGORITHM_COLORS[name]

        # histogram (top row)
        histogram = axes[0, position]
        histogram.hist(values, bins=15, color=color, edgecolor="white", linewidth=1.5)
        histogram.set_title(f"{ALGORITHM_NAMES[name]}: histogram")
        histogram.set_xlabel(label)
        histogram.set_ylabel("Number of runs")

        # Q-Q plot (bottom row)
        qq = axes[1, position]
        stats.probplot(values, dist="norm", plot=qq)
        qq.get_lines()[0].set(color=color, markersize=5)
        qq.get_lines()[1].set(color="#555555", linewidth=1.5)
        qq.set_title(f"{ALGORITHM_NAMES[name]}: Q-Q plot")
        qq.set_xlabel("Theoretical quantiles (normal)")
        qq.set_ylabel(f"Observed ({label})")

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/{filename}", dpi=200)
    plt.close(fig)
    print(f"  Saved {FIGURES_DIR}/{filename}")


# ---------------------------------------------------------------- ANOVA and effect size

def eta_squared(runtimes, column):
    """
    Calculate eta-squared: η² = SS_between / SS_total

    Parameters:
        runtimes (pd.DataFrame): The cleaned experiment data
        column (str): The column the ANOVA was run on

    Returns:
        float: Eta-squared
    """
    # calculate SS_total:
    grand_mean = runtimes[column].mean()
    ss_total = ((runtimes[column] - grand_mean) ** 2).sum()

    # mean runtime and number of measurements for each algorithm
    groups = runtimes.groupby("algorithm")[column]
    group_means = groups.mean()
    group_counts = groups.count()

    # calculate SS_between:
    ss_between = (group_counts * (group_means - grand_mean) ** 2).sum()
    
    # calculate eta-squared: η² = SS_between / SS_total
    eta_squared = ss_between / ss_total
    
    return eta_squared


def effect_size_label(eta2):
    """Label eta-squared using the course benchmarks (0.01 small, 0.06 medium, 0.14 large)."""
    if eta2 >= 0.14:
        return "large"
    elif eta2 >= 0.06:
        return "medium"
    elif eta2 >= 0.01:
        return "small"
    else:
        return "negligible"


def anova(runtimes, column):
    """
    Run a one-way ANOVA across the three algorithms

    Parameters:
        runtimes (pd.DataFrame): The cleaned experiment data
        column (str): The column to analyze

    Returns:
        tuple: (F statistic, df between, df within, p-value)
    """
    # num of groups
    k = 3

    # calculate the one way anova
    merge_times = runtimes.loc[runtimes["algorithm"] == "merge", column]
    quick_times = runtimes.loc[runtimes["algorithm"] == "quick", column]
    heap_times  = runtimes.loc[runtimes["algorithm"] == "heap", column]
    f_stat, p_value = stats.f_oneway(merge_times, quick_times, heap_times)

    # Calcualte degrees of freedom
    deg_free_between = k - 1
    deg_free_within = len(runtimes) - k

    return f_stat, deg_free_between, deg_free_within, p_value


# ---------------------------------------------------------------- Post hoc tests

def tukey_hsd(runtimes, column):
    """
    Run Tukey's HSD on every pair of algorithms and convert the results to ratios

    Parameters:
        runtimes (pd.DataFrame): The cleaned experiment data
        column (str): The column to analyze (log runtimes)

    Returns:
        pd.DataFrame: Ratio, 95% CI, and adjusted p-value for each pair
    """
    result = pairwise_tukeyhsd(
        endog=runtimes[column],
        groups=runtimes["algorithm"],
        alpha=ALPHA
    )
    table = pd.DataFrame(result.summary().data[1:], columns=result.summary().data[0])

    table = table.rename(columns={"group1": "algorithm_1", "group2": "algorithm_2",
                                  "p-adj": "p_adj"})

    # convert the log differences into ratios (how many times as long algorithm_2 takes as algorithm_1)
    table["ratio"] = np.exp(table["meandiff"].astype(float))
    table["ratio_ci_lower"] = np.exp(table["lower"].astype(float))
    table["ratio_ci_upper"] = np.exp(table["upper"].astype(float))

    return table[["algorithm_1", "algorithm_2", "ratio", "ratio_ci_lower", "ratio_ci_upper", "p_adj", "reject"]]


def raw_differences(runtimes):
    """
    Mean and median differences in milliseconds for each pair (practical significance)

    Parameters:
        runtimes (pd.DataFrame): The cleaned experiment data

    Returns:
        pd.DataFrame: Differences in ms, as (algorithm_2 - algorithm_1)
    """
    means = runtimes.groupby("algorithm")["runtime(ms)"].mean()
    medians = runtimes.groupby("algorithm")["runtime(ms)"].median()
    rows = []
    for first, second in itertools.combinations(sorted(ALGORITHM_ORDER), 2):
        rows.append({"algorithm_1": first, "algorithm_2": second,
                     "mean_diff(ms)": means[second] - means[first],
                     "median_diff(ms)": medians[second] - medians[first]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Figures

def plot_boxplot(runtimes):
    """Save a box plot of runtime by algorithm (box = IQR, dots = IQR-rule outliers)."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    data = [runtimes.loc[runtimes["algorithm"] == name, "runtime(ms)"] for name in ALGORITHM_ORDER]

    boxes = ax.boxplot(data, patch_artist=True, widths=0.5,
                       medianprops={"color": "#222222", "linewidth": 2},
                       flierprops={"marker": "o", "markersize": 6, "markerfacecolor": "white"})
    for box, name in zip(boxes["boxes"], ALGORITHM_ORDER):
        box.set(facecolor=ALGORITHM_COLORS[name], alpha=0.85, edgecolor="#333333")
    for flier, name in zip(boxes["fliers"], ALGORITHM_ORDER):
        flier.set(markeredgecolor=ALGORITHM_COLORS[name])

    ax.set_xticks(range(1, len(ALGORITHM_ORDER) + 1),
                  [ALGORITHM_NAMES[name] for name in ALGORITHM_ORDER])
    ax.set_ylabel("Runtime (ms)")
    ax.set_title("Runtime by sorting algorithm (10,000 integers, 53 runs each)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/runtime_boxplot.png", dpi=200)
    plt.close(fig)
    print(f"  Saved {FIGURES_DIR}/runtime_boxplot.png")


def plot_posthoc_ci(table, test_name):
    """
    Save a chart of each pairwise ratio with its 95% confidence interval, with the values labeled

    Each row is one pair of algorithms. The dot is the ratio (labeled above), the line is the
    95% confidence interval (ends labeled below), and the dashed line at 1 means no difference.
    A line that doesn't touch 1 means that pair is significantly different.

    Parameters:
        table (pd.DataFrame): Tukey's HSD results with ratio, ratio_ci_lower, and ratio_ci_upper
        test_name (str): Name of the post hoc test, used in the chart title

    Returns:
        None: Saves the chart to reports/figures/posthoc_ci.png
    """
    fig, ax = plt.subplots(figsize=(8, 4.2))

    labels = [f"{ALGORITHM_NAMES[b]} / {ALGORITHM_NAMES[a]}"
              for a, b in zip(table["algorithm_1"], table["algorithm_2"])]
    positions = list(range(len(table)))

    ratios = table["ratio"].astype(float)
    lowers = table["ratio_ci_lower"].astype(float)
    uppers = table["ratio_ci_upper"].astype(float)

    ax.hlines(positions, lowers, uppers, color="#2a78d6", linewidth=2)
    ax.plot(ratios, positions, "o", color="#2a78d6", markersize=8)
    ax.axvline(1, color="#888888", linestyle="--", linewidth=1)

    # label the dot (above) and both ends of the line (below)
    for position, ratio, lower, upper in zip(positions, ratios, lowers, uppers):
        ax.annotate(f"{ratio:.2f}", (ratio, position), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=10, fontweight="bold")
        ax.annotate(f"{lower:.2f}", (lower, position), xytext=(0, -15),
                    textcoords="offset points", ha="center", fontsize=9, color="#555555")
        ax.annotate(f"{upper:.2f}", (upper, position), xytext=(0, -15),
                    textcoords="offset points", ha="center", fontsize=9, color="#555555")

    ax.set_yticks(positions, labels)
    ax.set_ylim(-0.6, len(table) - 0.4)
    ax.set_xlabel("Runtime ratio (95% CI)  -  1 = no difference")
    ax.set_title(f"{test_name}: pairwise differences")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/posthoc_ci.png", dpi=200)
    plt.close(fig)
    print(f"Saved {FIGURES_DIR}/posthoc_ci.png")

# ---------------------------------------------------------------- Steps
def step_1_descriptive_stats(runtimes_clean):
    """
    Step 1: print and save the descriptive statistics for each algorithm

    Parameters:
        runtimes_clean (pd.DataFrame): The cleaned experiment data
    """
    descriptive_stats(runtimes_clean)

def step_2_assumption_checks(runtimes_clean):
    """
    Step 2: check independence, normality, and equal variances on the raw runtimes

    Parameters:
        runtimes_clean (pd.DataFrame): The cleaned experiment data

    Returns:
        dict: Shapiro-Wilk p-value for each algorithm (raw runtimes)
    """
    print("-------------------------- Step 2/6 - Start: Assumption checks (raw runtimes) ----------------")
    print("Independence: This is already met by the design of the experiment (new array each run, shuffled run order)")

    print("\nNormality (Shapiro-Wilk):")
    raw_normality = check_normality(runtimes_clean, "runtime(ms)")

    print("\nEqual variances:")
    check_equal_variance(runtimes_clean, "runtime(ms)")

    print("\nFigures:")
    plot_distributions(runtimes_clean, "runtime(ms)", "Runtime (ms)", "distributions_raw.png")
    print("-------------------------- Step 2/6 - End: Assumption checks (raw runtimes) ------------------\n")

    return raw_normality

def step_3_log_transform_check(runtimes_clean, raw_normality):
    """
    Step 3: use log runtimes if the raw runtimes are not normal and right-skewed,
    then confirm equal variances on the scale used

    Parameters:
        runtimes_clean (pd.DataFrame): The cleaned experiment data
        raw_normality (dict): Shapiro-Wilk p-values from Step 2

    Returns:
        str: The column to analyze ('log_runtime' or 'runtime(ms)')
    """
    print("-------------------------- Step 3/6 - Start: Log transform check ----------------------------------")
    skewness = runtimes_clean.groupby("algorithm")["runtime(ms)"].skew()
    not_normal = any(p <= ALPHA for p in raw_normality.values())
    right_skewed = (skewness > 0).all()

    if not_normal and right_skewed:
        use_log = True
        column = "log_runtime"
        print(f"Runtimes are not normal and are right-skewed -> using log-transformed runtimes: {use_log}")

        print("\nNormality after log transform (Shapiro-Wilk):")
        check_normality(runtimes_clean, column)

        print("\nSkewness after log transform:")
        print(runtimes_clean.groupby("algorithm")["log_runtime"].skew().round(2).to_string())

        print("\nFigures:")
        plot_distributions(runtimes_clean, column, "log(runtime)", "distributions_log.png")
    else:
        use_log = False
        column = "runtime(ms)"
        print(f"Runtimes are normal or not right-skewed -> using log-transformed runtimes: {use_log}")

    # confirm equal variances on the scale used
    print(f"\nEqual variances on the scale used ({column}):")
    levene_p = check_equal_variance(runtimes_clean, column)
    if levene_p <= ALPHA:
        raise ValueError("Variances are unequal: the analysis plan calls for Welch's ANOVA")

    # conclusion based on the results above
    print("\nConclusion: The log runtimes are still somewhat skewed, but there's insufficient evidence")
    print("that the variances differ, and with 53 runs per algorithm the group means should be close")
    print("to normal (Central Limit Theorem), so I will be using a one-way ANOVA next, with a")
    print("Kruskal-Wallis test as a check.")
    print("\n-------------------------- Step 3/6 - End: Log transform check ------------------------------------\n")

    return column


def step_4_anova(runtimes_clean, column):
    """
    Step 4: run the one-way ANOVA and eta-squared, plus a Kruskal-Wallis robustness check

    Parameters:
        runtimes_clean (pd.DataFrame): The cleaned experiment data
        column (str): The column to analyze (from Step 3)
    """
    print("-------------------------- Step 4/6 - Start: ANOVA ------------------------------------------")
    f_stat, deg_free_between, deg_free_within, p_value = anova(runtimes_clean, column)
    eta2 = eta_squared(runtimes_clean, column)

    print(f"One-way ANOVA on {column}")
    print(f"F({deg_free_between}, {deg_free_within}) = {f_stat:.2f}, p = {p_value:.4g}")

    if p_value <= ALPHA:
        print(f"\np <= {ALPHA}: reject H0. At least one algorithm's mean runtime differs from the others.")
    else:
        print(f"\np > {ALPHA}: fail to reject H0. There is insufficient evidence that the mean runtimes differ.")

    print(f"\nEffect size: η² = {eta2:.4f} ({effect_size_label(eta2)} effect)")
    print(f"The algorithm explains {eta2 * 100:.1f}% of the variation in {column}")

    # robustness check: Kruskal-Wallis (uses ranks, so it isn't affected by skew or outliers)
    merge_times = runtimes_clean.loc[runtimes_clean["algorithm"] == "merge", "runtime(ms)"]
    quick_times = runtimes_clean.loc[runtimes_clean["algorithm"] == "quick", "runtime(ms)"]
    heap_times = runtimes_clean.loc[runtimes_clean["algorithm"] == "heap", "runtime(ms)"]

    h_stat, kw_p_value = stats.kruskal(merge_times, quick_times, heap_times)

    print("\nRobustness check (Kruskal-Wallis):")
    print("Since the log runtimes weren't fully normal, I am doing this test to check the ANOVA result")
    print("without assuming normality. It compares the order of the runs, not their times,")
    print("so skew and outliers don't affect it.")

    print("\nWhat Kruskal-Wallis is:")
    print("It is the nonparametric version of the one-way ANOVA. It is basically an extension")
    print("of the Mann-Whitney U test, but for more than 2 groups.")

    print(f"\nH = {h_stat:.2f}, p = {kw_p_value:.4g}")
    if kw_p_value <= ALPHA:
        print("Same conclusion as the ANOVA: at least one algorithm's runtime differs")
    else:
        print("Different conclusion from the ANOVA: no evidence the runtimes differ")
    print("\n-------------------------- Step 4/6 - End: ANOVA --------------------------------------------\n")

def step_5_post_hoc(runtimes_clean, column):
    """
    Step 5: run Tukey's HSD on every pair of algorithms and show the differences in milliseconds

    Parameters:
        runtimes_clean (pd.DataFrame): The cleaned experiment data
        column (str): The column to analyze (from Step 3)
    """
    print("-------------------------- Step 5/6 - Start: Post hoc tests ---------------------------------")
    print("As a result of rejecting H0 in Step 4 (p <= 0.05), we are moving forward with Tukey's HSD\n")

    # compare every pair of algorithms (on the log runtimes)
    posthoc = tukey_hsd(runtimes_clean, column)

    # print and save the results, and draw the confidence interval chart
    print("Tukey's HSD results:")
    print(posthoc.round(4).to_string(index=False))
    print("\nratio = how many times as long algorithm_2 takes as algorithm_1")

    posthoc.round(4).to_csv(f"{RESULTS_DIR}/posthoc_results.csv", index=False)
    print(f"\nSaved {RESULTS_DIR}/posthoc_results.csv")
    plot_posthoc_ci(posthoc, "Tukey's HSD")

    # the plain differences in milliseconds
    print("\nDifferences in milliseconds (raw runtimes):")
    print(raw_differences(runtimes_clean).round(2).to_string(index=False))
    print("\n-------------------------- Step 5/6 - End: Post hoc tests -----------------------------------\n")

def step_6_figures(runtimes_clean):
    """
    Step 6: save the main results figure (box plot of runtime by algorithm)

    Parameters:
        runtimes_clean (pd.DataFrame): The cleaned experiment data
    """
    print("-------------------------- Step 6/6 - Start: Figures -----------------------------------------------")
    plot_boxplot(runtimes_clean)
    print("-------------------------- Step 6/6 - End: Figures -------------------------------------------------\n")


# ---------------------------------------------------------------- Main

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    runtimes_clean = pd.read_csv(CLEAN_FILE)

    step_1_descriptive_stats(runtimes_clean)
    raw_normality = step_2_assumption_checks(runtimes_clean)
    column = step_3_log_transform_check(runtimes_clean, raw_normality)
    step_4_anova(runtimes_clean, column)
    step_5_post_hoc(runtimes_clean, column)
    step_6_figures(runtimes_clean)


if __name__ == "__main__":
    main()
