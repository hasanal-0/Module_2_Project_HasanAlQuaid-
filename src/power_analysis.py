import math
import pandas as pd
from statsmodels.stats.power import FTestAnovaPower

# Parameters
alpha = 0.05
power = 0.80
algo_groups = 3

# Read in the pilot data
pilot = pd.read_csv("data/raw/pilot_runtimes.csv")

# calculate SS_total:
grand_mean = pilot["runtime(ms)"].mean()
ss_total = ((pilot["runtime(ms)"] - grand_mean) ** 2).sum()

# mean runtime and number of measurements for each algorithm
groups = pilot.groupby("algorithm")["runtime(ms)"]
group_means = groups.mean()
group_counts = groups.count()

# calculate SS_between:
ss_between = (
    group_counts * (group_means - grand_mean) ** 2
).sum()

# calculate eta-squared: η² = SS_between / SS_total
eta_squared = ss_between / ss_total

# use eta-squared to calculate cohens_f
cohens_f = math.sqrt(eta_squared / (1 - eta_squared))

# power analysis (solve_power gives the TOTAL across all 3 groups)
analysis = FTestAnovaPower()
total = analysis.solve_power(effect_size=cohens_f, 
                            alpha=alpha, 
                            power=power, 
                            k_groups=algo_groups)

# analysis.solve_power will return nan when the effect is so large that only a few runs are needed
if math.isnan(total):
    # So if the effect is very large we will default to this number
    run_needed = 2
else:
    # else we take the value it gives and divide by 3 groups, and round up to nearest whole number.
    run_needed = math.ceil(total / algo_groups)

# output the results
print("\n----------------------------------------Pilot-------------------------------------------------")
print(f"η² = {eta_squared:.4f}, Cohen's f = {cohens_f:.4f}")
print(f"Based on the pilot alone, we would need {run_needed} per group ({run_needed * algo_groups} total)")
print("----------------------------------------------------------------------------------------------")

## Planned effect size:
# Since the cohens f/eta-squared were very large, to be conservative I plan to use a Medium effect: 0.25(cohens f), 0.06(eta-squared)
# alpha, power, and k_groups will all remain the same

#cohens f benchmark values:
# Small effect: 0.10
# Medium effect: 0.25
# Large effect: 0.40

#eta-squared benchmark values:
# Small effect: 0.01
# Medium effect: 0.06
# Large effect: 0.14

planned_cohens_f = 0.25

# Power analysis (solve_power gives the TOTAL across all 3 groups)
planned_analysis = FTestAnovaPower()
planned_total = planned_analysis.solve_power(effect_size=planned_cohens_f, 
                                            alpha=alpha, 
                                            power=power, 
                                            k_groups=algo_groups)

# we take the value it gives and divide by 3 groups, and round up to nearest whole number.
planned_run_needed = math.ceil(planned_total / algo_groups)

# Output the results
print("\n----------------------------------------Planned-------------------------------------------------")
print("Planned for a medium effect (η² = 0.06, Cohen's f = 0.25)")
print(f"We would need {planned_run_needed} per group, and a total of {planned_run_needed * algo_groups} runs")
print("------------------------------------------------------------------------------------------------")