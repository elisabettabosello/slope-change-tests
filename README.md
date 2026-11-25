# Testing for a Change in Slope with an Unidentified Breakpoint: A Comparison of Supremum and Score-Based Tests through Monte Carlo Simulations

This repository contains the Python code developed for the Monte Carlo study comparing the supremum-based test proposed by Davies (2002) and the score-based test proposed by Muggeo (2016) for detecting a change in slope at an unknown breakpoint.

The aim is to compare the finite-sample properties of the two tests in terms of empirical size and power under different simulation scenarios.

## Requirements

- Python 3.12 or later
- uv

The required Python libraries are NumPy, SciPy, Statsmodels, Pandas, and Matplotlib. Dependencies are specified in `pyproject.toml`.

To set up the environment, run the following command from the project root:

```bash
uv sync
```

## Python files

| File | Description |
| --- | --- |
| `simulation_setup.py` | Defines the simulation parameters and generates the covariates, responses, and breakpoint locations. |
| `monte_carlo.py` | Executes the Monte Carlo replications, computes empirical rejection rates, and saves the results in CSV files. |
| `davies_test.py` | Implements Davies' supremum-based test and computes the corresponding $p$-values. |
| `muggeo_test.py` | Implements Muggeo's score-based test and computes the corresponding $p$-values. |
| `latex_tables.py` | Converts the simulation results into LaTeX tables. |
| `analyze_simulation_results.py` | Compares the simulation results using descriptive statistics and graphical summaries. |

## Execution workflow

First, run the Monte Carlo simulations (Step 1). Once the simulation results are available, Steps 2 and 3 can be executed independently. The two shared modules (`simulation_setup.py` and `monte_carlo.py`) are imported automatically and do not need to be executed separately.

### Step 1 — Run the Monte Carlo simulations

```bash
uv run davies_test.py
uv run muggeo_test.py
```

Both scripts use the same simulation design and the shared Monte Carlo procedure. The default random seed is 123, and the default number of breakpoint grid points is 30 for both tests.

The simulation design considers:

- Sample sizes: 30, 100, and 500.
- Nominal significance levels: 0.01, 0.05, and 0.10.
- Signal-to-noise ratios under the alternative: 0.25, 0.5, 1, and 2.
- Breakpoint quantiles: 0.10, 0.50, 0.75, and 0.90.
- Covariate distributions: Uniform(0,1), Beta(1,2), Beta(2,2), and Beta(2,5) with a gap in (0.25, 0.35).
- Monte Carlo replications: 10,000 per size scenario and 5,000 per power scenario.

The results are saved in `davies_results/` and `muggeo_results/`, respectively. Each directory contains:

- `size.csv`: empirical rejection rates under the null hypothesis.
- `power_unif.csv`, `power_beta12.csv`, `power_beta22.csv`, and `power_beta25gap.csv`: empirical power for each covariate distribution.
- `failures.csv`: numbers of valid and failed replications for each scenario.

### Step 2 — Generate LaTeX tables

```bash
uv run latex_tables.py
```

This script reads the CSV files generated in Step 1 and produces 13 LaTeX tables:

- One table comparing the empirical size of both tests across all covariate distributions, sample sizes, and significance levels.
- Twelve tables comparing empirical power, one for each combination of covariate distribution and nominal significance level.

The LaTeX code is saved as `.txt` files in `latex_tables/`.

### Step 3 — Analyze the simulation results

```bash
uv run analyze_simulation_results.py
```

This script reads the original CSV results generated in Step 1 and performs a comparative analysis of the empirical size and power of the two tests.

The empirical size analysis computes the range and mean of the absolute differences between the tests, the mean absolute deviation from the nominal significance level for each test, and the number of scenarios in which each test is closer to the nominal level.

The empirical power analysis computes the range, mean, and median of the absolute power differences and counts the scenarios in which each test is more powerful. For each significance level, scenarios are also ranked in descending order of absolute power difference, with particular attention to the top 10, 20, and 30 scenarios.

The results are saved in `simulation_analysis/`:

- `size_summary.txt`: descriptive statistics for empirical size.
- `power_summary.txt`: descriptive statistics for empirical power.
- `figures/distribution_topN.png`: frequencies of covariate distributions among the scenarios with the largest absolute power differences.
- `figures/characteristics_heatmap.png`: frequencies of selected scenario characteristics among the largest absolute power differences (sample size 500, signal-to-noise ratio 1 or 2, and breakpoint quantile 0.50 or 0.75).

## References

- Davies, R.B., 2002. Hypothesis testing when a nuisance parameter is present only under the alternative: linear model case. Biometrika 89 (2), 484--489.
- Muggeo, V.M.R., 2016. Testing with a nuisance parameter present only under the alternative: a score-based approach with application to segmented modelling. Journal of Statistical Computation and Simulation 86 (15), 3059--3067.
