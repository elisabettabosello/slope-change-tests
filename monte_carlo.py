"""Shared Monte Carlo size and power simulations for the Davies (2002) and Muggeo (2016) tests."""

from pathlib import Path
import numpy as np
import pandas as pd
from collections.abc import Callable

from simulation_setup import (
    ALPHAS,
    BREAKPOINT_QUANTILES,
    DELTA_POWER,
    DISTRIBUTIONS,
    POWER_LAMBDAS,
    POWER_REPLICATIONS,
    SAMPLE_SIZES,
    SIGMA_SIZE,
    SIZE_REPLICATIONS,
    generate_response,
    generate_x,
    psi_quantile,
)


def run_scenario(
    n: int,
    distribution: str,
    signal_ratio: float,
    quantile: float,
    replications: int,
    rng: np.random.Generator,
    test_function: Callable[[np.ndarray, np.ndarray], float],
) -> tuple[np.ndarray, int, int, float | None]:
    """Estimate empirical rejection rates for a single Monte Carlo scenario.

    Parameters:
        n : int
            Sample size.
        distribution : str
            Covariate distribution ("unif", "beta12", "beta22", or "beta25gap").
        signal_ratio : float
            Signal-to-noise ratio |delta| / sigma.
        quantile : float
            Quantile level used to determine the breakpoint under H1.
        replications : int
            Number of Monte Carlo replications.
        rng : np.random.Generator
            Random number generator used for the simulations.
        test_function : Callable[[np.ndarray, np.ndarray], float]
            Function taking the explanatory and response data and returning a p-value.

    Returns:
        rates : np.ndarray of shape (len(ALPHAS),)
            Empirical rejection rates at each significance level.
        failed : int
            Number of failed replications.
        valid : int
            Number of successful replications.
        psi : float or None
            Breakpoint location under H1, or None under H0.
    """

    if signal_ratio == 0:
        delta, sigma, psi = 0.0, SIGMA_SIZE, None
    else:
        delta, sigma = DELTA_POWER, abs(DELTA_POWER) / signal_ratio
        psi = psi_quantile(distribution, quantile)

    rejections = np.zeros(len(ALPHAS), dtype=int)
    failed = 0

    for _ in range(replications):
        try:
            x = generate_x(n, distribution, rng)
            y = generate_response(x, rng, delta, sigma, psi)
            p_value = test_function(x, y)
            rejections += p_value < np.asarray(ALPHAS)
        except ValueError:
            failed += 1

    valid = replications - failed
    rates = rejections / valid if valid else np.full(len(ALPHAS), np.nan)

    return rates, failed, valid, psi


def run_size_study(
    rng: np.random.Generator,
    test_function: Callable[[np.ndarray, np.ndarray], float],
    output_dir: Path,
) -> list[dict[str, str | int | float]]:
    """
    Run the size study and save empirical rejection rates to a CSV file.
    Return the numbers of valid and failed replications for each scenario.

    Parameters:
        rng : np.random.Generator
            NumPy random number generator.
        test_function : Callable[[np.ndarray, np.ndarray], float]
            Function taking the explanatory and response data and returning a p-value.
        output_dir : Path
            Directory where simulation results are saved.

    Returns:
        failure_rows : list[dict[str, str | int | float]]
            List of dictionaries containing simulation configurations
            and the corresponding numbers of valid and failed replications.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failure_rows = []

    for distribution in DISTRIBUTIONS:
        for n in SAMPLE_SIZES:
            rates, failed, valid, _ = run_scenario(
                n, distribution, 0.0, 0.0, SIZE_REPLICATIONS, rng, test_function
            )

            for alpha, rate in zip(ALPHAS, rates):
                rows.append(
                    dict(
                        distribution=distribution, n=n, alpha=alpha, empirical_size=rate
                    )
                )

            failure_rows.append(
                dict(
                    study="size",
                    distribution=distribution,
                    n=n,
                    **{"lambda": 0.0},
                    psi_quantile=0.0,
                    psi="",
                    replications=SIZE_REPLICATIONS,
                    failed=failed,
                    valid=valid,
                )
            )

            print(f"SIZE  {distribution:10s} n={n:3d} | failed={failed}")

    pd.DataFrame(rows).to_csv(output_dir / "size.csv", index=False)

    return failure_rows


def run_power_study(
    rng: np.random.Generator,
    test_function: Callable[[np.ndarray, np.ndarray], float],
    output_dir: Path,
) -> list[dict[str, str | int | float]]:
    """
    Run the power study and save empirical rejection rates to CSV files.
    Return the numbers of valid and failed replications for each scenario.

    Parameters:
        rng : np.random.Generator
            NumPy random number generator.
        test_function : Callable[[np.ndarray, np.ndarray], float]
            Function taking the explanatory and response data and returning a p-value.
        output_dir : Path
            Directory where simulation results are saved.

    Returns:
        failure_rows : list[dict[str, str | int | float]]
            List of dictionaries containing simulation configurations
            and the corresponding numbers of valid and failed replications.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    failure_rows = []

    for distribution in DISTRIBUTIONS:
        rows = []

        for n in SAMPLE_SIZES:
            for signal_ratio in POWER_LAMBDAS:
                for quantile in BREAKPOINT_QUANTILES:

                    rates, failed, valid, psi = run_scenario(
                        n,
                        distribution,
                        signal_ratio,
                        quantile,
                        POWER_REPLICATIONS,
                        rng,
                        test_function,
                    )

                    rows.append(
                        dict(
                            n=n,
                            **{"lambda": signal_ratio},
                            psi_quantile=quantile,
                            psi=psi,
                            **{
                                f"power_alpha_{alpha:.2f}": rate
                                for alpha, rate in zip(ALPHAS, rates)
                            },
                        )
                    )

                    failure_rows.append(
                        dict(
                            study="power",
                            distribution=distribution,
                            n=n,
                            **{"lambda": signal_ratio},
                            psi_quantile=quantile,
                            psi=psi,
                            replications=POWER_REPLICATIONS,
                            failed=failed,
                            valid=valid,
                        )
                    )

                    print(
                        f"POWER {distribution:10s} n={n:3d} "
                        f"lambda={signal_ratio:g} q={quantile:.2f} "
                        f"| failed={failed}"
                    )

        pd.DataFrame(rows).to_csv(output_dir / f"power_{distribution}.csv", index=False)

    return failure_rows


def run_all(
    test_function: Callable[[np.ndarray, np.ndarray], float],
    output_dir: Path,
    seed: int = 123,
) -> None:
    """
    Run both studies and save the size, power, and failure results to CSV files.

    Parameters:
        test_function : Callable[[np.ndarray, np.ndarray], float]
            Function taking the explanatory and response data and returning a p-value.
        output_dir : Path
            Directory where simulation results are saved.
        seed : int, default=123
            Seed used to initialize the random number generator.

    Returns:
        None
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    size_failures = run_size_study(rng, test_function, output_dir)
    power_failures = run_power_study(rng, test_function, output_dir)

    pd.DataFrame(size_failures + power_failures).to_csv(
        output_dir / "failures.csv", index=False
    )
    print(f"Results available in: {output_dir}", flush=True)
