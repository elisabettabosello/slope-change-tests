"""Shared simulation design for the Monte Carlo studies of the Davies (2002) and Muggeo (2016) tests."""

import numpy as np
from scipy import stats

SAMPLE_SIZES = (30, 100, 500)
ALPHAS = (0.01, 0.05, 0.10)
POWER_LAMBDAS = (0.25, 0.5, 1.0, 2.0)
BREAKPOINT_QUANTILES = (0.10, 0.50, 0.75, 0.90)
DISTRIBUTIONS = ("unif", "beta12", "beta22", "beta25gap")
SIZE_REPLICATIONS = 10000
POWER_REPLICATIONS = 5000
BETA1 = 0.15
DELTA_POWER = -0.15
SIGMA_SIZE = 0.05
GAP = (0.25, 0.35)


def beta_parameters(distribution: str) -> tuple[int, int]:
    """
    Return the shape parameters of the selected beta distribution.

    Parameters:
        distribution : str
            Name of the beta distribution.

    Returns:
        tuple[int, int]
            Shape parameters of the beta distribution.
    """

    return {"beta12": (1, 2), "beta22": (2, 2), "beta25gap": (2, 5)}[distribution]


def generate_x(n: int, distribution: str, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a sample of covariate observations.

    Parameters:
        n : int
            Number of observations to generate.
        distribution : str
            Covariate distribution ("unif", "beta12", "beta22", or "beta25gap").
        rng : np.random.Generator
            NumPy random number generator.

    Returns:
        x : np.ndarray of shape (n,)
            Generated covariate observations.
    """

    if distribution == "unif":
        return rng.uniform(0, 1, size=n)

    a, b = beta_parameters(distribution)
    x = rng.beta(a, b, size=n)
    if distribution == "beta25gap":
        mask = (x > GAP[0]) & (x < GAP[1])
        while mask.any():
            x[mask] = rng.beta(a, b, size=mask.sum())
            mask = (x > GAP[0]) & (x < GAP[1])
    return x


def psi_quantile(distribution: str, quantile: float) -> float:
    """
    Return the specified quantile of the given distribution.

    Parameters:
        distribution : str
            Covariate distribution ("unif", "beta12", "beta22", or "beta25gap").
        quantile : float
            Quantile level between 0 and 1.

    Returns:
        float
            Quantile of the specified distribution.
    """

    if distribution == "unif":
        return float(stats.uniform.ppf(quantile))

    dist = stats.beta(*beta_parameters(distribution))
    if distribution != "beta25gap":
        return float(dist.ppf(quantile))

    lower_mass = dist.cdf(GAP[0])
    removed_mass = dist.cdf(GAP[1]) - lower_mass
    target = quantile * (1 - removed_mass)
    return float(dist.ppf(target if target <= lower_mass else target + removed_mass))


def generate_response(
    x: np.ndarray,
    rng: np.random.Generator,
    delta: float,
    sigma: float,
    psi: float | None = None,
) -> np.ndarray:
    """
    Generate responses from the Gaussian segmented regression model.

    Parameters:
        x : np.ndarray of shape (n,)
            Observed values of the explanatory variable.
        rng : np.random.Generator
            NumPy random number generator.
        delta : float
            Change in slope.
        sigma : float
            Standard deviation of the Gaussian errors.
        psi : float or None, default=None
            Breakpoint location. Set to None under the null hypothesis.

    Returns:
        y : np.ndarray of shape (n,)
            Generated response observations.
    """
    W = np.maximum(x - psi, 0.0) if psi is not None else np.zeros_like(x)
    return BETA1 * x + delta * W + rng.normal(0.0, sigma, size=len(x))
