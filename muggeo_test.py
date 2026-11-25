"""Muggeo (2016) Monte Carlo study."""

from pathlib import Path
import numpy as np
import statsmodels.api as sm
import scipy

from monte_carlo import run_all

OUTPUT_DIR = Path(__file__).resolve().parent / "muggeo_results"


def muggeo_test_2016(x: np.ndarray, y: np.ndarray, k: int = 30) -> float:
    """
    Return the two-sided p-value of the Muggeo (2016) test
    using the exact finite-sample null distribution.

        Parameters:
            x : np.ndarray of shape (n,)
                Explanatory variable containing n observations.
            y : np.ndarray of shape (n,)
                Response variable containing n observations.
            k : int, default=30
                Number of grid points used to evaluate the average.

        Returns:
            p_value : float
                Two-sided p-value of the Muggeo (2016) test.

        Raises:
            ValueError
                If the null-model design matrix does not have rank 2,
                if the score denominator is not positive, or if the
                computed p-value is not finite.
    """

    order = np.argsort(x)
    x, y = x[order], y[order]
    n = len(x)

    # Null-model design matrix.
    X = sm.add_constant(x)
    if np.linalg.matrix_rank(X) != 2:
        raise ValueError("The null-model design matrix must have rank 2.")

    # Averaging approach.
    grid = np.linspace(x[1], x[-2], k)
    wbar = np.mean(np.maximum(x[:, None] - grid[None, :], 0.0), axis=1)

    # Residuals under H0.
    e0 = y - X @ np.linalg.lstsq(X, y)[0]

    # Numerator and denominator components.
    num = wbar @ e0
    wbar_1 = wbar - X @ np.linalg.lstsq(X, wbar)[0]
    den_sq = wbar_1 @ wbar_1

    if den_sq <= 0:
        raise ValueError("Score denominator is not positive.")

    den = np.sqrt(den_sq)

    # Standard deviation estimate under H0.
    sigma_hat0 = np.sqrt(np.sum(e0**2) / (n - 2))

    # Score statistic.
    s0 = num / (sigma_hat0 * den)

    # Exact null distribution.
    beta_stat = s0**2 / (n - 2)

    p_value = scipy.stats.beta.sf(beta_stat, 0.5, (n - 3) / 2)

    if not np.isfinite(p_value):
        raise ValueError("The computed p-value is not finite.")

    return float(p_value)


def main() -> None:
    """Run the Muggeo (2016) Monte Carlo simulations."""
    run_all(muggeo_test_2016, OUTPUT_DIR)


if __name__ == "__main__":
    main()
