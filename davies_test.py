"""Davies (2002) Monte Carlo study."""

from pathlib import Path
import numpy as np
from scipy import special, stats
import statsmodels.api as sm

from monte_carlo import run_all

OUTPUT_DIR = Path(__file__).resolve().parent / "davies_results"


def get_t_statistic_davies2002(x: np.ndarray, y: np.ndarray, psi: float) -> float:
    """
    Return the OLS t-statistic for testing a change in slope at a fixed breakpoint.

    Parameters:
        x : np.ndarray of shape (n,)
            Explanatory variable containing n observations.
        y : np.ndarray of shape (n,)
            Response variable containing n observations.
        psi : float
            Fixed breakpoint location.

    Returns:
        t_value : float
            OLS t-statistic for testing H0: delta = 0 at the fixed breakpoint.

    Raises:
        ValueError
            If the design matrix does not have full rank (rank 3) or the computed t-statistic
            is not finite.
    """

    # Fit y = beta_0 + beta_1*x + delta*(x-psi)_+ + error.
    X = sm.add_constant(x)
    w = np.maximum(x - psi, 0.0)
    Z = np.column_stack([X, w])

    if np.linalg.matrix_rank(Z) != 3:
        raise ValueError(f"Design matrix has rank below 3 at psi={psi}.")

    result = sm.OLS(y, Z).fit()
    t_value = float(result.tvalues[-1])

    if not np.isfinite(t_value):
        raise ValueError(f"Non-finite t-statistic at psi={psi}.")

    return t_value


def davies_test_2002(x: np.ndarray, y: np.ndarray, k: int = 30) -> float:
    """
    Return the approximate two-sided p-value of the Davies (2002) test.

    Parameters:
        x : np.ndarray of shape (n,)
            Explanatory variable containing n observations.
        y : np.ndarray of shape (n,)
            Response variable containing n observations.
        k : int, default=30
            Number of grid points used to evaluate the t-statistic process
            over the breakpoint interval.

    Returns:
        p_value : float
            Approximate two-sided p-value of the Davies (2002) test.

    Raises:
        ValueError
            If the computed p-value is not finite.
    """

    order = np.argsort(x)
    x, y = x[order], y[order]
    q = len(x) - 3  # Residual degrees of freedom.

    grid = np.linspace(x[1], x[-2], k)
    t_values = np.array([get_t_statistic_davies2002(x, y, psi) for psi in grid])

    maximum = np.max(np.abs(t_values))
    B = t_values**2 / (t_values**2 + q)
    V = np.arcsin(np.sqrt(B))
    variation = np.abs(np.diff(V)).sum()

    # Numerically stable gamma-function ratio in the correction term.
    gamma_ratio = np.exp(special.gammaln((q + 1) / 2) - special.gammaln(q / 2))
    correction = (
        (1 + maximum**2 / q) ** (-(q - 1) / 2)
        * gamma_ratio
        / (2 * np.sqrt(np.pi))
        * variation
    )

    p_value = float(2 * (stats.t.sf(maximum, q) + correction))

    if not np.isfinite(p_value):
        raise ValueError("The computed p-value is not finite.")

    return min(1.0, p_value)


def main() -> None:
    """Run the Davies (2002) Monte Carlo simulations."""
    run_all(davies_test_2002, OUTPUT_DIR)


if __name__ == "__main__":
    main()
