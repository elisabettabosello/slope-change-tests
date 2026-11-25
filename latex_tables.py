"""Convert Monte Carlo simulation results for the Davies and Muggeo tests from CSV files into LaTeX tables saved as .txt files."""

from pathlib import Path
import pandas as pd

from simulation_setup import (
    ALPHAS,
    BREAKPOINT_QUANTILES,
    DISTRIBUTIONS,
    POWER_LAMBDAS,
    SAMPLE_SIZES,
)

DAVIES_DIR = Path(__file__).resolve().parent / "davies_results"
MUGGEO_DIR = Path(__file__).resolve().parent / "muggeo_results"
OUTPUT_DIR = Path(__file__).resolve().parent / "latex_tables"

NAMES = {
    "unif": r"$\mathrm{Unif}(0,1)$",
    "beta12": r"$\mathrm{Beta}(1,2)$",
    "beta22": r"$\mathrm{Beta}(2,2)$",
    "beta25gap": r"$\mathrm{Beta}(2,5)$ with gap",
}

NAMES_TITLE = {
    "unif": r"\text{Uniform}(0,1)",
    "beta12": r"\mathrm{Beta}(1,2)",
    "beta22": r"\mathrm{Beta}(2,2)",
    "beta25gap": r"\mathrm{Beta}(2,5)\text{ with gap}",
}


def value(
    df: pd.DataFrame,
    filters: dict[str, str | int | float],
    column: str,
) -> str:
    """Extract exactly one simulation result.

    Parameters:
        df : pd.DataFrame
            DataFrame containing the simulation results.
        filters : dict[str, str | int | float]
            Dictionary specifying the conditions used to select a row.
        column : str
            Name of the column from which to extract the result.

    Returns:
        str
            Selected value formatted to four decimal places.
    """

    mask = pd.Series(True, index=df.index)
    for key, expected in filters.items():
        if key not in df:
            raise KeyError(f"Missing column: {key}")
        if isinstance(expected, (float, int)) and key not in ("n",):
            mask &= (pd.to_numeric(df[key]) - expected).abs() < 1e-9
        else:
            mask &= df[key] == expected
    selected = df.loc[mask, column]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {filters}, found {len(selected)}")
    return f"{selected.iloc[0]:.4f}"


def size_table(davies: pd.DataFrame, muggeo: pd.DataFrame) -> str:
    """Generate a LaTeX table of empirical size for all distributions,
    sample sizes, and nominal significance levels.

        Parameters:
            davies : pd.DataFrame
                Simulation results for Davies' test.
            muggeo : pd.DataFrame
                Simulation results for Muggeo's score-based test.

        Returns:
            str
                LaTeX code for a table containing empirical rejection
                rates for all distributions, sample sizes, and
                nominal significance levels.
    """

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Empirical type I error rates of the Davies test (D) and of the Muggeo score test (S) with $M_{\mathrm{MC}}=10{,}000$ replicates.}",
        r"\label{tab:size}",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        " & & "
        + " & ".join(
            rf"\multicolumn{{2}}{{c}}{{$\alpha={alpha:.2f}$}}" for alpha in ALPHAS
        )
        + r" \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        r"Distribution & $n$ & D & S & D & S & D & S \\",
        r"\midrule",
    ]
    for i, distribution in enumerate(DISTRIBUTIONS):
        for j, n in enumerate(SAMPLE_SIZES):
            label = (
                rf"\multirow{{{len(SAMPLE_SIZES)}}}{{*}}{{{NAMES[distribution]}}}"
                if j == 0
                else ""
            )
            cells = []
            for alpha in ALPHAS:
                criteria = {"distribution": distribution, "n": n, "alpha": alpha}
                cells += [
                    value(davies, criteria, "empirical_size"),
                    value(muggeo, criteria, "empirical_size"),
                ]
            lines.append(" & ".join([label, str(n), *cells]) + r" \\")
        if i != len(DISTRIBUTIONS) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def power_table(
    davies: pd.DataFrame,
    muggeo: pd.DataFrame,
    distribution: str,
    alpha: float,
) -> str:
    """Generate a LaTeX table of empirical power for a given
    covariate distribution and nominal significance level.

        Parameters:
            davies : pd.DataFrame
                Simulation results for Davies' test.
            muggeo : pd.DataFrame
                Simulation results for Muggeo's score-based test.
            distribution : str
                Covariate distribution considered in the table.
            alpha : float
                Nominal significance level considered in the table.

        Returns:
            str
                LaTeX code for a table containing empirical power
                for all sample sizes, signal-to-noise ratios,
                and breakpoint quantiles.
    """

    column = f"power_alpha_{alpha:.2f}"
    quantiles = BREAKPOINT_QUANTILES
    distribution_name = NAMES_TITLE[distribution]
    caption = (
        r"Empirical powers "
        rf"($X\sim{distribution_name}$, "
        r"$M_{\mathrm{MC}}=5{,}000$ replicates) "
        r"of the Davies test (D) and of the Muggeo score test (S), "
        rf"for nominal level $\alpha = {alpha:.2f}$."
    )
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{tab:power_{distribution}_{int(round(alpha * 100)):02d}}}",
        "\\begin{tabular}{ll" + "cc" * len(quantiles) + "}",
        r"\toprule",
        rf"\multicolumn{{2}}{{c}}{{}} & \multicolumn{{{2 * len(quantiles)}}}{{c}}{{$\psi$ (quantile)}} \\",
        "\\cmidrule(lr){3-" + str(2 + 2 * len(quantiles)) + "}",
        " & & "
        + " & ".join(rf"\multicolumn{{2}}{{c}}{{{q:g}}}" for q in quantiles)
        + r" \\",
        "".join(
            rf"\cmidrule(lr){{{3 + 2*i}-{4 + 2*i}}}" for i in range(len(quantiles))
        ),
        r"$n$ & $\lambda$ & " + " & ".join(["D", "S"] * len(quantiles)) + r" \\",
        r"\midrule",
    ]
    for i, n in enumerate(SAMPLE_SIZES):
        for j, signal_ratio in enumerate(POWER_LAMBDAS):
            label = rf"\multirow{{{len(POWER_LAMBDAS)}}}{{*}}{{{n}}}" if j == 0 else ""
            cells = []
            for q in quantiles:
                criteria = {"n": n, "lambda": signal_ratio, "psi_quantile": q}
                cells += [
                    value(davies, criteria, column),
                    value(muggeo, criteria, column),
                ]
            lines.append(" & ".join([label, f"{signal_ratio:g}", *cells]) + r" \\")
        if i != len(SAMPLE_SIZES) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def generate_latex_tables(
    davies_dir: Path = DAVIES_DIR,
    muggeo_dir: Path = MUGGEO_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """
    Read the saved simulation results and export one empirical size
    table and twelve empirical power tables as .txt files.

        Parameters:
            davies_dir : Path
                Directory containing the Davies simulation results.
            muggeo_dir : Path
                Directory containing the Muggeo simulation results.
            output_dir : Path
                Directory where the LaTeX tables are saved.

        Returns:
            None
    """

    size_d = pd.read_csv(davies_dir / "size.csv")
    size_s = pd.read_csv(muggeo_dir / "size.csv")
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "SIZE_ALL_ALL.txt"
    path.write_text(size_table(size_d, size_s), encoding="utf-8")
    print(f"Created: {path}")

    for distribution in DISTRIBUTIONS:
        power_d = pd.read_csv(davies_dir / f"power_{distribution}.csv")
        power_s = pd.read_csv(muggeo_dir / f"power_{distribution}.csv")
        for alpha in ALPHAS:
            path = output_dir / f"POWER_{distribution}_ALPHA_{alpha:.2f}.txt"
            path.write_text(
                power_table(power_d, power_s, distribution, alpha), encoding="utf-8"
            )
            print(f"Created: {path}")


if __name__ == "__main__":
    generate_latex_tables()
