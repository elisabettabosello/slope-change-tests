"""Analyze the empirical size and power of the Davies (2002) and Muggeo (2016) tests."""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

DAVIES_DIR = BASE_DIR / "davies_results"
MUGGEO_DIR = BASE_DIR / "muggeo_results"
OUTPUT_DIR = BASE_DIR / "simulation_analysis"
FIGURES_DIR = OUTPUT_DIR / "figures"

TOP_NS = (10, 20, 30)
DISTRIBUTION_LABELS = ("Uniform(0,1)", "Beta(1,2)", "Beta(2,2)", "Beta(2,5) with gap")
TOP_COLORS = ("#173F7A", "#3979B7", "#A3C7E8")

from simulation_setup import ALPHAS, DISTRIBUTIONS


def analyze_size(
    davies_dir: Path,
    muggeo_dir: Path,
    output_path: Path,
) -> None:
    """Compare empirical sizes and their deviations from the nominal significance levels.

    Parameters:
        davies_dir : Path
            Directory containing the size.csv file for the Davies test.
        muggeo_dir : Path
            Directory containing the size.csv file for the Muggeo test.
        output_path : Path
            Path to the TXT file where the report is saved.

    Returns:
        None
            The results are written to a TXT file.
    """

    keys = ["distribution", "n", "alpha"]
    davies = pd.read_csv(davies_dir / "size.csv").rename(
        columns={"empirical_size": "size_D"}
    )
    muggeo = pd.read_csv(muggeo_dir / "size.csv").rename(
        columns={"empirical_size": "size_M"}
    )
    merged = davies.merge(muggeo, on=keys)

    merged["difference"] = merged["size_M"] - merged["size_D"]
    merged["abs_difference"] = merged["difference"].abs()
    merged["deviation_D"] = (merged["size_D"] - merged["alpha"]).abs()
    merged["deviation_M"] = (merged["size_M"] - merged["alpha"]).abs()

    lines = [
        "EMPIRICAL SIZE ANALYSIS",
        "=" * 60,
        "",
    ]
    for alpha in ALPHAS:
        selected = merged[merged["alpha"] == alpha]
        differences = selected["abs_difference"]

        lines.extend(
            [
                f"ALPHA = {alpha:.2f} ({len(selected)} configurations)",
                "-" * 60,
                f"Absolute difference range: [{differences.min():.4f}, {differences.max():.4f}]",
                f"Mean absolute difference: {differences.mean():.4f}",
                f"Median absolute difference: {differences.median():.4f}",
                f"Mean absolute deviation from alpha (Davies): {selected['deviation_D'].mean():.4f}",
                f"Mean absolute deviation from alpha (Muggeo): {selected['deviation_M'].mean():.4f}",
                f"Muggeo size > Davies size: {(selected['difference'] > 0).sum()}",
                f"Davies size > Muggeo size: {(selected['difference'] < 0).sum()}",
                f"Equal empirical sizes: {(selected['difference'] == 0).sum()}",
                f"Muggeo closer to alpha: {(selected['deviation_M'] < selected['deviation_D']).sum()}",
                f"Davies closer to alpha: {(selected['deviation_D'] < selected['deviation_M']).sum()}",
                "",
            ]
        )
    lines.extend(
        [
            "ALL CONFIGURATIONS AND SIGNIFICANCE LEVELS",
            "-" * 60,
            f"Muggeo size > Davies size: {(merged['difference'] > 0).sum()}/{len(merged)}",
            f"Davies size > Muggeo size: {(merged['difference'] < 0).sum()}/{len(merged)}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_rankings(
    davies_dir: Path,
    muggeo_dir: Path,
) -> dict[float, pd.DataFrame]:
    """Read power results and rank scenarios by absolute power difference.

    Parameters:
        davies_dir : Path
            Directory containing the Davies test power results.
        muggeo_dir : Path
            Directory containing the Muggeo test power results.

    Returns:
        rankings: dict[float, pd.DataFrame]
            Dictionary mapping each nominal significance level to a DataFrame
            containing the simulation configurations and the corresponding
            empirical powers of both tests, ranked by decreasing absolute
            power difference.
    """

    def read_results(directory: Path) -> pd.DataFrame:
        """
        Read and combine empirical power results.

        Parameters:
            directory : Path
                Directory containing the power CSV files.

        Returns:
            pd.DataFrame
                Combined simulation results in long format.
        """
        frames = []

        for distribution in DISTRIBUTIONS:
            df = pd.read_csv(directory / f"power_{distribution}.csv")

            # Convert the three power columns into long format.
            df = df.melt(
                id_vars=["n", "lambda", "psi_quantile"],
                value_vars=[f"power_alpha_{alpha:.2f}" for alpha in ALPHAS],
                var_name="alpha",
                value_name="power",
            )

            # Recover the nominal significance level from the column name.
            df["alpha"] = (
                df["alpha"].str.replace("power_alpha_", "", regex=False).astype(float)
            )

            df["distribution"] = distribution
            frames.append(df)

        return pd.concat(frames, ignore_index=True)

    # Read the Davies and Muggeo power results.
    davies = read_results(davies_dir)
    muggeo = read_results(muggeo_dir)

    # Match configurations and compute power differences.
    merged = davies.merge(
        muggeo,
        on=["distribution", "n", "lambda", "psi_quantile", "alpha"],
        suffixes=("_D", "_S"),
    )
    merged["difference"] = merged["power_S"] - merged["power_D"]
    merged["abs_difference"] = merged["difference"].abs()

    # Rank scenarios separately for each significance level.
    rankings = {}

    for alpha in ALPHAS:
        ranked = merged[merged["alpha"] == alpha].copy()
        rankings[alpha] = ranked.sort_values(
            "abs_difference", ascending=False
        ).reset_index(drop=True)

    return rankings


def plot_distribution_frequencies(
    rankings: dict[float, pd.DataFrame],
    figures_dir: Path,
) -> None:
    """
    Plot the frequency of each covariate distribution among the top 10, 20,
    and 30 scenarios with the largest absolute power differences, for each
    nominal significance level.

    Parameters:
        rankings : dict[float, pd.DataFrame]
            Dictionary mapping each nominal significance level to a DataFrame
            containing scenarios ranked by decreasing absolute power difference.
        figures_dir : Path
            Directory where the resulting figure is saved.

    Returns:
        None
            The function saves the figure as a PNG file and does not return a value.
    """

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    y = np.arange(len(DISTRIBUTIONS))
    height = 0.23
    for ax, alpha in zip(axes, ALPHAS):
        if ax is not axes[0]:
            ax.tick_params(axis="y", left=False)

        df = rankings[alpha]
        for j, (top_n, color) in enumerate(zip(TOP_NS, TOP_COLORS)):
            percentages = (
                df.head(top_n)["distribution"]
                .value_counts()
                .reindex(DISTRIBUTIONS, fill_value=0)
                / top_n
                * 100
            )
            ax.barh(
                y + (j - 1) * height,
                percentages,
                height=height,
                color=color,
                label=f"Top {top_n}",
            )
        ax.set_title(rf"$\alpha={alpha:.2f}$")
        ax.set_yticks(y, DISTRIBUTION_LABELS)
        ax.set_xlim(0, 50)
        ax.set_xlabel("Frequency (%)")
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Covariate distribution")
    axes[-1].legend(frameon=False, loc="upper right")
    fig.suptitle("Covariate distributions among the largest power differences")
    fig.tight_layout()
    fig.savefig(figures_dir / "distribution_topN.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_characteristic_heatmaps(
    rankings: dict[float, pd.DataFrame],
    figures_dir: Path,
) -> None:
    """
    Generate heatmaps showing the frequency of selected scenario
    characteristics (n=500, lambda in {1,2}, and q in {0.50,0.75})
    among the top 10, 20, and 30 scenarios with the largest absolute
    power differences, for each nominal significance level.

    Parameters:
        rankings : dict[float, pd.DataFrame]
            Dictionary mapping each nominal significance level to a DataFrame
            containing scenarios ranked by decreasing absolute power difference.
        figures_dir : Path
            Directory where the resulting figure is saved.

    Returns:
        None
            The function saves the figure as a PNG file and does not return a value.
    """

    labels = (r"$n=500$", r"$\lambda\in\{1,2\}$", r"$q\in\{0.50,0.75\}$")
    cmap = LinearSegmentedColormap.from_list("custom_blues", TOP_COLORS[::-1])
    fig, axes = plt.subplots(
        1, 3, figsize=(12, 3.5), sharey=True, constrained_layout=True
    )
    for ax, alpha in zip(axes, ALPHAS):
        if ax is not axes[0]:
            ax.tick_params(axis="y", left=False)

        matrix = np.empty((3, len(TOP_NS)))
        for j, top_n in enumerate(TOP_NS):
            top = rankings[alpha].head(top_n)
            matrix[:, j] = (
                100 * (top["n"] == 500).mean(),
                100 * top["lambda"].isin((1, 2)).mean(),
                100 * top["psi_quantile"].isin((0.50, 0.75)).mean(),
            )
        im = ax.imshow(matrix, cmap=cmap, vmin=50, vmax=100, aspect="auto")
        ax.set_title(rf"$\alpha={alpha:.2f}$")
        ax.set_xticks(range(len(TOP_NS)), [f"Top {n}" for n in TOP_NS])
        ax.set_yticks(range(len(labels)), labels)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=15,
                )
    fig.colorbar(im, ax=axes, label="Frequency (%)", shrink=0.85, pad=0.06)
    fig.suptitle("Recurrent characteristics among the largest power differences")
    fig.savefig(
        figures_dir / "characteristics_heatmap.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


def write_power_report(
    rankings: dict[float, pd.DataFrame],
    output_path: Path,
) -> None:
    """
    Save a TXT report summarizing absolute power differences
    for the top 10, 20, and 30 scenarios,
    for each nominal significance level and across all scenarios.

    Parameters:
        rankings : dict[float, pd.DataFrame]
            Dictionary mapping each nominal significance level to a DataFrame
            containing scenarios ranked by decreasing absolute power difference.
        output_path : Path
            Path of the TXT file where the report is saved.

    Returns:
        None
            The function saves the report as a TXT file and does not return a value.
    """

    lines = [
        "EMPIRICAL POWER ANALYSIS",
        "=" * 60,
        "",
    ]
    for alpha in ALPHAS:
        df = rankings[alpha]
        lines.extend((f"ALPHA = {alpha:.2f}", "-" * 60))
        for top_n in TOP_NS:
            top = df.head(top_n)
            differences = top["abs_difference"]
            lines.extend(
                (
                    f"Top {top_n}:",
                    f"  Range: [{differences.min():.4f}, {differences.max():.4f}]",
                    f"  Mean: {differences.mean():.4f}",
                    f"  Median: {differences.median():.4f}",
                    f"  Muggeo > Davies: {(top['difference'] > 0).sum()}/{top_n}",
                    f"  Davies > Muggeo: {(top['difference'] < 0).sum()}/{top_n}",
                    f"  Equal power: {(top['difference'] == 0).sum()}/{top_n}",
                    "",
                )
            )
        differences = df["abs_difference"]
        lines.extend(
            (
                f"All {len(df)} scenarios:",
                f"  Range: [{differences.min():.4f}, {differences.max():.4f}]",
                f"  Mean: {differences.mean():.4f}",
                f"  Median: {differences.median():.4f}",
                f"  Muggeo > Davies: {(df['difference'] > 0).sum()}/{len(df)}",
                f"  Davies > Muggeo: {(df['difference'] < 0).sum()}/{len(df)}",
                f"  Equal power: {(df['difference'] == 0).sum()}/{len(df)}",
                "",
                "=" * 60,
                "",
            )
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Analyze empirical size and power results and save the outputs."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Empirical size analysis.
    analyze_size(DAVIES_DIR, MUGGEO_DIR, OUTPUT_DIR / "size_summary.txt")

    # Empirical power analysis.
    rankings = build_rankings(DAVIES_DIR, MUGGEO_DIR)

    plot_distribution_frequencies(rankings, FIGURES_DIR)
    plot_characteristic_heatmaps(rankings, FIGURES_DIR)

    write_power_report(rankings, OUTPUT_DIR / "power_summary.txt")

    print(f"Figures saved in: {FIGURES_DIR}")
    print(f"Size report: {OUTPUT_DIR / 'size_summary.txt'}")
    print(f"Power report: {OUTPUT_DIR / 'power_summary.txt'}")


if __name__ == "__main__":
    main()
