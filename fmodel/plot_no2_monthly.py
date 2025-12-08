"""
Compute monthly mean NO2_satellite values from site_1_train_data.csv and
plot 12-point lines (months Jan-Dec) for each year found in the file.

Usage:
    python plot_no2_monthly.py

Output:
    Saves plot to `no2_monthly_by_year.png` in the current directory and
    also displays the plot window.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = Path("Data_SIH_2025/site_1_train_data.csv")
OUTPUT_PATH = Path("no2_monthly_by_year.png")


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load CSV and ensure NO2_satellite is numeric."""
    df = pd.read_csv(csv_path)
    df["NO2_satellite"] = pd.to_numeric(df["NO2_satellite"], errors="coerce")
    return df


def monthly_means(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with mean NO2_satellite per (year, month).
    Months are 1-12; missing months will appear as NaN.
    """
    grouped = (
        df.groupby(["year", "month"])["NO2_satellite"]
        .mean()
        .reset_index()
        .sort_values(["year", "month"])
    )
    # Ensure month is integer for nicer ticks
    grouped["month"] = grouped["month"].astype(int)
    return grouped


def plot_monthly(grouped: pd.DataFrame) -> None:
    """Plot 12-point series per year."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for year, data in grouped.groupby("year"):
        data = data.sort_values("month").set_index("month")
        # Reindex to all 12 months for consistent x-axis; NaNs will break the line
        data = data.reindex(range(1, 13))
        ax.plot(
            data.index,
            data["NO2_satellite"],
            marker="o",
            label=str(int(year)),
        )

    ax.set_xlabel("Month")
    ax.set_ylabel("Mean NO2_satellite")
    ax.set_title("Monthly mean NO2_satellite per year")
    ax.set_xticks(range(1, 13))
    ax.legend(title="Year", fontsize="small")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved plot to {OUTPUT_PATH.resolve()}")
    plt.show()


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"CSV not found at {DATA_PATH}")

    df = load_data(DATA_PATH)
    grouped = monthly_means(df)
    plot_monthly(grouped)


if __name__ == "__main__":
    main()

