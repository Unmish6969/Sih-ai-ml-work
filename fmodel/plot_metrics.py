"""
Plot bar charts for R² and RMSE (train vs val) per site and target.
Creates two images:
  - metrics_r2_bar.png
  - metrics_rmse_bar.png
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "plots"
OUTPUT_R2 = os.path.join(OUTPUT_DIR, "metrics_r2_bar.png")
OUTPUT_RMSE = os.path.join(OUTPUT_DIR, "metrics_rmse_bar.png")


def get_latest_metrics_csv():
    """Find the latest all_sites_metrics CSV in models/xgboost_per_site."""
    pattern = os.path.join("models", "xgboost_per_site", "all_sites_metrics_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No metrics files found matching {pattern}")
    return max(files, key=os.path.getctime)


def plot_metric(df, value_col, title, outfile):
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=df,
        x="Site",
        y=value_col,
        hue="Split",
        palette="tab10",
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Site", fontsize=12)
    ax.set_ylabel(value_col, fontsize=12)
    # Annotate bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.3f}" if "R²" in value_col else f"{height:.2f}",
                    (p.get_x() + p.get_width() / 2., height),
                    ha="center", va="bottom", fontsize=8, rotation=0)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outfile}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metrics_csv = get_latest_metrics_csv()
    print(f"Using metrics file: {metrics_csv}")
    df = pd.read_csv(metrics_csv)

    # Build separate dataframes for R² and RMSE with train/val
    r2_rows = []
    rmse_rows = []
    for _, row in df.iterrows():
        site = int(row["Site"])
        target = row["Target"]
        # R2
        r2_rows.append({"Site": site, "Target": target, "Split": "Train", "Value": row["Train_R²"]})
        r2_rows.append({"Site": site, "Target": target, "Split": "Val", "Value": row["R²"]})
        # RMSE
        rmse_rows.append({"Site": site, "Target": target, "Split": "Train", "Value": row["Train_RMSE"]})
        rmse_rows.append({"Site": site, "Target": target, "Split": "Val", "Value": row["RMSE"]})

    r2_df = pd.DataFrame(r2_rows)
    rmse_df = pd.DataFrame(rmse_rows)

    # Separate plots for O3 and NO2 for clarity
    for tgt in ["O3", "NO2"]:
        plot_metric(
            r2_df[r2_df["Target"] == tgt],
            value_col="Value",
            title=f"{tgt} - R² (Train vs Val)",
            outfile=OUTPUT_R2.replace(".png", f"_{tgt}.png"),
        )
        plot_metric(
            rmse_df[rmse_df["Target"] == tgt],
            value_col="Value",
            title=f"{tgt} - RMSE (Train vs Val)",
            outfile=OUTPUT_RMSE.replace(".png", f"_{tgt}.png"),
        )


if __name__ == "__main__":
    main()

