"""
Create scatter plots of predicted vs true values per site.

Usage examples:
  python plot_pred_vs_true.py --pred_dir predictions/per_site --truth_file ground_truth.csv
  python plot_pred_vs_true.py --pred_dir predictions/per_site --truth_dir data/true_per_site

Requirements for merging:
  - Predictions must contain: year, month, day, hour, O3_predicted, NO2_predicted
  - True data must contain: year, month, day, hour, O3_target, NO2_target
  - If multiple sites, include site_id in both; otherwise process one site at a time.

Outputs:
  - Scatter plots saved to plots/per_site_pred_vs_true/site_{id}_O3.png
  - Scatter plots saved to plots/per_site_pred_vs_true/site_{id}_NO2.png
  - Printed metrics (RMSE, MAE, R², Bias) per site and target
"""

import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def load_truth_for_site(site_id, truth_file=None, truth_dir=None):
    """
    Load ground-truth data for a site.
    - If truth_file is provided, it should contain site_id (for multi-site) or be single-site.
    - If truth_dir is provided, expects files like site_{id}_*.csv.
    """
    if truth_file:
        df = pd.read_csv(truth_file)
        if "site_id" in df.columns:
            df = df[df["site_id"] == site_id].copy()
            if df.empty:
                raise FileNotFoundError(f"No rows for site {site_id} in {truth_file}")
        return df

    if truth_dir:
        pattern = os.path.join(truth_dir, f"site_{site_id}_*.csv")
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No truth files found for site {site_id} in {truth_dir}")
        return pd.read_csv(files[0])

    raise ValueError("Provide either truth_file or truth_dir")


def ensure_cols(df, required, name):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {name}: {missing}")


def merge_pred_true(pred_df, true_df):
    # Determine merge keys
    keys = ["year", "month", "day", "hour"]
    if "site_id" in pred_df.columns and "site_id" in true_df.columns:
        keys = ["site_id"] + keys
    merged = pd.merge(pred_df, true_df, on=keys, how="inner", suffixes=("", "_true"))
    if merged.empty:
        raise ValueError("Merged dataframe is empty; check that prediction and truth files align on keys.")
    return merged


def plot_scatter(merged, target, site_id, out_dir):
    sns.set_style("whitegrid")
    plt.figure(figsize=(6, 6))
    y_true = merged[f"{target}_true_std"]
    y_pred = merged[f"{target}_pred_std"]

    plt.scatter(y_true, y_pred, alpha=0.6, s=12, color="tab:blue", edgecolor="none")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2, label="Ideal")
    plt.xlabel(f"{target} Actual", fontsize=12)
    plt.ylabel(f"{target} Predicted", fontsize=12)
    plt.title(f"{target} - Predicted vs Actual (Site {site_id})", fontsize=13, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    outfile = os.path.join(out_dir, f"site_{site_id}_{target}_pred_vs_true.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    return outfile


def compute_metrics(merged, target):
    y_true = merged[f"{target}_true_std"]
    y_pred = merged[f"{target}_pred_std"]
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "Bias": bias}


def process_site(pred_file, truth_file, truth_dir, out_dir,
                pred_o3_col, pred_no2_col, true_o3_col, true_no2_col):
    print(f"\nProcessing {pred_file}")
    pred_df = pd.read_csv(pred_file)

    # Infer site_id from filename or column
    site_id = None
    if "site_id" in pred_df.columns:
        if pred_df["site_id"].nunique() != 1:
            raise ValueError(f"Prediction file {pred_file} contains multiple site_id values.")
        site_id = int(pred_df["site_id"].iloc[0])
    else:
        # Try to parse from filename: site_{id}_predictions.csv
        base = os.path.basename(pred_file)
        parts = base.replace(".csv", "").split("_")
        for p in parts:
            if p.isdigit():
                site_id = int(p)
                break
        if site_id is None:
            raise ValueError("Could not infer site_id; include site_id column or use site_{id}_*.csv naming.")

    # Required columns
    ensure_cols(pred_df, ["year", "month", "day", "hour", pred_o3_col, pred_no2_col], "predictions")

    # Load truth
    true_df = load_truth_for_site(site_id, truth_file=truth_file, truth_dir=truth_dir)
    ensure_cols(true_df, ["year", "month", "day", "hour", true_o3_col, true_no2_col], "ground truth")

    # Merge
    merged = merge_pred_true(pred_df, true_df)
    # Normalize column names for downstream steps
    merged = merged.rename(columns={
        pred_o3_col: "O3_pred_std",
        pred_no2_col: "NO2_pred_std",
        true_o3_col: "O3_true_std",
        true_no2_col: "NO2_true_std",
    })
    print(f"  Merged rows: {len(merged)}")

    # Metrics and plots
    metrics = {}
    for target in ["O3", "NO2"]:
        metrics[target] = compute_metrics(merged, target)
        outfile = plot_scatter(merged, target, site_id, out_dir)
        print(f"  {target} scatter saved: {outfile}")
        print(f"  {target} metrics: RMSE={metrics[target]['RMSE']:.3f}, "
              f"MAE={metrics[target]['MAE']:.3f}, R²={metrics[target]['R2']:.3f}, "
              f"Bias={metrics[target]['Bias']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Create scatter plots of predicted vs true values per site.")
    parser.add_argument("--pred_dir", type=str, default="predictions/per_site", help="Directory with per-site predictions.")
    parser.add_argument("--truth_file", type=str, default=None, help="Combined ground truth CSV (multi-site supported).")
    parser.add_argument("--truth_dir", type=str, default=None, help="Directory with per-site truth CSVs (e.g., site_1_*.csv).")
    parser.add_argument("--output_dir", type=str, default="plots/per_site_pred_vs_true", help="Output directory for plots.")
    parser.add_argument("--pred_o3_col", type=str, default="O3_predicted", help="Column name for predicted O3.")
    parser.add_argument("--pred_no2_col", type=str, default="NO2_predicted", help="Column name for predicted NO2.")
    parser.add_argument("--true_o3_col", type=str, default="O3_target", help="Column name for true O3.")
    parser.add_argument("--true_no2_col", type=str, default="NO2_target", help="Column name for true NO2.")
    args = parser.parse_args()

    if not args.truth_file and not args.truth_dir:
        raise ValueError("You must provide either --truth_file or --truth_dir")

    pred_files = sorted(glob.glob(os.path.join(args.pred_dir, "site_*_predictions.csv")))
    if not pred_files:
        raise FileNotFoundError(f"No prediction files found in {args.pred_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    for pred_file in pred_files:
        process_site(
            pred_file,
            args.truth_file,
            args.truth_dir,
            args.output_dir,
            pred_o3_col=args.pred_o3_col,
            pred_no2_col=args.pred_no2_col,
            true_o3_col=args.true_o3_col,
            true_no2_col=args.true_no2_col,
        )


if __name__ == "__main__":
    main()

