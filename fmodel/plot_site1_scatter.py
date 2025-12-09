"""
Plot predicted vs. observed values for each site using XGBoost per-site outputs.

Usage:
    python plot_site1_scatter.py
It reads, per site:
    - predictions/xgboost_per_site/site_<id>_predictions.csv (O3_predicted, NO2_predicted)
    - unseen_output_blh/unseen_output_blh/site_<id>_unseen_output.csv (O3_target, NO2_target)
and saves, per site & gas:
    - plots/per_site_pred_vs_true/site_<id>_<gas>_scatter.png
    - plots/per_site_pred_vs_true/site_<id>_<gas>_time.png
"""
from pathlib import Path
import math

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


PRED_DIR = Path("predictions/xgboost_per_site")
TARGET_DIR = Path("unseen_output_blh/unseen_output_blh")
OUT_DIR = Path("plots/per_site_pred_vs_true")
SITE_IDS = [1, 2, 3, 4, 5, 6, 7]


def _build_datetime_from_parts(df: pd.DataFrame) -> pd.Series:
    """Create a pandas datetime from year/month/day/hour float columns."""
    return pd.to_datetime(
        dict(
            year=df["year"].astype(int),
            month=df["month"].astype(int),
            day=df["day"].astype(int),
            hour=df["hour"].astype(int),
        ),
        errors="coerce",
    )


def load_data(site_id: int) -> pd.DataFrame:
    """Load prediction and target CSVs for a site and align rows by datetime."""
    pred_path = PRED_DIR / f"site_{site_id}_predictions.csv"
    target_path = TARGET_DIR / f"site_{site_id}_unseen_output.csv"

    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"Missing target file: {target_path}")

    pred = pd.read_csv(pred_path)
    target = pd.read_csv(target_path)

    # Build/parse datetime columns.
    if "datetime" in pred.columns:
        pred["datetime"] = pd.to_datetime(pred["datetime"], errors="coerce")
    else:
        pred["datetime"] = _build_datetime_from_parts(pred)
    target["datetime"] = _build_datetime_from_parts(target)

    # Inner-join on datetime to ensure both have the same timestamps.
    merged = pred.merge(
        target[["datetime", "O3_target", "NO2_target"]],
        on="datetime",
        how="inner",
        suffixes=("_pred", "_target"),
    )

    if merged.empty:
        raise ValueError("No overlapping datetimes between predictions and targets.")

    # Sort by time to make plotting consistent.
    merged = merged.sort_values("datetime").reset_index(drop=True)
    return merged


def make_scatter(df: pd.DataFrame, col_pred: str, col_true: str, ax: plt.Axes) -> None:
    """Create a single scatter plot and annotate metrics."""
    y_true = df[col_true]
    y_pred = df[col_pred]

    r2 = r2_score(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))

    ax.scatter(y_true, y_pred, s=10, alpha=0.6, edgecolors="none")
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    ax.plot(lims, lims, "k--", linewidth=1, label="1:1 line")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f"{col_true} (observed)")
    ax.set_ylabel(f"{col_pred} (predicted)")
    ax.set_title(f"{col_pred.replace('_predicted', '')}: R²={r2:.3f}, RMSE={rmse:.2f}")
    ax.legend()


def make_time_plot(
    df: pd.DataFrame, col_pred: str, col_true: str, label: str, ax: plt.Axes
) -> None:
    """Plot predicted and actual values over time on the same axis."""
    ax.plot(df["datetime"], df[col_true], label=f"{label} actual", linewidth=1.2)
    ax.plot(df["datetime"], df[col_pred], label=f"{label} predicted", linewidth=1.2)
    ax.set_ylabel(label)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)


def plot_site(site_id: int) -> None:
    df = load_data(site_id)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for gas in ["O3", "NO2"]:
        col_pred = f"{gas}_predicted"
        col_true = f"{gas}_target"

        # Scatter
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        make_scatter(df, col_pred, col_true, ax)
        fig.suptitle(f"Site {site_id} {gas}: predictions vs observations")
        fig.tight_layout()
        out_scatter = OUT_DIR / f"site_{site_id}_{gas}_scatter.png"
        fig.savefig(out_scatter)
        plt.close(fig)

        # Time series
        tfig, tax = plt.subplots(figsize=(10, 4), dpi=150)
        make_time_plot(df, col_pred, col_true, gas, tax)
        tfig.suptitle(f"Site {site_id} {gas}: over time (aligned)")
        tfig.tight_layout()
        out_time = OUT_DIR / f"site_{site_id}_{gas}_time.png"
        tfig.savefig(out_time)
        plt.close(tfig)

        print(f"Site {site_id} {gas}: saved {out_scatter} and {out_time}")


def main() -> None:
    for site_id in SITE_IDS:
        try:
            plot_site(site_id)
        except FileNotFoundError as e:
            print(f"Skipping site {site_id}: {e}")
        except Exception as e:
            print(f"Error plotting site {site_id}: {e}")


if __name__ == "__main__":
    main()

