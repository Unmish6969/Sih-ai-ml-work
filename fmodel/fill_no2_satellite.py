"""Fill satellite columns (NO2 & HCHO) per day and derive ratio_satellite.

For every (year, month, day):
1) Impute a missing full-day satellite value by averaging previous/next day
   values (only when both neighbors exist). The imputed value is placed on the
   first hour row of that day.
2) For NO2: normalize NO2_forecast by the day's max and scale by the day's
   NO2_satellite (filled) to get NO2_satellite_filled.
3) For HCHO: normalize O3_forecast by the day's max and scale by the day's
   HCHO_satellite (filled) to get HCHO_satellite_filled.
4) ratio_satellite is recomputed as HCHO_satellite_filled / NO2_satellite_filled
   (set to NA when denominator is missing or zero).

Outputs are written to Data_SIH_2025_with_blh/Data_SIH_2025_with_blh/with_satellite/<file>_with_satellite.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).parent / "Data_SIH_2025_with_blh" / "Data_SIH_2025_with_blh"
OUTPUT_DIR = DATA_DIR / "with_satellite"


def impute_daily_value(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Impute missing full-day satellite values by averaging previous/next day."""
    sort_cols = ["year", "month", "day", "hour"]
    ordered = df.sort_values(sort_cols).reset_index()

    daily_first = (
        ordered.groupby(["year", "month", "day"])[column]
        .apply(lambda s: s.dropna().iloc[0] if not s.dropna().empty else pd.NA)
    )

    day_keys = list(daily_first.index)
    filled_values: dict[tuple, float] = {}

    for i, key in enumerate(day_keys):
        if pd.notna(daily_first.loc[key]):
            continue

        prev_val = None
        next_val = None

        # Look backward
        for j in range(i - 1, -1, -1):
            candidate = daily_first.iloc[j]
            if pd.notna(candidate):
                prev_val = float(candidate)
                break

        # Look forward
        for j in range(i + 1, len(day_keys)):
            candidate = daily_first.iloc[j]
            if pd.notna(candidate):
                next_val = float(candidate)
                break

        if prev_val is not None and next_val is not None:
            filled_values[key] = (prev_val + next_val) / 2.0

    # Assign imputed value to the first row of each missing day
    for key, value in filled_values.items():
        mask = (
            (ordered["year"] == key[0])
            & (ordered["month"] == key[1])
            & (ordered["day"] == key[2])
        )
        if not mask.any():
            continue
        first_idx = ordered.index[mask][0]
        ordered.loc[first_idx, column] = value

    # Restore original row order
    return ordered.set_index("index").sort_index()


def fill_from_forecast(
    df: pd.DataFrame, forecast_col: str, sat_col: str, out_col: str
) -> pd.DataFrame:
    """Normalize forecast by daily max and scale by daily satellite value."""
    required_cols = {"year", "month", "day", forecast_col, sat_col}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input: {', '.join(sorted(missing))}")

    filled = df.copy()

    def _fill_group(group: pd.DataFrame) -> pd.DataFrame:
        max_forecast = group[forecast_col].max(skipna=True)
        sat_values = group[sat_col].dropna().unique()
        sat_value = sat_values[0] if len(sat_values) else None

        # If we cannot derive a scaling factor, leave as NA.
        if sat_value is None or pd.isna(max_forecast) or max_forecast <= 0:
            group[out_col] = pd.NA
            return group

        normalized = group[forecast_col] / max_forecast
        group[out_col] = normalized * sat_value
        return group

    filled = filled.groupby(["year", "month", "day"], group_keys=False).apply(
        _fill_group
    )
    return filled


def process_file(path: Path) -> Path:
    """Process one CSV file and write to OUTPUT_DIR."""
    df = pd.read_csv(path)

    # Impute daily satellite values (in-place) for NO2 and HCHO.
    df = impute_daily_value(df, "NO2_satellite")
    df = impute_daily_value(df, "HCHO_satellite")

    # Fill per-hour satellite series from forecasts.
    df = fill_from_forecast(
        df, forecast_col="NO2_forecast", sat_col="NO2_satellite", out_col="NO2_satellite_filled"
    )
    df = fill_from_forecast(
        df, forecast_col="O3_forecast", sat_col="HCHO_satellite", out_col="HCHO_satellite_filled"
    )

    # Recompute ratio using filled values; guard against zero / missing.
    def _compute_ratio(row):
        no2 = row.get("NO2_satellite_filled")
        hcho = row.get("HCHO_satellite_filled")
        if pd.isna(no2) or pd.isna(hcho) or no2 == 0:
            return pd.NA
        return hcho / no2

    df["ratio_satellite"] = df.apply(_compute_ratio, axis=1)

    # Drop raw satellite columns per user request; keep filled versions.
    df = df.drop(columns=["NO2_satellite", "HCHO_satellite"], errors="ignore")

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{path.stem}_with_satellite.csv"
    df.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    csv_files = [
        p
        for p in DATA_DIR.glob("*.csv")
        if not p.stem.endswith("_with_satellite")
    ]

    if not csv_files:
        print("No CSV files found to process.")
        return

    written = []
    skipped = []
    for path in csv_files:
        try:
            out_path = process_file(path)
            written.append(out_path)
        except ValueError as exc:
            skipped.append((path.name, str(exc)))

    for path in written:
        print(f"Wrote {path}")

    if skipped:
        print("Skipped files:")
        for name, reason in skipped:
            print(f" - {name}: {reason}")


if __name__ == "__main__":
    main()
