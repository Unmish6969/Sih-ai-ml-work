"""Merge solar columns from all_sites_sza_added into with_satellite CSVs.

Source directory:
  Data_SIH_2025_with_blh/all_sites_sza_added
Target directory:
  Data_SIH_2025_with_blh/Data_SIH_2025_with_blh/with_satellite

Columns copied (overwriting if present):
  cosSZA, solar_elevation, sunset_flag,
  solar_declination_deg, hour_angle_deg, solar_time_hr, EoT_min

Run:
  python merge_sza_columns.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).parent / "Data_SIH_2025_with_blh" / "all_sites_sza_added"
TGT_DIR = (
    Path(__file__).parent
    / "Data_SIH_2025_with_blh"
    / "Data_SIH_2025_with_blh"
    / "with_satellite"
)

COLUMNS = [
    "cosSZA",
    "solar_elevation",
    "sunset_flag",
    "solar_declination_deg",
    "hour_angle_deg",
    "solar_time_hr",
    "EoT_min",
]


def merge_file(src: Path, tgt: Path) -> None:
    src_df = pd.read_csv(src)
    tgt_df = pd.read_csv(tgt)

    missing = [c for c in COLUMNS if c not in src_df.columns]
    if missing:
        raise ValueError(f"{src.name} missing columns: {', '.join(missing)}")

    if len(src_df) != len(tgt_df):
        raise ValueError(f"Row mismatch {src.name} ({len(src_df)}) vs {tgt.name} ({len(tgt_df)})")

    tgt_df[COLUMNS] = src_df[COLUMNS].values
    tgt_df.to_csv(tgt, index=False)
    print(f"Updated {tgt}")


def main() -> None:
    if not SRC_DIR.exists() or not TGT_DIR.exists():
        print("Source or target directory missing.")
        return

    src_files = {p.name: p for p in SRC_DIR.glob("*.csv")}
    tgt_files = {p.name: p for p in TGT_DIR.glob("*.csv")}

    common = set(src_files).intersection(tgt_files)
    if not common:
        print("No matching filenames between source and target.")
        return

    for name in sorted(common):
        merge_file(src_files[name], tgt_files[name])


if __name__ == "__main__":
    main()

