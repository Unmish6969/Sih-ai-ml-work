"""Generate missing-value report for filled satellite columns.

Looks for CSVs produced by fill_no2_satellite (files ending with
_with_satellite.csv) inside Data_SIH_2025/with_satellite. Summarizes per-file
rows and missing counts for:
- NO2_satellite_filled
- HCHO_satellite_filled
- ratio_satellite
Also prints grand totals.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_DIR = (
    Path(__file__).parent
    / "Data_SIH_2025_with_blh"
    / "Data_SIH_2025_with_blh"
    / "with_satellite"
)
REPORT_FILE = DATA_DIR / "satellite_filled_report.txt"


def summarize_file(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    required = [
        "NO2_satellite_filled",
        "HCHO_satellite_filled",
        "ratio_satellite",
    ]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns {missing_cols} in {path.name}")

    total = len(df)
    return {
        "file": path.name,
        "rows": total,
        "no2_missing": df["NO2_satellite_filled"].isna().sum(),
        "hcho_missing": df["HCHO_satellite_filled"].isna().sum(),
        "ratio_missing": df["ratio_satellite"].isna().sum(),
    }


def main() -> None:
    if not DATA_DIR.exists():
        print(f"No directory found: {DATA_DIR}")
        return

    files = sorted(DATA_DIR.glob("*_with_satellite.csv"))
    if not files:
        print(f"No *_with_satellite.csv files found in {DATA_DIR}")
        return

    summaries = [summarize_file(p) for p in files]

    lines = [
        "File, rows, no2_missing, hcho_missing, ratio_missing, "
        "no2_missing_pct, hcho_missing_pct, ratio_missing_pct"
    ]
    total_rows = 0
    total_no2 = 0
    total_hcho = 0
    total_ratio = 0
    for s in summaries:
        no2_pct = (s["no2_missing"] / s["rows"] * 100) if s["rows"] else 0.0
        hcho_pct = (s["hcho_missing"] / s["rows"] * 100) if s["rows"] else 0.0
        ratio_pct = (s["ratio_missing"] / s["rows"] * 100) if s["rows"] else 0.0
        lines.append(
            f"{s['file']}, {s['rows']}, {s['no2_missing']}, "
            f"{s['hcho_missing']}, {s['ratio_missing']}, "
            f"{no2_pct:.2f}%, {hcho_pct:.2f}%, {ratio_pct:.2f}%"
        )
        total_rows += s["rows"]
        total_no2 += s["no2_missing"]
        total_hcho += s["hcho_missing"]
        total_ratio += s["ratio_missing"]

    lines.append(
        f"TOTAL, {total_rows}, {total_no2}, {total_hcho}, {total_ratio}, "
        f"{(total_no2 / total_rows * 100 if total_rows else 0):.2f}%, "
        f"{(total_hcho / total_rows * 100 if total_rows else 0):.2f}%, "
        f"{(total_ratio / total_rows * 100 if total_rows else 0):.2f}%"
    )

    REPORT_FILE.parent.mkdir(exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nReport saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
