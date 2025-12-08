"""Add BLH + time + solar + wind engineered features to all *_with_satellite.csv.

Features added (using eps=1e-3):
- NO2_forecast_per_blh = NO2_forecast / (blh_forecast + eps)
- O3_forecast_per_blh  = O3_forecast  / (blh_forecast + eps)
- blh_delta_1h         = blh_forecast - blh_forecast.shift(1)
- blh_rel_change       = blh_forecast / (blh_forecast.shift(1) + eps)
- sin_hour, cos_hour   = cyclic hour encodings
- cosSZA               = cos(SZA_deg * pi / 180)
- solar_elevation      = 90 - SZA_deg
- sunset_flag          = 1 if solar_elevation <= 0 else 0
- wind_speed           = sqrt(u^2 + v^2)
- wind_dir_deg         = (degrees(atan2(-u, -v)) + 360) % 360
- wind_sector          = round(wind_dir_deg / 45) % 8

Files are updated in place under:
Data_SIH_2025_with_blh/Data_SIH_2025_with_blh/with_satellite
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

EPS = 1e-3

DATA_DIR = (
    Path(__file__).parent
    / "Data_SIH_2025_with_blh"
    / "Data_SIH_2025_with_blh"
    / "with_satellite"
)

# Column names (present in BLH datasets)
BLH_COL = "blh_forecast"
SZA_COL = "SZA_deg"
U_COL = "u_forecast"
V_COL = "v_forecast"

# Site lat/lon (from lat_lon_sites.txt)
SITE_LAT_LON = {
    "site_1": (28.69536, 77.18168),
    "site_2": (28.5718, 77.07125),
    "site_3": (28.58278, 77.23441),
    "site_4": (28.82286, 77.10197),
    "site_5": (28.53077, 77.27123),
    "site_6": (28.72954, 77.09601),
    "site_7": (28.71052, 77.24951),
}


def _extract_site_id(filename: str) -> str | None:
    # expects names like site_1_train_data_with_satellite.csv
    for key in SITE_LAT_LON:
        if filename.startswith(key):
            return key
    return None


def _compute_sza_simple(lat_deg: float, day_of_year: pd.Series, hour: pd.Series) -> pd.Series:
    """Compute SZA_deg using a simple declination/hour-angle formula."""
    lat_rad = np.deg2rad(lat_deg)
    decl_rad = np.deg2rad(23.45) * np.sin(np.deg2rad((360.0 / 365.0) * (284.0 + day_of_year)))
    h_rad = np.deg2rad(15.0 * (hour - 12.0))
    cos_sza = (
        np.sin(lat_rad) * np.sin(decl_rad)
        + np.cos(lat_rad) * np.cos(decl_rad) * np.cos(h_rad)
    )
    cos_sza = np.clip(cos_sza, -1.0, 1.0)
    sza_rad = np.arccos(cos_sza)
    sza_deg = np.rad2deg(sza_rad)
    return pd.Series({"SZA_deg": sza_deg, "cosSZA_pre": cos_sza})


def _add_sza(df: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    """Compute SZA_deg, cosSZA, solar_elevation, sunset_flag via EoT-based solar time.

    Hours are interpreted in Asia/Kolkata local time.
    """
    dt = pd.to_datetime(
        df[["year", "month", "day", "hour"]].astype(int, copy=False), errors="coerce"
    ).dt.tz_localize("Asia/Kolkata", nonexistent="NaT", ambiguous="NaT")

    N = dt.dt.dayofyear
    local_time = dt.dt.hour + dt.dt.minute / 60.0 + dt.dt.second / 3600.0

    # Equation of Time (minutes)
    B = 2 * np.pi * (N - 81) / 364.0
    EoT = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)

    # Solar time correction (India standard meridian 82.5E)
    std_meridian = 82.5
    lon_correction = (lon - std_meridian) / 15.0  # hours
    solar_time = (local_time + lon_correction + (EoT / 60.0)) % 24.0
    hour_angle_deg = 15.0 * (solar_time - 12.0)

    # Declination (deg)
    decl_deg = 23.45 * np.sin(2 * np.pi * (284 + N) / 365.0)

    # SZA
    lat_r = np.deg2rad(lat)
    decl_r = np.deg2rad(decl_deg)
    h_r = np.deg2rad(hour_angle_deg)
    cos_sza = np.sin(lat_r) * np.sin(decl_r) + np.cos(lat_r) * np.cos(decl_r) * np.cos(
        h_r
    )
    cos_sza = np.clip(cos_sza, -1.0, 1.0)
    sza_rad = np.arccos(cos_sza)
    sza_deg = np.rad2deg(sza_rad)

    df[SZA_COL] = sza_deg
    df["cosSZA"] = cos_sza
    df["solar_elevation"] = 90.0 - sza_deg
    df["sunset_flag"] = (df["solar_elevation"] <= 0).astype(int)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    required_base = {
        "year",
        "month",
        "day",
        "hour",
        "NO2_forecast",
        "O3_forecast",
        BLH_COL,
        U_COL,
        V_COL,
    }
    missing = required_base.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

    ordered = df.sort_values(["year", "month", "day", "hour"]).reset_index()

    # BLH-derived
    blh = ordered[BLH_COL]
    ordered["NO2_forecast_per_blh"] = ordered["NO2_forecast"] / (blh + EPS)
    ordered["O3_forecast_per_blh"] = ordered["O3_forecast"] / (blh + EPS)
    ordered["blh_delta_1h"] = blh.diff()
    ordered["blh_rel_change"] = blh / (blh.shift(1) + EPS)

    # Time cyclic
    angle = 2 * np.pi * (ordered["hour"] % 24) / 24.0
    ordered["sin_hour"] = np.sin(angle)
    ordered["cos_hour"] = np.cos(angle)

    # Solar (SZA computed upstream per file)
    if SZA_COL in ordered.columns:
        sza_rad = ordered[SZA_COL] * np.pi / 180.0
        ordered["cosSZA"] = np.cos(sza_rad)
        ordered["solar_elevation"] = 90.0 - ordered[SZA_COL]
        ordered["sunset_flag"] = (ordered["solar_elevation"] <= 0).astype(int)

    # Wind
    u = ordered[U_COL]
    v = ordered[V_COL]
    ordered["wind_speed"] = np.sqrt(u ** 2 + v ** 2)
    ordered["wind_dir_deg"] = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0
    ordered["wind_sector"] = np.round(ordered["wind_dir_deg"] / 45.0) % 8

    # Restore original row order
    return ordered.set_index("index").sort_index()


def process_file(path: Path) -> None:
    df = pd.read_csv(path)

    # Always recompute solar position to ensure columns are populated/updated.
    site_id = _extract_site_id(path.name)
    lat, lon = SITE_LAT_LON.get(site_id, (None, None))
    if lat is None or lon is None:
        raise ValueError(f"Lat/Lon not found for site {site_id} (file {path.name})")
    df = _add_sza(df, lat, lon)

    df = add_features(df)
    df.to_csv(path, index=False)
    print(f"Updated {path}")


def main() -> None:
    if not DATA_DIR.exists():
        print(f"Directory not found: {DATA_DIR}")
        return

    files = sorted(DATA_DIR.glob("*_with_satellite.csv"))
    if not files:
        print("No *_with_satellite.csv files found.")
        return

    for path in files:
        process_file(path)


if __name__ == "__main__":
    main()
