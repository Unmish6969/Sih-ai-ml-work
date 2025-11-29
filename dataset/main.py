"""
Monthly-chunk downloader + GRIB->NetCDF conversion + merge
For CAMS EAC4 (2019–2024, monthly).
"""

import os
import time
import csv
import cdsapi
import calendar
import xarray as xr
from datetime import date

# ----------------- CONFIG -----------------
TXT_PATH = "lat_lon.txt"
OUTPUT_DIR = "cams_downloads_monthly"
os.makedirs(OUTPUT_DIR, exist_ok=True)

START_YEAR = 2019
END_YEAR = 2024
# 1-based index of the site to process (set to 4 to only download site 4)
SITE_TO_PROCESS = 1

DELTA_DEGREES = 0.05
MAX_RETRIES = 5
INITIAL_BACKOFF = 5

CLEAN_MONTHLY_AFTER_MERGE = False
# ------------------------------------------

def read_coords_from_txt(path):
    coords = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        sample = f.read(2048); f.seek(0)
        delim = "\t" if "\t" in sample else ","
        reader = csv.reader(f, delimiter=delim)
        for row in reader:
            if not row:
                continue
            try:
                lat = float(row[1].strip())
                lon = float(row[2].strip())
                coords.append((lat, lon))
            except:
                continue
    return coords

def build_area_box(lat, lon, delta=DELTA_DEGREES):
    return [
        round(lat + delta, 6),
        round(lon - delta, 6),
        round(lat - delta, 6),
        round(lon + delta, 6)
    ]

def client_create():
    print("Using ~/.cdsapirc credentials")
    return cdsapi.Client()

def retrieve_with_retries(client, dataset, request, outfile):
    attempt = 0
    backoff = INITIAL_BACKOFF
    while attempt < MAX_RETRIES:
        try:
            client.retrieve(dataset, request, outfile)
            return True
        except Exception as e:
            attempt += 1
            print(f"[attempt {attempt}/{MAX_RETRIES}] {e}")
            if attempt >= MAX_RETRIES:
                print("FAILED permanently for:", outfile)
                return False
            print(f"Retrying in {backoff} sec...")
            time.sleep(backoff)
            backoff *= 2

def download_month_for_site(client, dataset, site_idx, lat, lon, year, month, out_dir):
    last_day = calendar.monthrange(year, month)[1]
    date_str = f"{year}-{month:02d}-01/{year}-{month:02d}-{last_day:02d}"

    area_box = build_area_box(lat, lon)
    request = {
        "model_level": ["60"],
        "date": date_str,
        "time": [
            "00:00","03:00","06:00","09:00",
            "12:00","15:00","18:00","21:00"
        ],
        "data_format": "grib",
        "variable": [
            "black_carbon_aerosol_optical_depth_550nm",
            "particulate_matter_1um",
            "particulate_matter_2.5um",
            "particulate_matter_10um",
            "surface_pressure",
            "total_aerosol_optical_depth_550nm",
            "total_column_carbon_monoxide",
            "total_column_formaldehyde",
            "total_column_nitrogen_dioxide",
            "total_column_nitrogen_monoxide",
            "total_column_ozone",
            "total_column_sulphur_dioxide",
            "carbon_monoxide",
            "formaldehyde",
            "nitrogen_dioxide",
            "nitrogen_monoxide",
            "ozone",
            "sulphur_dioxide",
            "uv_visible_albedo_for_diffuse_radiation",
            "uv_visible_albedo_for_direct_radiation"
        ],
        "area": area_box
    }

    out_grib = os.path.join(
        out_dir,
        f"cams_site_{site_idx+1}_{year}-{month:02d}.grib"
    )

    print(f"Downloading site {site_idx+1}, {year}-{month:02d} → {out_grib}")
    ok = retrieve_with_retries(client, dataset, request, out_grib)
    return out_grib if ok else None

def grib_to_netcdf(grib_path, nc_path):
    try:
        ds = xr.open_dataset(grib_path, engine="cfgrib")
        encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
        ds.to_netcdf(nc_path, encoding=encoding)
        ds.close()
        return True
    except Exception as e:
        print("GRIB→NC fail:", grib_path, e)
        return False

def merge_monthly_netcdfs(nc_files, out_merged):
    try:
        ds = xr.open_mfdataset(
            nc_files,
            combine="by_coords",
            engine="netcdf4"
        )
        ds.to_netcdf(out_merged)
        ds.close()
        return True
    except Exception as e:
        print("Merge failed:", e)
        return False

def process_site(client, dataset, site_idx, lat, lon, out_dir):
    monthly_gribs = []

    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            g = download_month_for_site(client, dataset, site_idx, lat, lon, year, month, out_dir)
            if g:
                monthly_gribs.append(g)
            time.sleep(2)

    monthly_ncs = []
    for g in monthly_gribs:
        nc = g.replace(".grib", ".nc")
        if grib_to_netcdf(g, nc):
            monthly_ncs.append(nc)

    final_out = os.path.join(
        out_dir,
        f"cams_site_{site_idx+1}_{START_YEAR}-{END_YEAR}.nc"
    )

    if merge_monthly_netcdfs(monthly_ncs, final_out):
        print("Merged:", final_out)

        if CLEAN_MONTHLY_AFTER_MERGE:
            for p in monthly_gribs + monthly_ncs:
                try: os.remove(p)
                except: pass
    else:
        print(f"!! Merge failed for site {site_idx+1}")

def main():
    coords = read_coords_from_txt(TXT_PATH)

    if not coords:
        print("No coordinates in TXT.")
        return

    print(f"Found {len(coords)} sites.")
    if SITE_TO_PROCESS < 1 or SITE_TO_PROCESS > len(coords):
        print(f"Configured site {SITE_TO_PROCESS} does not exist in the coordinate list.")
        return

    target_lat, target_lon = coords[SITE_TO_PROCESS - 1]
    print(f"\nProcessing only site {SITE_TO_PROCESS} ({target_lat},{target_lon})")

    client = client_create()
    dataset = "cams-global-reanalysis-eac4"

    process_site(
        client,
        dataset,
        SITE_TO_PROCESS - 1,
        target_lat,
        target_lon,
        OUTPUT_DIR
    )

    print("\nALL DONE.")

if __name__ == "__main__":
    main()
