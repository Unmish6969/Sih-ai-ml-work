🌫️ BLH + Time/Solar/Wind Feature Engineering — Final Reference

This document is the definitive spec for adding Boundary Layer Height (BLH) plus time, solar, and wind features to the datasets and models.

## What is BLH?

- BLH (Boundary Layer Height) is the height of the lowest atmospheric mixing layer.
- Low BLH traps air near the surface → higher surface pollution (especially NO₂).
- High BLH allows vertical mixing → disperses pollution → lower surface concentrations.

## Why BLH matters for pollution modeling

- NO₂: roughly inversely proportional to BLH (low BLH → high surface NO₂).
- O₃: higher BLH helps mix ozone down from aloft (noting night titration effects).

## Key relations

- NO₂ surface ~ NO₂ column / BLH
- O₃ surface ~ O₃ column / BLH

## BLH engineered features (must-have)

Use a small epsilon to avoid divide-by-zero (e.g., `eps = 1e-3`):

- Physical dilution for NO₂: `NO2_forecast_per_blh = NO2_forecast / (blh + eps)`
- Physical dilution for O₃: `O3_forecast_per_blh = O3_forecast / (blh + eps)`
- Hourly change of BLH: `blh_delta_1h = blh - blh.shift(1)`
- Relative hourly BLH change: `blh_rel_change = blh / (blh.shift(1) + eps)`

## Time features (cyclic encoding)

- `angle = 2π * hour / 24`
- `sin_hour = sin(angle)`
- `cos_hour = cos(angle)`

## Solar & photochemical features

- Compute SZA in-script via pvlib using site lat/lon and UTC time:
  - `SZA_deg = pvlib.solarposition.get_solarposition(..., latitude, longitude)["apparent_zenith"]`
- `SZA_rad = SZA_deg * π / 180`
- `cosSZA = cos(SZA_rad)`
- `solar_elevation = 90 - SZA_deg`
- `sunset_flag = 1 if solar_elevation <= 0 else 0`
- (Optional if UV available) `photochem_index = cosSZA * cams_hourly_aluvd`

## Wind & transport features

- `wind_speed = sqrt(u^2 + v^2)`
- `wind_dir_deg = (degrees(atan2(-u, -v)) + 360) % 360`
- `wind_sector = round(wind_dir_deg / 45) % 8`

## Implementation notes

- Add these to every site dataset before training.
- Ensure `blh` is aligned hourly with forecasts (ERA5/CAMS).
- Consider clipping extreme `blh` to reasonable physical ranges before ratios.
