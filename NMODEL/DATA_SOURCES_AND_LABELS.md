# Data Sources and Target Labels

## 📁 CSV Files Used for Training

The model uses **three CSV files per site** located in the `F/` directory:

### For Site 3 (as in your example):
1. **`F/site3_train.csv`** - Training data (30,432 samples)
2. **`F/site3_val.csv`** - Validation data (4,320 samples)  
3. **`F/site3_test.csv`** - Test data (1,897 samples)

### Available Sites:
- Site 3: `site3_train.csv`, `site3_val.csv`, `site3_test.csv`
- Site 5: `site5_train.csv`, `site5_val.csv`, `site5_test.csv`
- Site 6: `site6_train.csv`, `site6_val.csv`, `site6_test.csv`
- Site 7: `site7_train.csv`, `site7_val.csv`, `site7_test.csv`

## 🎯 Target Labels (Y Variables)

The model can predict **two different air quality targets**:

### 1. **NO2 (Nitrogen Dioxide)**
- **Target Column**: `NO2_target`
- **Units**: Likely µg/m³ or ppb (parts per billion)
- **Description**: Ground-level NO2 concentration measurements

### 2. **O3 (Ozone)**
- **Target Column**: `O3_target`
- **Units**: Likely µg/m³ or ppb
- **Description**: Ground-level O3 concentration measurements

## 📊 How Targets Are Selected

When you run:
```bash
python ensemble_main.py --site 3 --target NO2
```
- Uses `NO2_target` column as the target variable (y)

When you run:
```bash
python ensemble_main.py --site 3 --target O3
```
- Uses `O3_target` column as the target variable (y)

When you run:
```bash
python ensemble_main.py --site 3 --both-targets
```
- Trains **two separate models**:
  1. One model for `NO2_target`
  2. One model for `O3_target`

## 📋 Data Structure

Each CSV file contains:

### Target Columns (Y labels):
- `NO2_target` - Nitrogen Dioxide measurements (target for NO2 model)
- `O3_target` - Ozone measurements (target for O3 model)

### Feature Columns (X - used as inputs):
- **Temporal features**: `datetime`, `Year`, `sin_hour`, `cos_hour`, `day_of_week`, `is_weekend`, `day_of_year`
- **Solar features**: `SZA_deg`, `cosSZA`, `solar_elevation`, `sunset_flag`
- **Meteorological forecasts**: `O3_forecast`, `NO2_forecast`, `T_forecast`, `q_forecast`, `u_forecast`, `v_forecast`, `w_forecast`
- **ERA5 reanalysis**: `era5_t2m`, `era5_d2m`, `era5_tcc`, `era5_u10`, `era5_v10`, `era5_blh`
- **Wind features**: `wind_speed`, `wind_dir_deg`, `wind_sector`
- **Satellite data**: `NO2_satellite_daily`, `HCHO_satellite_daily`, `ratio_satellite_daily`, `NO2_satellite_delta`
- **CAMS model outputs**: `cams_hourly_*` (multiple columns for various pollutants)
- **Lag features**: `NO2_target_lag1`, `NO2_target_lag24`, `O3_target_lag1`
- **Other features**: `photochem_index`, `RH`, `cams_hourly_aluvd`, `cams_hourly_aluvp`, etc.
- **Metadata**: `_original_row_marker` (used for sample weights)

### Total Features:
- **61 features** total (as shown in your training output)
- **1 static feature**: `Year`
- **7 known future features**: Forecast columns (e.g., `NO2_forecast`, `O3_forecast`, etc.)
- **53 observed features**: All other input features

## 🔍 Data Loading Process

1. **Load CSV files** (`utils.load_data()`):
   - Reads `F/site{site_id}_train.csv`
   - Reads `F/site{site_id}_val.csv`
   - Reads `F/site{site_id}_test.csv`

2. **Prepare features** (`utils.prepare_features()`):
   - Extracts target: `y = df[f'{target}_target']` (e.g., `NO2_target` or `O3_target`)
   - Extracts features: All columns except `datetime`, `NO2_target`, `O3_target`, `_original_row_marker`
   - Creates sample weights: 1.0 for original rows, 0.5 for imputed rows
   - **Handles NaN values** in both features and target (NEW FIX)

3. **Feature groups** (`utils.get_feature_groups()`):
   - **Static**: `['Year']` - Constant per site
   - **Known future**: Columns with 'forecast' in name - Future meteorological forecasts
   - **Observed**: Everything else - Historical measurements and features

## 📝 Summary

| Aspect | Details |
|--------|---------|
| **Training CSV** | `F/site{site_id}_train.csv` |
| **Validation CSV** | `F/site{site_id}_val.csv` |
| **Test CSV** | `F/site{site_id}_test.csv` |
| **NO2 Target** | `NO2_target` column |
| **O3 Target** | `O3_target` column |
| **Total Features** | 61 features |
| **Sample Weights** | 1.0 (original) / 0.5 (imputed) |

## ⚠️ Important Notes

1. **Both targets are in the same CSV files** - The model selects which one to use based on the `--target` argument
2. **Target columns are excluded from features** - They are only used as the y variable
3. **Lag features exist** - `NO2_target_lag1`, `NO2_target_lag24`, `O3_target_lag1` are used as features (not targets)
4. **NaN handling** - The code now automatically handles NaN values in target columns (fills with median or drops rows if >10% NaN)


