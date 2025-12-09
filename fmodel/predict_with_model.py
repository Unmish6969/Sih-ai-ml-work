"""
Complete Prediction Script for Unseen Data
Uses trained XGBoost models to predict O3 and NO2 on new unseen data.

This script handles:
1. Loading unseen input CSV
2. Filling satellite data (one value per day → all hours)
3. All feature engineering (lag, rolling, smoothing, temporal)
4. Loading models and making predictions
"""

import pandas as pd
import numpy as np
import joblib
import json
import glob
import os
from pathlib import Path


def impute_daily_value(df, column):
    """Impute missing full-day satellite values by averaging previous/next day."""
    ordered = df.sort_values(["year", "month", "day", "hour"]).reset_index()
    
    # Get first non-null value for each day
    daily_first = (
        ordered.groupby(["year", "month", "day"])[column]
        .apply(lambda s: s.dropna().iloc[0] if not s.dropna().empty else pd.NA)
    )
    
    day_keys = list(daily_first.index)
    filled_values = {}
    
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
    
    # Assign imputed value to first row of each missing day
    for key, value in filled_values.items():
        mask = (
            (ordered["year"] == key[0]) &
            (ordered["month"] == key[1]) &
            (ordered["day"] == key[2])
        )
        if mask.any():
            first_idx = ordered.index[mask][0]
            ordered.loc[first_idx, column] = value
    
    return ordered.set_index("index").sort_index()


def fill_from_forecast(df, forecast_col, sat_col, out_col):
    """Normalize forecast by daily max and scale by daily satellite value."""
    filled = df.copy()
    
    def _fill_group(group):
        max_forecast = group[forecast_col].max(skipna=True)
        sat_values = group[sat_col].dropna().unique()
        sat_value = sat_values[0] if len(sat_values) else None
        
        if sat_value is None or pd.isna(max_forecast) or max_forecast <= 0:
            group[out_col] = pd.NA
            return group
        
        normalized = group[forecast_col] / max_forecast
        group[out_col] = normalized * sat_value
        return group
    
    filled = filled.groupby(["year", "month", "day"], group_keys=False).apply(_fill_group)
    return filled


def fill_satellite_data(df):
    """Fill satellite data: one value per day → all hours."""
    # Step 1: Impute missing daily satellite values
    if 'NO2_satellite' in df.columns:
        df = impute_daily_value(df, "NO2_satellite")
    if 'HCHO_satellite' in df.columns:
        df = impute_daily_value(df, "HCHO_satellite")
    
    # Step 2: Fill per-hour satellite series from forecasts
    if 'NO2_satellite' in df.columns and 'NO2_forecast' in df.columns:
        df = fill_from_forecast(df, "NO2_forecast", "NO2_satellite", "NO2_satellite_filled")
    if 'HCHO_satellite' in df.columns and 'O3_forecast' in df.columns:
        df = fill_from_forecast(df, "O3_forecast", "HCHO_satellite", "HCHO_satellite_filled")
    
    # Step 3: Recompute ratio_satellite
    def _compute_ratio(row):
        no2 = row.get("NO2_satellite_filled")
        hcho = row.get("HCHO_satellite_filled")
        if pd.isna(no2) or pd.isna(hcho) or no2 == 0:
            return pd.NA
        return hcho / no2
    
    if 'NO2_satellite_filled' in df.columns and 'HCHO_satellite_filled' in df.columns:
        df["ratio_satellite"] = df.apply(_compute_ratio, axis=1)
    
    # Drop raw satellite columns (we use filled versions)
    df = df.drop(columns=["NO2_satellite", "HCHO_satellite"], errors="ignore")
    
    return df


def add_derived_features(df):
    """Add derived forecast features."""
    # BLH features
    if 'blh_forecast' in df.columns:
        df['NO2_forecast_per_blh'] = df['NO2_forecast'] / (df['blh_forecast'] + 1e-6)
        df['O3_forecast_per_blh'] = df['O3_forecast'] / (df['blh_forecast'] + 1e-6)
        df['blh_delta_1h'] = df['blh_forecast'].diff()
        df['blh_rel_change'] = df['blh_forecast'].pct_change()
    
    # Wind features
    if 'u_forecast' in df.columns and 'v_forecast' in df.columns:
        df['wind_speed'] = np.sqrt(df['u_forecast']**2 + df['v_forecast']**2)
        df['wind_dir_deg'] = np.degrees(np.arctan2(df['v_forecast'], df['u_forecast'])) % 360
        df['wind_sector'] = (df['wind_dir_deg'] / 45).astype(int) % 8
    
    # Hour sin/cos (if not already present)
    if 'hour' in df.columns and 'sin_hour' not in df.columns:
        df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    return df


def smooth_satellite_data(df):
    """Smooth satellite data and drop raw columns."""
    satellite_cols = ['NO2_satellite_filled', 'HCHO_satellite_filled', 'ratio_satellite']
    
    for col in satellite_cols:
        if col in df.columns:
            df[f'{col}_smooth'] = df[col].rolling(window=3, min_periods=1, center=True).mean()
            df = df.drop(col, axis=1)
    
    return df


def add_temporal_features(df):
    """Add temporal features with sin/cos encoding."""
    if all(col in df.columns for col in ['year', 'month', 'day', 'hour']):
        # Create datetime
        df['datetime'] = pd.to_datetime(
            df[['year', 'month', 'day']].astype(int).astype(str).agg('-'.join, axis=1) + 
            ' ' + df['hour'].astype(int).astype(str) + ':00:00',
            errors='coerce'
        )
        
        # Day of week
        df['dayofweek'] = df['datetime'].dt.dayofweek
        
        # Season
        df['season'] = df['datetime'].dt.month % 12 // 3 + 1
        
        # Is weekend
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        
        # Sin/Cos encoding for hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Sin/Cos encoding for month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Drop datetime
        df = df.drop('datetime', axis=1)
    
    return df


def add_lag_features(df, use_forecasts=True):
    """Add lag features. For prediction, use forecasts as proxy."""
    lag_periods = [1, 2, 3, 6, 12]
    
    if use_forecasts:
        # Use forecasts as proxy for targets (simpler but less accurate)
        for lag in lag_periods:
            if 'NO2_forecast' in df.columns:
                df[f'NO2_target_lag{lag}'] = df['NO2_forecast'].shift(lag)
            if 'O3_forecast' in df.columns:
                df[f'O3_target_lag{lag}'] = df['O3_forecast'].shift(lag)
    else:
        # For iterative prediction, this would use previous predictions
        # This is a placeholder - implement iterative prediction separately
        pass
    
    return df


def add_rolling_features(df):
    """Add rolling window features."""
    # Rolling features for NO2
    if 'NO2_forecast' in df.columns:
        for window in [3, 6, 12]:
            df[f'NO2_roll{window}'] = df['NO2_forecast'].rolling(window=window, min_periods=1).mean()
    
    # Rolling features for O3
    if 'O3_forecast' in df.columns:
        for window in [3, 6, 12]:
            df[f'O3_roll{window}'] = df['O3_forecast'].rolling(window=window, min_periods=1).mean()
    
    # Rolling features for temperature
    if 'T_forecast' in df.columns:
        df['temp_roll3'] = df['T_forecast'].rolling(window=3, min_periods=1).mean()
    
    # Rolling features for relative humidity
    if 'q_forecast' in df.columns:
        df['rh_roll3'] = df['q_forecast'].rolling(window=3, min_periods=1).mean()
    
    # Rolling features for wind speed
    if 'wind_speed' in df.columns:
        df['ws_roll3'] = df['wind_speed'].rolling(window=3, min_periods=1).mean()
    
    return df


def prepare_features_for_prediction(df, feature_cols):
    """Prepare features matching training pipeline."""
    # Select only the features used in training
    missing_features = [f for f in feature_cols if f not in df.columns]
    if missing_features:
        print(f"  Warning: Missing features: {missing_features}")
        for f in missing_features:
            df[f] = np.nan
    
    X = df[feature_cols].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Check for infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    return X


def find_latest_models(model_dir, site_id):
    """Find the latest trained models for a site."""
    site_dir = os.path.join(model_dir, f"site_{site_id}")
    
    if not os.path.exists(site_dir):
        raise FileNotFoundError(f"Model directory not found: {site_dir}")
    
    o3_models = glob.glob(os.path.join(site_dir, "O3_model_*.pkl"))
    no2_models = glob.glob(os.path.join(site_dir, "NO2_model_*.pkl"))
    metadata_files = glob.glob(os.path.join(site_dir, "metadata_*.json"))
    
    if not o3_models or not no2_models:
        raise FileNotFoundError(f"No models found for site {site_id}")
    
    o3_model_path = max(o3_models, key=os.path.getctime)
    no2_model_path = max(no2_models, key=os.path.getctime)
    metadata_path = max(metadata_files, key=os.path.getctime) if metadata_files else None
    
    return o3_model_path, no2_model_path, metadata_path


def predict_unseen_data(csv_path, site_id, model_dir, output_path=None):
    """
    Complete prediction pipeline for unseen data.
    
    Parameters:
    -----------
    csv_path : str
        Path to input CSV file (format: site_X_unseen_input_data.csv)
    site_id : int
        Site ID (1-7)
    model_dir : str
        Path to model directory (e.g., "models/xgboost_per_site")
    output_path : str, optional
        Path to save predictions. If None, saves to same directory as input.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with predictions
    """
    print(f"="*70)
    print(f"Predicting for Site {site_id}")
    print(f"="*70)
    
    # Step 1: Load data
    print(f"\nStep 1: Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.sort_values(by=['year', 'month', 'day', 'hour']).reset_index(drop=True)
    print(f"  Loaded {len(df)} samples")
    
    # Step 2: Fill satellite data
    print(f"\nStep 2: Filling satellite data (one value per day → all hours)")
    df = fill_satellite_data(df)
    print(f"  Satellite data filled")
    
    # Step 3: Add derived features
    print(f"\nStep 3: Adding derived features")
    df = add_derived_features(df)
    
    # Step 4: Feature engineering
    print(f"\nStep 4: Feature engineering")
    print(f"  4.1: Smoothing satellite data")
    df = smooth_satellite_data(df)
    
    print(f"  4.2: Adding temporal features")
    df = add_temporal_features(df)
    
    print(f"  4.3: Adding lag features (using forecasts as proxy)")
    df = add_lag_features(df, use_forecasts=True)
    
    print(f"  4.4: Adding rolling window features")
    df = add_rolling_features(df)
    
    # Step 5: Load models and metadata
    print(f"\nStep 5: Loading models")
    o3_model_path, no2_model_path, metadata_path = find_latest_models(model_dir, site_id)
    
    o3_model = joblib.load(o3_model_path)
    no2_model = joblib.load(no2_model_path)
    print(f"  Models loaded from:")
    print(f"    O3: {o3_model_path}")
    print(f"    NO2: {no2_model_path}")
    
    # Get feature list from metadata
    if metadata_path:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        feature_cols = metadata['features']
        print(f"  Features from metadata: {len(feature_cols)}")
    else:
        # Fallback: infer features
        EXCLUDE_FEATURES = ['O3_target', 'NO2_target', 'site_id']
        feature_cols = [col for col in df.columns if col not in EXCLUDE_FEATURES]
        print(f"  Warning: No metadata found, inferred {len(feature_cols)} features")
    
    # Step 6: Prepare features
    print(f"\nStep 6: Preparing features for model")
    X = prepare_features_for_prediction(df, feature_cols)
    print(f"  Features prepared: {X.shape}")
    
    # Step 7: Make predictions
    print(f"\nStep 7: Making predictions")
    o3_predictions = o3_model.predict(X)
    no2_predictions = no2_model.predict(X)
    print(f"  Predictions made for {len(o3_predictions)} samples")
    
    # Step 8: Create output
    output_df = df[['year', 'month', 'day', 'hour']].copy()
    output_df['O3_predicted'] = o3_predictions
    output_df['NO2_predicted'] = no2_predictions
    
    # Save predictions
    if output_path is None:
        output_path = csv_path.replace('.csv', '_predictions.csv')
    
    output_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    
    # Summary statistics
    print(f"\n{'='*70}")
    print(f"Prediction Summary")
    print(f"{'='*70}")
    print(f"O3 Predictions:")
    print(f"  Mean: {o3_predictions.mean():.4f}")
    print(f"  Std:  {o3_predictions.std():.4f}")
    print(f"  Min:  {o3_predictions.min():.4f}")
    print(f"  Max:  {o3_predictions.max():.4f}")
    print(f"\nNO2 Predictions:")
    print(f"  Mean: {no2_predictions.mean():.4f}")
    print(f"  Std:  {no2_predictions.std():.4f}")
    print(f"  Min:  {no2_predictions.min():.4f}")
    print(f"  Max:  {no2_predictions.max():.4f}")
    print(f"{'='*70}")
    
    return output_df


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict O3 and NO2 using trained XGBoost models')
    parser.add_argument('csv_path', type=str, help='Path to input CSV file')
    parser.add_argument('site_id', type=int, help='Site ID (1-7)')
    parser.add_argument('--model_dir', type=str, default='models/xgboost_per_site',
                        help='Path to model directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save predictions (default: input_path_predictions.csv)')
    
    args = parser.parse_args()
    
    predict_unseen_data(
        csv_path=args.csv_path,
        site_id=args.site_id,
        model_dir=args.model_dir,
        output_path=args.output
    )


if __name__ == "__main__":
    # Example usage (uncomment to use directly)
    # predict_unseen_data(
    #     csv_path="Data_SIH_2025_with_blh/Data_SIH_2025_with_blh/site_1_unseen_input_data.csv",
    #     site_id=1,
    #     model_dir="models/xgboost_per_site",
    #     output_path="predictions_site_1.csv"
    # )
    
    # Or use command line:
    main()




