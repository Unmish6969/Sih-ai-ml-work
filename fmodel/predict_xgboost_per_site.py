"""
XGBoost Prediction Script - Per Site Models
Loads trained per-site XGBoost models and makes predictions on unseen data.
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
import glob
from datetime import datetime

# Reuse the feature-engineering pipeline from predict_with_model to keep
# training/prediction parity (forecasts as proxies for lags/rolls).
from predict_with_model import (
    fill_satellite_data,
    add_derived_features,
    smooth_satellite_data,
    add_temporal_features,
)

# Configuration
DATA_DIR = "Data_SIH_2025_with_blh/Data_SIH_2025_with_blh/with_satellite"
MODEL_DIR = "models/xgboost_per_site"
OUTPUT_DIR = "predictions/xgboost_per_site"
N_SITES = 7

# Features to exclude (targets only; feature_cols from metadata drives selection)
EXCLUDE_FEATURES = ['O3_target', 'NO2_target']


def find_latest_site_models(model_dir, site_id):
    """Find the latest trained models and imputer for a specific site."""
    site_dir = os.path.join(model_dir, f"site_{site_id}")
    
    if not os.path.exists(site_dir):
        raise FileNotFoundError(f"Model directory not found: {site_dir}")
    
    o3_models = glob.glob(os.path.join(site_dir, "O3_model_*.pkl"))
    no2_models = glob.glob(os.path.join(site_dir, "NO2_model_*.pkl"))
    imputer_files = glob.glob(os.path.join(site_dir, "imputer_*.pkl"))
    
    if not o3_models or not no2_models:
        raise FileNotFoundError(f"No models found for site {site_id} in {site_dir}")
    
    # Get latest models (by timestamp in filename)
    o3_model_path = max(o3_models, key=os.path.getctime)
    no2_model_path = max(no2_models, key=os.path.getctime)
    imputer_path = max(imputer_files, key=os.path.getctime) if imputer_files else None
    
    # Find corresponding metadata
    metadata_files = glob.glob(os.path.join(site_dir, "metadata_*.json"))
    metadata_path = max(metadata_files, key=os.path.getctime) if metadata_files else None
    
    return o3_model_path, no2_model_path, metadata_path, imputer_path


def load_site_models(site_id, o3_model_path, no2_model_path, metadata_path, imputer_path):
    """Load models, metadata, and imputer for a site."""
    print(f"Loading models for site {site_id}...")
    o3_model = joblib.load(o3_model_path)
    no2_model = joblib.load(no2_model_path)
    imputer = joblib.load(imputer_path) if imputer_path else None
    
    metadata = None
    feature_cols = None
    if metadata_path:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        feature_cols = metadata['features']
        print(f"  Models loaded. Features: {len(feature_cols)}")
        print(f"  O3 Validation R²: {metadata['O3_metrics']['val_r2']:.4f}")
        print(f"  NO2 Validation R²: {metadata['NO2_metrics']['val_r2']:.4f}")
    else:
        print(f"  Models loaded (no metadata found)")
    
    if imputer is None:
        print("  ⚠️ No imputer found; will fall back to median fill at inference.")
    
    return o3_model, no2_model, feature_cols, metadata, imputer


def _ensure_feature_columns(df: pd.DataFrame, feature_cols):
    """Add any missing feature columns as NaN to preserve ordering."""
    missing_features = [f for f in feature_cols if f not in df.columns]
    if missing_features:
        print(f"  Warning: Missing features: {missing_features}")
        for f in missing_features:
            df[f] = np.nan
    return df


def _init_lag_roll_columns(df: pd.DataFrame):
    """Add empty lag/rolling columns expected by the model (lag1,2,3,6,12 + roll3,6,12)."""
    lag_periods = [1, 2, 3, 6, 12]
    for gas in ["O3", "NO2"]:
        for lag in lag_periods:
            df[f"{gas}_target_lag{lag}"] = np.nan
        for window in [3, 6, 12]:
            df[f"{gas}_roll{window}"] = np.nan
    return df


def _apply_imputer(row_df: pd.DataFrame, imputer):
    """Apply the saved imputer to a single-row dataframe."""
    if imputer is not None:
        return pd.DataFrame(imputer.transform(row_df), columns=row_df.columns, index=row_df.index)
    # Fallback: median fill per row_df (should rarely happen)
    return row_df.fillna(row_df.median())


def add_forecast_roll3(df: pd.DataFrame) -> pd.DataFrame:
    """Recreate forecast-based roll3 features expected by some trained models."""
    df = df.copy()
    if 'T_forecast' in df.columns:
        df['temp_roll3'] = df['T_forecast'].shift(1).rolling(window=3, min_periods=1).mean()
    if 'q_forecast' in df.columns:
        df['rh_roll3'] = df['q_forecast'].shift(1).rolling(window=3, min_periods=1).mean()
    if 'wind_speed' in df.columns:
        df['ws_roll3'] = df['wind_speed'].shift(1).rolling(window=3, min_periods=1).mean()
    return df


def add_ratio_satellite_smooth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recreate ratio_satellite_smooth expected by older models.
    If inputs missing, creates the column filled with NaN to keep feature order.
    """
    df = df.copy()
    col_name = "ratio_satellite_smooth"
    if 'HCHO_satellite_filled' in df.columns and 'NO2_satellite_filled' in df.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = df['HCHO_satellite_filled'] / df['NO2_satellite_filled']
        df[col_name] = ratio.ewm(span=6, adjust=False).mean().shift(1)
    else:
        df[col_name] = np.nan
    return df


def recursive_predict(
    df: pd.DataFrame,
    feature_cols,
    imputer,
    o3_model,
    no2_model,
    o3_hist_init=None,
    no2_hist_init=None,
):
    """
    Sequentially predict and build lag/rolling features from prior predictions
    to match the training-time autoregressive feature set.
    """
    df = df.copy()
    df = _init_lag_roll_columns(df)
    df = _ensure_feature_columns(df, feature_cols)

    # Deques to hold past predictions for lags/rolls
    o3_hist = list(o3_hist_init) if o3_hist_init is not None else []
    no2_hist = list(no2_hist_init) if no2_hist_init is not None else []
    lag_periods = [1, 2, 3, 6, 12]
    roll_windows = [3, 6, 12]

    o3_preds = []
    no2_preds = []

    for idx in range(len(df)):
        row = df.iloc[idx].copy()

        # Populate lags from prediction history
        for gas, hist in [("O3", o3_hist), ("NO2", no2_hist)]:
            for lag in lag_periods:
                col = f"{gas}_target_lag{lag}"
                if len(hist) >= lag:
                    row[col] = hist[-lag]
                else:
                    row[col] = np.nan
            for window in roll_windows:
                col = f"{gas}_roll{window}"
                if len(hist) >= 1:
                    window_vals = hist[-window:] if len(hist) >= window else hist
                    row[col] = float(np.mean(window_vals))
                else:
                    row[col] = np.nan

        # Build single-row DF with full feature set in correct order
        row_df = pd.DataFrame([row])[feature_cols]
        row_df = row_df.replace({pd.NA: np.nan})
        row_df = row_df.replace([np.inf, -np.inf], np.nan)
        row_df = _apply_imputer(row_df, imputer)

        # Predict
        o3_pred = float(o3_model.predict(row_df)[0])
        no2_pred = float(no2_model.predict(row_df)[0])

        o3_preds.append(o3_pred)
        no2_preds.append(no2_pred)

        # Update history
        o3_hist.append(o3_pred)
        no2_hist.append(no2_pred)

    return o3_preds, no2_preds


def predict_site(site_id, data_dir, model_dir, output_dir):
    """Make predictions for a single site using its trained models."""
    unseen_file = os.path.join(data_dir, f"site_{site_id}_unseen_input_data_with_satellite.csv")
    train_file = os.path.join(data_dir, f"site_{site_id}_train_data_with_satellite.csv")
    
    if not os.path.exists(unseen_file):
        print(f"Warning: {unseen_file} not found. Skipping site {site_id}.")
        return None
    
    print(f"\nProcessing site {site_id}...")
    df = pd.read_csv(unseen_file)
    df = df.sort_values(by=["year", "month", "day", "hour"]).reset_index(drop=True)
    print(f"  Loaded {len(df)} samples")
    
    # Find and load models
    try:
        o3_model_path, no2_model_path, metadata_path, imputer_path = find_latest_site_models(model_dir, site_id)
        o3_model, no2_model, feature_cols, metadata, imputer = load_site_models(
            site_id, o3_model_path, no2_model_path, metadata_path, imputer_path
        )
    except Exception as e:
        print(f"  Error loading models for site {site_id}: {e}")
        return None
    
    # Seed histories with last 12 true targets from training data (teacher forcing init)
    o3_hist_init, no2_hist_init = [], []
    try:
        train_df = pd.read_csv(train_file)
        if all(col in train_df.columns for col in ["year", "month", "day", "hour"]):
            train_df = train_df.sort_values(by=["year", "month", "day", "hour"]).reset_index(drop=True)
        if "O3_target" in train_df.columns:
            o3_hist_init = train_df["O3_target"].dropna().tail(12).tolist()
        if "NO2_target" in train_df.columns:
            no2_hist_init = train_df["NO2_target"].dropna().tail(12).tolist()
    except Exception as e:
        print(f"  Warning: could not seed history from train data: {e}")
        o3_hist_init, no2_hist_init = [], []
    
    # ---- Feature engineering to mirror training (base features) ----
    df = fill_satellite_data(df)
    df = add_derived_features(df)
    df = add_forecast_roll3(df)  # recreate temp/rh/ws roll3 expected by models
    df = add_ratio_satellite_smooth(df)  # recreate ratio_satellite_smooth or placeholder
    df = smooth_satellite_data(df)
    df = add_temporal_features(df)

    # Sequential/recursive prediction to build lag/rolling features from prior predictions
    o3_predictions, no2_predictions = recursive_predict(
        df,
        feature_cols,
        imputer,
        o3_model,
        no2_model,
        o3_hist_init=o3_hist_init,
        no2_hist_init=no2_hist_init,
    )
    
    # Create output dataframe
    output_df = df[['year', 'month', 'day', 'hour']].copy()
    output_df['O3_predicted'] = o3_predictions
    output_df['NO2_predicted'] = no2_predictions
    
    # Save predictions
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"site_{site_id}_predictions.csv")
    output_df.to_csv(output_file, index=False)
    print(f"  Predictions saved to: {output_file}")
    
    return output_df


def main():
    """Main prediction function."""
    print("="*70)
    print("XGBoost Model Prediction - Per Site Models (O3 and NO2)")
    print("="*70)
    
    # Make predictions for all sites
    all_predictions = []
    
    for site_id in range(1, N_SITES + 1):
        try:
            predictions = predict_site(site_id, DATA_DIR, MODEL_DIR, OUTPUT_DIR)
            if predictions is not None:
                predictions['site_id'] = site_id
                all_predictions.append(predictions)
        except Exception as e:
            print(f"Error processing site {site_id}: {e}")
            continue
    
    # Combine all predictions
    if all_predictions:
        combined_predictions = pd.concat(all_predictions, ignore_index=True)
        combined_output_file = os.path.join(OUTPUT_DIR, "all_sites_predictions.csv")
        combined_predictions.to_csv(combined_output_file, index=False)
        print(f"\nCombined predictions saved to: {combined_output_file}")
        
        # Summary statistics
        print(f"\n{'='*70}")
        print("Prediction Summary")
        print(f"{'='*70}")
        print(f"Total predictions: {len(combined_predictions)}")
        print(f"\nO3 Predictions:")
        print(f"  Mean: {combined_predictions['O3_predicted'].mean():.4f}")
        print(f"  Std:  {combined_predictions['O3_predicted'].std():.4f}")
        print(f"  Min:  {combined_predictions['O3_predicted'].min():.4f}")
        print(f"  Max:  {combined_predictions['O3_predicted'].max():.4f}")
        print(f"\nNO2 Predictions:")
        print(f"  Mean: {combined_predictions['NO2_predicted'].mean():.4f}")
        print(f"  Std:  {combined_predictions['NO2_predicted'].std():.4f}")
        print(f"  Min:  {combined_predictions['NO2_predicted'].min():.4f}")
        print(f"  Max:  {combined_predictions['NO2_predicted'].max():.4f}")
        
        # Per-site summary
        print(f"\nPer-Site Summary:")
        print(f"{'Site':<6} {'Samples':<10} {'O3 Mean':<12} {'O3 Std':<12} {'NO2 Mean':<12} {'NO2 Std':<12}")
        print("-" * 70)
        for site_id in range(1, N_SITES + 1):
            site_preds = combined_predictions[combined_predictions['site_id'] == site_id]
            if len(site_preds) > 0:
                print(f"Site {site_id:<3} {len(site_preds):<10} "
                      f"{site_preds['O3_predicted'].mean():<12.4f} {site_preds['O3_predicted'].std():<12.4f} "
                      f"{site_preds['NO2_predicted'].mean():<12.4f} {site_preds['NO2_predicted'].std():<12.4f}")
    
    print("="*70)


if __name__ == "__main__":
    main()

