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

# Configuration
DATA_DIR = "Data_SIH_2025_with_blh/Data_SIH_2025_with_blh/with_satellite"
MODEL_DIR = "models/xgboost_per_site"
OUTPUT_DIR = "predictions/xgboost_per_site"
N_SITES = 7

# Features to exclude (same as training)
EXCLUDE_FEATURES = ['O3_target', 'NO2_target', 'year', 'month', 'day', 'hour', 'site_id']


def find_latest_site_models(model_dir, site_id):
    """Find the latest trained models for a specific site."""
    site_dir = os.path.join(model_dir, f"site_{site_id}")
    
    if not os.path.exists(site_dir):
        raise FileNotFoundError(f"Model directory not found: {site_dir}")
    
    o3_models = glob.glob(os.path.join(site_dir, "O3_model_*.pkl"))
    no2_models = glob.glob(os.path.join(site_dir, "NO2_model_*.pkl"))
    
    if not o3_models or not no2_models:
        raise FileNotFoundError(f"No models found for site {site_id} in {site_dir}")
    
    # Get latest models (by timestamp in filename)
    o3_model_path = max(o3_models, key=os.path.getctime)
    no2_model_path = max(no2_models, key=os.path.getctime)
    
    # Find corresponding metadata
    metadata_files = glob.glob(os.path.join(site_dir, "metadata_*.json"))
    if metadata_files:
        metadata_path = max(metadata_files, key=os.path.getctime)
    else:
        metadata_path = None
    
    return o3_model_path, no2_model_path, metadata_path


def load_site_models(site_id, o3_model_path, no2_model_path, metadata_path):
    """Load models and their metadata for a site."""
    print(f"Loading models for site {site_id}...")
    o3_model = joblib.load(o3_model_path)
    no2_model = joblib.load(no2_model_path)
    
    if metadata_path:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        feature_cols = metadata['features']
        print(f"  Models loaded. Features: {len(feature_cols)}")
        print(f"  O3 Validation R²: {metadata['O3_metrics']['val_r2']:.4f}")
        print(f"  NO2 Validation R²: {metadata['NO2_metrics']['val_r2']:.4f}")
    else:
        # If no metadata, we'll need to infer features from the data
        feature_cols = None
        print(f"  Models loaded (no metadata found)")
    
    return o3_model, no2_model, feature_cols, metadata if metadata_path else None


def prepare_features(df, feature_cols):
    """Prepare features for prediction (same preprocessing as training)."""
    if feature_cols is None:
        # Infer features from data (exclude targets and time identifiers)
        feature_cols = [col for col in df.columns if col not in EXCLUDE_FEATURES]
    
    # Check if all required features exist
    missing_features = [f for f in feature_cols if f not in df.columns]
    if missing_features:
        print(f"  Warning: Missing features: {missing_features}")
        # Add missing features with NaN
        for f in missing_features:
            df[f] = np.nan
    
    X = df[feature_cols].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Check for infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    return X


def predict_site(site_id, data_dir, model_dir, output_dir):
    """Make predictions for a single site using its trained models."""
    unseen_file = os.path.join(data_dir, f"site_{site_id}_unseen_input_data_with_satellite.csv")
    
    if not os.path.exists(unseen_file):
        print(f"Warning: {unseen_file} not found. Skipping site {site_id}.")
        return None
    
    print(f"\nProcessing site {site_id}...")
    df = pd.read_csv(unseen_file)
    print(f"  Loaded {len(df)} samples")
    
    # Find and load models
    try:
        o3_model_path, no2_model_path, metadata_path = find_latest_site_models(model_dir, site_id)
        o3_model, no2_model, feature_cols, metadata = load_site_models(
            site_id, o3_model_path, no2_model_path, metadata_path
        )
    except Exception as e:
        print(f"  Error loading models for site {site_id}: {e}")
        return None
    
    # Prepare features
    X = prepare_features(df, feature_cols)
    
    # Make predictions
    o3_predictions = o3_model.predict(X)
    no2_predictions = no2_model.predict(X)
    
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

