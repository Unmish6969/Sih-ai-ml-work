"""
XGBoost Prediction Script
Loads trained XGBoost models and makes predictions on unseen data.
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
MODEL_DIR = "models/xgboost"
OUTPUT_DIR = "predictions/xgboost"
N_SITES = 7

# Features to exclude (same as training)
EXCLUDE_FEATURES = ['O3_target', 'NO2_target', 'year', 'month', 'day', 'hour']


def find_latest_models(model_dir):
    """Find the latest trained models."""
    o3_models = glob.glob(os.path.join(model_dir, "O3_model_*.pkl"))
    no2_models = glob.glob(os.path.join(model_dir, "NO2_model_*.pkl"))
    
    if not o3_models or not no2_models:
        raise FileNotFoundError(f"No models found in {model_dir}. Please train models first.")
    
    # Get latest models (by timestamp in filename)
    o3_model_path = max(o3_models, key=os.path.getctime)
    no2_model_path = max(no2_models, key=os.path.getctime)
    
    # Load corresponding metadata to get feature list
    o3_metadata_path = o3_model_path.replace('.pkl', '_metadata.json').replace('O3_model_', 'O3_metadata_')
    no2_metadata_path = no2_model_path.replace('.pkl', '_metadata.json').replace('NO2_model_', 'NO2_metadata_')
    
    return o3_model_path, no2_model_path, o3_metadata_path, no2_metadata_path


def load_models(o3_model_path, no2_model_path, o3_metadata_path, no2_metadata_path):
    """Load models and their metadata."""
    print(f"Loading O3 model from: {o3_model_path}")
    o3_model = joblib.load(o3_model_path)
    
    print(f"Loading NO2 model from: {no2_model_path}")
    no2_model = joblib.load(no2_model_path)
    
    with open(o3_metadata_path, 'r') as f:
        o3_metadata = json.load(f)
    
    with open(no2_metadata_path, 'r') as f:
        no2_metadata = json.load(f)
    
    # Get feature columns (should be same for both)
    feature_cols = o3_metadata['features']
    
    print(f"Models loaded. Features: {len(feature_cols)}")
    
    return o3_model, no2_model, feature_cols, o3_metadata, no2_metadata


def prepare_features(df, feature_cols):
    """Prepare features for prediction (same preprocessing as training)."""
    # Check if all required features exist
    missing_features = [f for f in feature_cols if f not in df.columns]
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        # Add missing features with NaN
        for f in missing_features:
            df[f] = np.nan
    
    X = df[feature_cols].copy()
    
    # Handle missing values (use median from training - in production, should load from metadata)
    X = X.fillna(X.median())
    
    # Check for infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    return X


def predict_site(site_id, o3_model, no2_model, feature_cols, data_dir, output_dir):
    """Make predictions for a single site."""
    unseen_file = os.path.join(data_dir, f"site_{site_id}_unseen_input_data_with_satellite.csv")
    
    if not os.path.exists(unseen_file):
        print(f"Warning: {unseen_file} not found. Skipping site {site_id}.")
        return None
    
    print(f"\nProcessing site {site_id}...")
    df = pd.read_csv(unseen_file)
    print(f"  Loaded {len(df)} samples")
    
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
    print("="*60)
    print("XGBoost Model Prediction for O3 and NO2")
    print("="*60)
    
    # Find latest models
    print(f"\nLooking for models in: {MODEL_DIR}")
    o3_model_path, no2_model_path, o3_metadata_path, no2_metadata_path = find_latest_models(MODEL_DIR)
    
    # Load models
    o3_model, no2_model, feature_cols, o3_metadata, no2_metadata = load_models(
        o3_model_path, no2_model_path, o3_metadata_path, no2_metadata_path
    )
    
    print(f"\nModel Info:")
    print(f"  O3 Model - Timestamp: {o3_metadata['timestamp']}")
    print(f"  NO2 Model - Timestamp: {no2_metadata['timestamp']}")
    print(f"  O3 Validation R²: {o3_metadata['metrics']['val_r2']:.4f}")
    print(f"  NO2 Validation R²: {no2_metadata['metrics']['val_r2']:.4f}")
    
    # Make predictions for all sites
    print(f"\n{'='*60}")
    print("Making Predictions")
    print(f"{'='*60}")
    
    all_predictions = []
    for site_id in range(1, N_SITES + 1):
        predictions = predict_site(site_id, o3_model, no2_model, feature_cols, DATA_DIR, OUTPUT_DIR)
        if predictions is not None:
            predictions['site_id'] = site_id
            all_predictions.append(predictions)
    
    # Combine all predictions
    if all_predictions:
        combined_predictions = pd.concat(all_predictions, ignore_index=True)
        combined_output_file = os.path.join(OUTPUT_DIR, "all_sites_predictions.csv")
        combined_predictions.to_csv(combined_output_file, index=False)
        print(f"\nCombined predictions saved to: {combined_output_file}")
        
        # Summary statistics
        print(f"\n{'='*60}")
        print("Prediction Summary")
        print(f"{'='*60}")
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
    
    print("="*60)


if __name__ == "__main__":
    main()

