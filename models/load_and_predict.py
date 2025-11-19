"""
Script to load saved models and make predictions
Example usage of saved pickle/joblib models
"""

import pandas as pd
import numpy as np
from train_and_evaluate import load_models, prepare_features, process_unseen_data
import os
import sys

def predict_with_saved_models(site_num, unseen_file, data_dir='Data_SIH_2025', format='joblib'):
    """
    Load saved models and make predictions on unseen data
    
    Args:
        site_num: Site number (1-7)
        unseen_file: Path to unseen input data CSV
        data_dir: Data directory
        format: 'pkl' or 'joblib' (default: joblib)
    """
    print(f"\n{'='*70}")
    print(f"Loading Saved Models for Site {site_num}")
    print(f"{'='*70}\n")
    
    # Load models
    try:
        model_data = load_models(site_num, data_dir, format=format)
        print(f"Models loaded successfully!")
        print(f"  Site: {model_data['site_num']}")
        print(f"  Saved at: {model_data.get('saved_at', 'Unknown')}")
        print(f"  Number of features: {len(model_data['feature_cols'])}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please train the models first using: python train_and_evaluate.py {site_num}")
        return None
    
    # Extract models and feature columns
    model_o3 = model_data['model_o3']
    model_no2 = model_data['model_no2']
    feature_cols = model_data['feature_cols']
    
    # Process unseen data
    print(f"\nProcessing unseen data from: {unseen_file}")
    unseen_df = process_unseen_data(unseen_file, site_num)
    
    # Prepare features
    X_unseen, _ = prepare_features(unseen_df, is_training=False)
    
    # Ensure all feature columns are present
    for col in feature_cols:
        if col not in X_unseen.columns:
            X_unseen[col] = 0
            print(f"Warning: Added missing feature {col}")
    
    # Reorder columns to match training
    X_unseen = X_unseen[feature_cols]
    
    # Make predictions
    print("\nMaking predictions...")
    predictions_o3 = model_o3.predict(X_unseen)
    predictions_no2 = model_no2.predict(X_unseen)
    
    # Create output dataframe
    output_df = unseen_df[['year', 'month', 'day', 'hour', 'datetime']].copy()
    output_df['O3_target'] = predictions_o3
    output_df['NO2_target'] = predictions_no2
    
    print(f"Generated predictions for {len(output_df)} rows")
    
    # Save predictions
    output_dir = os.path.join(data_dir, 'predictions')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'site_{site_num}_predictions_from_saved.csv')
    output_df.to_csv(output_file, index=False)
    print(f"\n[OK] Predictions saved to: {output_file}")
    
    return output_df

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python load_and_predict.py <site_num> [unseen_file] [format]")
        print("  site_num: Site number (1-7)")
        print("  unseen_file: Path to unseen input data (optional, uses default)")
        print("  format: 'pkl' or 'joblib' (default: joblib)")
        print("\nExample:")
        print("  python load_and_predict.py 1")
        print("  python load_and_predict.py 1 Data_SIH_2025/site_1_unseen_input_data.csv joblib")
        return
    
    site_num = int(sys.argv[1])
    data_dir = 'Data_SIH_2025'
    
    if len(sys.argv) >= 3:
        unseen_file = sys.argv[2]
    else:
        unseen_file = os.path.join(data_dir, f'site_{site_num}_unseen_input_data.csv')
    
    format = 'joblib'
    if len(sys.argv) >= 4:
        format = sys.argv[3]
    
    predict_with_saved_models(site_num, unseen_file, data_dir, format)

if __name__ == "__main__":
    main()

