"""
XGBoost Model Training Script - Per Site Models
Trains separate XGBoost models for each site (7 sites total).
Each site model predicts both O3_target and NO2_target.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import json
from datetime import datetime
import joblib

# Configuration
DATA_DIR = "Data_SIH_2025_with_blh/Data_SIH_2025_with_blh/with_satellite"
MODEL_DIR = "models/xgboost_per_site"
N_SITES = 7

# Features to exclude from training
EXCLUDE_FEATURES = ['O3_target', 'NO2_target', 'year', 'month', 'day', 'hour', 'site_id']

# XGBoost hyperparameters
XGB_PARAMS = {
    'O3': {
        'objective': 'reg:squarederror',
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1
    },
    'NO2': {
        'objective': 'reg:squarederror',
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1
    }
}


def load_site_data(data_dir, site_id):
    """Load training data for a specific site."""
    train_file = os.path.join(data_dir, f"site_{site_id}_train_data_with_satellite.csv")
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    
    print(f"Loading site {site_id} data...")
    df = pd.read_csv(train_file)
    print(f"  Loaded {len(df)} samples from site {site_id}")
    
    return df


def prepare_features(df):
    """Prepare features and targets for training."""
    # Get all feature columns (exclude targets and time identifiers)
    feature_cols = [col for col in df.columns if col not in EXCLUDE_FEATURES]
    
    # Check if targets exist
    if 'O3_target' not in df.columns or 'NO2_target' not in df.columns:
        raise ValueError("Target columns (O3_target, NO2_target) not found in data!")
    
    X = df[feature_cols].copy()
    y_o3 = df['O3_target'].copy()
    y_no2 = df['NO2_target'].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Check for infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    print(f"  Features used: {len(feature_cols)}")
    
    return X, y_o3, y_no2, feature_cols


def train_model(X_train, y_train, X_val, y_val, target_name, params):
    """Train XGBoost model for a single target."""
    # Create model
    model = xgb.XGBRegressor(**params)
    
    # Train model with evaluation set for monitoring
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    train_bias = np.mean(y_train_pred - y_train)  # Bias = mean(predicted - actual)
    
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    val_bias = np.mean(y_val_pred - y_val)  # Bias = mean(predicted - actual)
    
    print(f"  {target_name} - Train R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, Bias: {train_bias:.4f}")
    print(f"  {target_name} - Val R²:   {val_r2:.4f}, RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, Bias: {val_bias:.4f}")
    
    return model, {
        'train_rmse': float(train_rmse),
        'train_mae': float(train_mae),
        'train_bias': float(train_bias),
        'train_r2': float(train_r2),
        'val_rmse': float(val_rmse),
        'val_mae': float(val_mae),
        'val_bias': float(val_bias),
        'val_r2': float(val_r2)
    }


def get_feature_importance(model, feature_names, top_n=20):
    """Get top N feature importances."""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance_df.head(top_n)


def save_site_models(site_id, o3_model, no2_model, feature_cols, o3_metrics, no2_metrics, 
                     o3_importance, no2_importance, model_dir, timestamp):
    """Save models and metadata for a site."""
    os.makedirs(model_dir, exist_ok=True)
    site_dir = os.path.join(model_dir, f"site_{site_id}")
    os.makedirs(site_dir, exist_ok=True)
    
    # Save models
    o3_model_path = os.path.join(site_dir, f"O3_model_{timestamp}.pkl")
    no2_model_path = os.path.join(site_dir, f"NO2_model_{timestamp}.pkl")
    
    joblib.dump(o3_model, o3_model_path)
    joblib.dump(no2_model, no2_model_path)
    
    print(f"  Models saved to: {site_dir}")
    
    # Save metadata
    metadata = {
        'site_id': site_id,
        'timestamp': timestamp,
        'n_features': len(feature_cols),
        'features': feature_cols,
        'O3_metrics': o3_metrics,
        'NO2_metrics': no2_metrics,
        'model_type': 'XGBoost',
        'hyperparameters': {
            'O3': XGB_PARAMS['O3'],
            'NO2': XGB_PARAMS['NO2']
        }
    }
    
    metadata_path = os.path.join(site_dir, f"metadata_{timestamp}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save feature importance
    o3_importance_path = os.path.join(site_dir, f"O3_importance_{timestamp}.csv")
    no2_importance_path = os.path.join(site_dir, f"NO2_importance_{timestamp}.csv")
    o3_importance.to_csv(o3_importance_path, index=False)
    no2_importance.to_csv(no2_importance_path, index=False)
    
    # Save standardized metrics (RMSE, MAE, Bias) in CSV format
    metrics_data = {
        'Site': [site_id, site_id],
        'Target': ['O3', 'NO2'],
        'RMSE': [o3_metrics['val_rmse'], no2_metrics['val_rmse']],
        'MAE': [o3_metrics['val_mae'], no2_metrics['val_mae']],
        'Bias': [o3_metrics['val_bias'], no2_metrics['val_bias']],
        'R²': [o3_metrics['val_r2'], no2_metrics['val_r2']]
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_csv_path = os.path.join(site_dir, f"metrics_{timestamp}.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"  Standardized metrics saved to: {metrics_csv_path}")
    
    return site_dir


def train_site_model(site_id, data_dir, model_dir, timestamp):
    """Train models for a single site."""
    print(f"\n{'='*70}")
    print(f"Training models for Site {site_id}")
    print(f"{'='*70}")
    
    # Load site data
    df = load_site_data(data_dir, site_id)
    
    # Prepare features
    X, y_o3, y_no2, feature_cols = prepare_features(df)
    
    # Split data (80% train, 20% validation)
    X_train, X_val, y_o3_train, y_o3_val = train_test_split(
        X, y_o3, test_size=0.2, random_state=42, shuffle=True
    )
    _, _, y_no2_train, y_no2_val = train_test_split(
        X, y_no2, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    
    # Train O3 model
    print(f"\n  Training O3 model...")
    o3_model, o3_metrics = train_model(
        X_train, y_o3_train, X_val, y_o3_val, 'O3', XGB_PARAMS['O3']
    )
    
    # Train NO2 model
    print(f"\n  Training NO2 model...")
    no2_model, no2_metrics = train_model(
        X_train, y_no2_train, X_val, y_no2_val, 'NO2', XGB_PARAMS['NO2']
    )
    
    # Feature importance
    o3_importance = get_feature_importance(o3_model, feature_cols, top_n=20)
    no2_importance = get_feature_importance(no2_model, feature_cols, top_n=20)
    
    # Save models
    site_dir = save_site_models(
        site_id, o3_model, no2_model, feature_cols,
        o3_metrics, no2_metrics, o3_importance, no2_importance,
        model_dir, timestamp
    )
    
    return {
        'site_id': site_id,
        'o3_metrics': o3_metrics,
        'no2_metrics': no2_metrics,
        'n_samples': len(df),
        'n_train': len(X_train),
        'n_val': len(X_val)
    }


def main():
    """Main training function."""
    print("="*70)
    print("XGBoost Model Training - Per Site Models (O3 and NO2)")
    print("="*70)
    
    # Create timestamp for model versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Train models for each site
    all_results = []
    
    for site_id in range(1, N_SITES + 1):
        try:
            result = train_site_model(site_id, DATA_DIR, MODEL_DIR, timestamp)
            all_results.append(result)
        except Exception as e:
            print(f"\nError training site {site_id}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*70}")
    print("Training Summary - All Sites")
    print(f"{'='*70}")
    print(f"Timestamp: {timestamp}")
    print(f"\n{'Site':<6} {'Target':<8} {'RMSE':<12} {'MAE':<12} {'Bias':<12} {'R²':<10}")
    print("-" * 70)
    
    # Prepare data for comprehensive metrics CSV
    all_metrics_data = []
    
    for result in all_results:
        site_id = result['site_id']
        o3_metrics = result['o3_metrics']
        no2_metrics = result['no2_metrics']
        
        # O3 metrics
        print(f"Site {site_id:<3} {'O3':<8} {o3_metrics['val_rmse']:<12.4f} {o3_metrics['val_mae']:<12.4f} {o3_metrics['val_bias']:<12.4f} {o3_metrics['val_r2']:<10.4f}")
        all_metrics_data.append({
            'Site': site_id,
            'Target': 'O3',
            'RMSE': o3_metrics['val_rmse'],
            'MAE': o3_metrics['val_mae'],
            'Bias': o3_metrics['val_bias'],
            'R²': o3_metrics['val_r2'],
            'Train_RMSE': o3_metrics['train_rmse'],
            'Train_MAE': o3_metrics['train_mae'],
            'Train_Bias': o3_metrics['train_bias'],
            'Train_R²': o3_metrics['train_r2']
        })
        
        # NO2 metrics
        print(f"Site {site_id:<3} {'NO2':<8} {no2_metrics['val_rmse']:<12.4f} {no2_metrics['val_mae']:<12.4f} {no2_metrics['val_bias']:<12.4f} {no2_metrics['val_r2']:<10.4f}")
        all_metrics_data.append({
            'Site': site_id,
            'Target': 'NO2',
            'RMSE': no2_metrics['val_rmse'],
            'MAE': no2_metrics['val_mae'],
            'Bias': no2_metrics['val_bias'],
            'R²': no2_metrics['val_r2'],
            'Train_RMSE': no2_metrics['train_rmse'],
            'Train_MAE': no2_metrics['train_mae'],
            'Train_Bias': no2_metrics['train_bias'],
            'Train_R²': no2_metrics['train_r2']
        })
    
    # Save comprehensive metrics CSV
    metrics_df = pd.DataFrame(all_metrics_data)
    metrics_csv_path = os.path.join(MODEL_DIR, f"all_sites_metrics_{timestamp}.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\nComprehensive metrics saved to: {metrics_csv_path}")
    
    # Save overall summary JSON
    summary_path = os.path.join(MODEL_DIR, f"training_summary_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'n_sites': len(all_results),
            'results': all_results
        }, f, indent=2)
    
    print(f"Summary JSON saved to: {summary_path}")
    print(f"All models saved in: {MODEL_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()

