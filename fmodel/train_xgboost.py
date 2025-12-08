"""
XGBoost Model Training Script
Trains XGBoost models to predict O3_target and NO2_target for all 7 sites.
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
MODEL_DIR = "models/xgboost"
N_SITES = 7

# Features to exclude from training
EXCLUDE_FEATURES = ['O3_target', 'NO2_target', 'year', 'month', 'day', 'hour']

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


def load_all_sites_data(data_dir, n_sites=7):
    """Load training data from all sites."""
    all_data = []
    
    for site_id in range(1, n_sites + 1):
        train_file = os.path.join(data_dir, f"site_{site_id}_train_data_with_satellite.csv")
        
        if not os.path.exists(train_file):
            print(f"Warning: {train_file} not found. Skipping site {site_id}.")
            continue
        
        print(f"Loading site {site_id} data...")
        df = pd.read_csv(train_file)
        df['site_id'] = site_id  # Add site identifier
        all_data.append(df)
        print(f"  Loaded {len(df)} samples from site {site_id}")
    
    if not all_data:
        raise ValueError("No data files found!")
    
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal combined data: {len(combined_data)} samples")
    return combined_data


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
    
    print(f"\nFeatures used: {len(feature_cols)}")
    print(f"Feature columns: {feature_cols[:10]}..." if len(feature_cols) > 10 else f"Feature columns: {feature_cols}")
    
    return X, y_o3, y_no2, feature_cols


def train_model(X_train, y_train, X_val, y_val, target_name, params):
    """Train XGBoost model for a single target."""
    print(f"\n{'='*60}")
    print(f"Training XGBoost model for {target_name}")
    print(f"{'='*60}")
    
    # Create model
    model = xgb.XGBRegressor(**params)
    
    # Train model with evaluation set for monitoring
    # Note: Early stopping removed for compatibility across XGBoost versions
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
    
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"\nTraining Metrics:")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  MAE:  {train_mae:.4f}")
    print(f"  R²:   {train_r2:.4f}")
    
    print(f"\nValidation Metrics:")
    print(f"  RMSE: {val_rmse:.4f}")
    print(f"  MAE:  {val_mae:.4f}")
    print(f"  R²:   {val_r2:.4f}")
    
    return model, {
        'train_rmse': float(train_rmse),
        'train_mae': float(train_mae),
        'train_r2': float(train_r2),
        'val_rmse': float(val_rmse),
        'val_mae': float(val_mae),
        'val_r2': float(val_r2)
    }


def get_feature_importance(model, feature_names, top_n=20):
    """Get top N feature importances."""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance_df.head(top_n)


def save_model(model, feature_cols, metrics, target_name, model_dir, timestamp):
    """Save model and metadata."""
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, f"{target_name}_model_{timestamp}.pkl")
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save metadata
    metadata = {
        'target': target_name,
        'timestamp': timestamp,
        'n_features': len(feature_cols),
        'features': feature_cols,
        'metrics': metrics,
        'model_type': 'XGBoost',
        'hyperparameters': XGB_PARAMS[target_name]
    }
    
    metadata_path = os.path.join(model_dir, f"{target_name}_metadata_{timestamp}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")
    
    return model_path, metadata_path


def main():
    """Main training function."""
    print("="*60)
    print("XGBoost Model Training for O3 and NO2 Prediction")
    print("="*60)
    
    # Create timestamp for model versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load data
    print(f"\nLoading data from: {DATA_DIR}")
    df = load_all_sites_data(DATA_DIR, N_SITES)
    
    # Prepare features
    X, y_o3, y_no2, feature_cols = prepare_features(df)
    
    # Split data (80% train, 20% validation)
    print(f"\nSplitting data: 80% train, 20% validation")
    X_train, X_val, y_o3_train, y_o3_val = train_test_split(
        X, y_o3, test_size=0.2, random_state=42, shuffle=True
    )
    _, _, y_no2_train, y_no2_val = train_test_split(
        X, y_no2, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Train O3 model
    o3_model, o3_metrics = train_model(
        X_train, y_o3_train, X_val, y_o3_val, 'O3', XGB_PARAMS['O3']
    )
    
    # Train NO2 model
    no2_model, no2_metrics = train_model(
        X_train, y_no2_train, X_val, y_no2_val, 'NO2', XGB_PARAMS['NO2']
    )
    
    # Feature importance
    print(f"\n{'='*60}")
    print("Top 20 Feature Importances - O3 Model")
    print(f"{'='*60}")
    o3_importance = get_feature_importance(o3_model, feature_cols, top_n=20)
    print(o3_importance.to_string(index=False))
    
    print(f"\n{'='*60}")
    print("Top 20 Feature Importances - NO2 Model")
    print(f"{'='*60}")
    no2_importance = get_feature_importance(no2_model, feature_cols, top_n=20)
    print(no2_importance.to_string(index=False))
    
    # Save models
    print(f"\n{'='*60}")
    print("Saving Models")
    print(f"{'='*60}")
    o3_model_path, o3_metadata_path = save_model(
        o3_model, feature_cols, o3_metrics, 'O3', MODEL_DIR, timestamp
    )
    no2_model_path, no2_metadata_path = save_model(
        no2_model, feature_cols, no2_metrics, 'NO2', MODEL_DIR, timestamp
    )
    
    # Save feature importance
    o3_importance_path = os.path.join(MODEL_DIR, f"O3_importance_{timestamp}.csv")
    no2_importance_path = os.path.join(MODEL_DIR, f"NO2_importance_{timestamp}.csv")
    o3_importance.to_csv(o3_importance_path, index=False)
    no2_importance.to_csv(no2_importance_path, index=False)
    print(f"\nFeature importance saved:")
    print(f"  O3: {o3_importance_path}")
    print(f"  NO2: {no2_importance_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}")
    print(f"Timestamp: {timestamp}")
    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Features: {len(feature_cols)}")
    print(f"\nO3 Model - Validation R²: {o3_metrics['val_r2']:.4f}, RMSE: {o3_metrics['val_rmse']:.4f}")
    print(f"NO2 Model - Validation R²: {no2_metrics['val_r2']:.4f}, RMSE: {no2_metrics['val_rmse']:.4f}")
    print(f"\nModels saved in: {MODEL_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()

