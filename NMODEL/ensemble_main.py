"""
Main script for training and evaluating the ensemble model (LightGBM + TFT).
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import warnings
import pickle
import sys
warnings.filterwarnings('ignore')

from utils import load_data, prepare_features, get_feature_groups, calculate_metrics
from stacking import EnsembleModel


def train_ensemble(site_id: int, target: str = 'NO2', data_dir: str = 'F',
                   use_cv: bool = True, n_splits: int = 5,
                   use_optuna: bool = False, save_model: bool = True,
                   quick_mode: bool = False):
    """
    Train ensemble model for a given site and target.
    
    Args:
        site_id: Site ID (3, 5, 6, or 7)
        target: Target variable ('NO2' or 'O3')
        data_dir: Directory containing data files
        use_cv: Whether to use cross-validation for OOF predictions
        n_splits: Number of CV splits
        use_optuna: Whether to use Optuna for LightGBM hyperparameter tuning
        save_model: Whether to save the trained model
        quick_mode: If True, use fewer trials (10) and folds (3) for faster testing
    """
    print(f"\n{'='*60}")
    print(f"Training Ensemble Model for Site {site_id}, Target: {target}")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading data...")
    train_df, val_df, test_df = load_data(site_id, data_dir)
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    # Prepare features
    print("\nPreparing features...")
    X_train, y_train, sw_train = prepare_features(train_df, target=target)
    X_val, y_val, sw_val = prepare_features(val_df, target=target)
    X_test, y_test, sw_test = prepare_features(test_df, target=target)
    
    print(f"  Number of features: {len(X_train.columns)}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Original rows in train: {sw_train.sum():.0f} ({sw_train.sum()/len(sw_train)*100:.1f}%)")
    
    # Get feature groups for TFT
    feature_groups = get_feature_groups(train_df)
    print(f"\nFeature groups:")
    print(f"  Static: {len(feature_groups['static'])}")
    print(f"  Known future: {len(feature_groups['known_future'])}")
    print(f"  Observed: {len(feature_groups['observed'])}")
    
    # Prepare TFT-specific features
    static_train = train_df[feature_groups['static']] if feature_groups['static'] else None
    static_val = val_df[feature_groups['static']] if feature_groups['static'] else None
    static_test = test_df[feature_groups['static']] if feature_groups['static'] else None
    
    known_future_train = train_df[feature_groups['known_future']] if feature_groups['known_future'] else None
    known_future_val = val_df[feature_groups['known_future']] if feature_groups['known_future'] else None
    known_future_test = test_df[feature_groups['known_future']] if feature_groups['known_future'] else None
    
    # For TFT, we need observed features (excluding static and known future)
    observed_cols = [col for col in X_train.columns if col in feature_groups['observed']]
    X_train_tft = X_train[observed_cols] if observed_cols else X_train
    X_val_tft = X_val[observed_cols] if observed_cols else X_val
    X_test_tft = X_test[observed_cols] if observed_cols else X_test
    
    # Initialize ensemble model
    print("\nInitializing ensemble model...")
    
    # LightGBM parameters
    # Quick mode: reduce trials for faster testing
    n_trials = 10 if (use_optuna and quick_mode) else (50 if use_optuna else 0)
    lgbm_params = {
        'use_optuna': use_optuna,
        'n_trials': n_trials
    }
    
    # TFT parameters
    tft_params = {
        'input_window': 72,
        'output_horizon': 1,  # Single-step prediction
        'hidden_size': 96,
        'num_layers': 2,
        'num_heads': 4,
        'dropout': 0.1,
        'batch_size': 128,
        'learning_rate': 1e-3,
        'epochs': 100,
        'early_stopping_patience': 10
    }
    
    # Stacking parameters
    stacking_params = {
        'meta_model_type': 'ridge',  # or 'lightgbm'
        'ridge_alpha': 1.0
    }
    
    ensemble = EnsembleModel(
        target=target,
        lgbm_params=lgbm_params,
        tft_params=tft_params,
        stacking_params=stacking_params
    )
    
    # Train ensemble
    print("\nTraining ensemble model...")
    try:
        ensemble.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            sample_weight_train=sw_train,
            sample_weight_val=sw_val,
            static_train=static_train,
            static_val=static_val,
            known_future_train=known_future_train,
            known_future_val=known_future_val,
            use_cv=use_cv,
            n_splits=n_splits
        )
        print("✓ Ensemble model training completed successfully")
    except Exception as e:
        print(f"\n✗ ERROR: Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("\nStopping training due to error.")
        sys.exit(1)
    
    # Evaluate individual base models and ensemble
    print("\n=== Model Evaluation ===")
    
    try:
        # Get individual model predictions for comparison
        y_train_val = pd.concat([y_train, y_val])
        
        # LightGBM predictions
        print("\n--- LightGBM Model ---")
        lgbm_val_pred = ensemble.lgbm_model.predict(X_val)
        # Check for NaN in predictions
        if np.any(np.isnan(lgbm_val_pred)):
            raise ValueError("LightGBM validation predictions contain NaN values")
        lgbm_val_metrics = calculate_metrics(y_val.values, lgbm_val_pred)
        print(f"Validation - RMSE: {lgbm_val_metrics['RMSE']:.4f}, MAE: {lgbm_val_metrics['MAE']:.4f}, Bias: {lgbm_val_metrics['Bias']:.4f}")
        
        lgbm_test_pred = ensemble.lgbm_model.predict(X_test)
        if np.any(np.isnan(lgbm_test_pred)):
            raise ValueError("LightGBM test predictions contain NaN values")
        lgbm_test_metrics = calculate_metrics(y_test.values, lgbm_test_pred)
        print(f"Test       - RMSE: {lgbm_test_metrics['RMSE']:.4f}, MAE: {lgbm_test_metrics['MAE']:.4f}, Bias: {lgbm_test_metrics['Bias']:.4f}")
        
        # TFT predictions
        print("\n--- TFT Model ---")
        y_history_val = y_train.iloc[-ensemble.tft_model.input_window:] if len(y_train) >= ensemble.tft_model.input_window else y_train
        tft_val_pred = ensemble.tft_model.predict(X_val_tft, y_history_val, static_val, known_future_val)
        # Handle NaN in TFT predictions
        if np.any(np.isnan(tft_val_pred)):
            print("Warning: TFT validation predictions contain NaN, replacing with LightGBM predictions")
            tft_val_pred = np.where(np.isnan(tft_val_pred), lgbm_val_pred, tft_val_pred)
        tft_val_metrics = calculate_metrics(y_val.values, tft_val_pred)
        print(f"Validation - RMSE: {tft_val_metrics['RMSE']:.4f}, MAE: {tft_val_metrics['MAE']:.4f}, Bias: {tft_val_metrics['Bias']:.4f}")
        
        y_history_test = y_train_val.iloc[-ensemble.tft_model.input_window:] if len(y_train_val) >= ensemble.tft_model.input_window else y_train_val
        tft_test_pred = ensemble.tft_model.predict(X_test_tft, y_history_test, static_test, known_future_test)
        if np.any(np.isnan(tft_test_pred)):
            print("Warning: TFT test predictions contain NaN, replacing with LightGBM predictions")
            tft_test_pred = np.where(np.isnan(tft_test_pred), lgbm_test_pred, tft_test_pred)
        tft_test_metrics = calculate_metrics(y_test.values, tft_test_pred)
        print(f"Test       - RMSE: {tft_test_metrics['RMSE']:.4f}, MAE: {tft_test_metrics['MAE']:.4f}, Bias: {tft_test_metrics['Bias']:.4f}")
        
        # Ensemble predictions
        print("\n--- Ensemble Model (LightGBM + TFT) ---")
        val_pred = ensemble.predict(X_val, y_history=None, static=static_val, 
                                   known_future=known_future_val, y_train_full=y_train)
        if np.any(np.isnan(val_pred)):
            raise ValueError("Ensemble validation predictions contain NaN values")
        val_metrics = calculate_metrics(y_val.values, val_pred)
        print(f"Validation - RMSE: {val_metrics['RMSE']:.4f}, MAE: {val_metrics['MAE']:.4f}, Bias: {val_metrics['Bias']:.4f}")
        
        test_pred = ensemble.predict(X_test, y_history=None, static=static_test,
                                     known_future=known_future_test, y_train_full=y_train_val)
        if np.any(np.isnan(test_pred)):
            raise ValueError("Ensemble test predictions contain NaN values")
        test_metrics = calculate_metrics(y_test.values, test_pred)
        print(f"Test       - RMSE: {test_metrics['RMSE']:.4f}, MAE: {test_metrics['MAE']:.4f}, Bias: {test_metrics['Bias']:.4f}")
        
        print("\n✓ Model evaluation completed successfully")
    except Exception as e:
        print(f"\n✗ ERROR: Model evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("\nStopping due to evaluation error.")
        sys.exit(1)
    
    # Save predictions, metrics, and model
    if save_model:
        output_dir = Path('predictions')
        output_dir.mkdir(exist_ok=True)
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        try:
            # Save test predictions
            test_results = pd.DataFrame({
                'datetime': test_df['datetime'],
                'true': y_test.values,
                'predicted_ensemble': test_pred,
                'predicted_lgbm': lgbm_test_pred,
                'predicted_tft': tft_test_pred
            })
            test_results.to_csv(output_dir / f'site{site_id}_{target}_test_predictions.csv', index=False)
            print(f"\n✓ Test predictions saved to: {output_dir / f'site{site_id}_{target}_test_predictions.csv'}")
            
            # Save validation predictions
            val_results = pd.DataFrame({
                'datetime': val_df['datetime'],
                'true': y_val.values,
                'predicted_ensemble': val_pred,
                'predicted_lgbm': lgbm_val_pred,
                'predicted_tft': tft_val_pred
            })
            val_results.to_csv(output_dir / f'site{site_id}_{target}_val_predictions.csv', index=False)
            print(f"✓ Validation predictions saved to: {output_dir / f'site{site_id}_{target}_val_predictions.csv'}")
            
            # Save metrics summary
            metrics_summary = pd.DataFrame({
                'Model': ['LightGBM', 'TFT', 'Ensemble'],
                'Validation_RMSE': [lgbm_val_metrics['RMSE'], tft_val_metrics['RMSE'], val_metrics['RMSE']],
                'Validation_MAE': [lgbm_val_metrics['MAE'], tft_val_metrics['MAE'], val_metrics['MAE']],
                'Validation_Bias': [lgbm_val_metrics['Bias'], tft_val_metrics['Bias'], val_metrics['Bias']],
                'Validation_R2': [lgbm_val_metrics['R2'], tft_val_metrics['R2'], val_metrics['R2']],
                'Test_RMSE': [lgbm_test_metrics['RMSE'], tft_test_metrics['RMSE'], test_metrics['RMSE']],
                'Test_MAE': [lgbm_test_metrics['MAE'], tft_test_metrics['MAE'], test_metrics['MAE']],
                'Test_Bias': [lgbm_test_metrics['Bias'], tft_test_metrics['Bias'], test_metrics['Bias']],
                'Test_R2': [lgbm_test_metrics['R2'], tft_test_metrics['R2'], test_metrics['R2']]
            })
            metrics_summary.to_csv(output_dir / f'site{site_id}_{target}_metrics.csv', index=False)
            print(f"✓ Metrics summary saved to: {output_dir / f'site{site_id}_{target}_metrics.csv'}")
            
            # Save trained ensemble model
            model_path = models_dir / f'site{site_id}_{target}_ensemble_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(ensemble, f)
            print(f"✓ Trained model saved to: {model_path}")
            
            # Save model metadata
            metadata = {
                'site_id': site_id,
                'target': target,
                'n_features': len(X_train.columns),
                'n_train_samples': len(X_train),
                'n_val_samples': len(X_val),
                'n_test_samples': len(X_test),
                'validation_metrics': val_metrics,
                'test_metrics': test_metrics,
                'feature_groups': feature_groups
            }
            metadata_path = models_dir / f'site{site_id}_{target}_model_metadata.pkl'
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            print(f"✓ Model metadata saved to: {metadata_path}")
            
            print(f"\n{'='*60}")
            print("✓ All files saved successfully!")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"\n✗ ERROR: Failed to save files: {e}")
            import traceback
            traceback.print_exc()
            print("\nStopping due to save error.")
            sys.exit(1)
    
    return ensemble, val_metrics, test_metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train ensemble model for air quality prediction')
    parser.add_argument('--site', type=int, default=3, choices=[3, 5, 6, 7],
                       help='Site ID (default: 3)')
    parser.add_argument('--target', type=str, default='NO2', choices=['NO2', 'O3'],
                       help='Target variable (default: NO2)')
    parser.add_argument('--data-dir', type=str, default='F',
                       help='Data directory (default: F)')
    parser.add_argument('--no-cv', action='store_true',
                       help='Disable cross-validation for OOF predictions')
    parser.add_argument('--n-splits', type=int, default=5,
                       help='Number of CV splits (default: 5)')
    parser.add_argument('--use-optuna', action='store_true',
                       help='Use Optuna for LightGBM hyperparameter tuning')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: 3 CV folds, 10 Optuna trials (for faster testing)')
    parser.add_argument('--both-targets', action='store_true',
                       help='Train models for both NO2 and O3')
    
    args = parser.parse_args()
    
    if args.both_targets:
        # Train for both targets
        for target in ['NO2', 'O3']:
            try:
                train_ensemble(
                    site_id=args.site,
                    target=target,
                    data_dir=args.data_dir,
                    use_cv=not args.no_cv,
                    n_splits=3 if args.quick else args.n_splits,  # Quick mode: 3 folds
                    use_optuna=args.use_optuna,
                    quick_mode=args.quick
                )
            except SystemExit:
                # Re-raise SystemExit to stop execution
                raise
            except Exception as e:
                print(f"\n✗ ERROR: Training {target} model failed: {e}")
                import traceback
                traceback.print_exc()
                print(f"\nStopping training due to error in {target} model.")
                sys.exit(1)
    else:
        # Train for single target
        try:
            train_ensemble(
                site_id=args.site,
                target=args.target,
                data_dir=args.data_dir,
                use_cv=not args.no_cv,
                n_splits=3 if args.quick else args.n_splits,  # Quick mode: 3 folds
                use_optuna=args.use_optuna,
                quick_mode=args.quick
            )
        except SystemExit:
            # Re-raise SystemExit to stop execution
            raise
        except Exception as e:
            print(f"\n✗ ERROR: Training failed: {e}")
            import traceback
            traceback.print_exc()
            print("\nStopping training due to error.")
            sys.exit(1)


if __name__ == '__main__':
    main()

