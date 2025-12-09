"""
LightGBM model implementation for air quality prediction.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Tuple, Optional, List, Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
from utils import time_series_cv_splits, prepare_features


class LightGBMModel:
    """
    LightGBM model for air quality prediction with sample weights and time-series CV.
    """
    
    def __init__(self, target: str = 'NO2', use_optuna: bool = False, n_trials: int = 50):
        """
        Initialize LightGBM model.
        
        Args:
            target: Target variable ('NO2' or 'O3')
            use_optuna: Whether to use Optuna for hyperparameter tuning
            n_trials: Number of Optuna trials if use_optuna=True
        """
        self.target = target
        self.use_optuna = use_optuna
        self.n_trials = n_trials
        self.model = None
        self.feature_names = None
        self.best_params = None
        
        # Default hyperparameters
        self.default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_boost_round': 3000,
            'early_stopping_rounds': 200,
            'learning_rate': 0.03,
            'num_leaves': 128,
            'max_depth': 10,
            'min_data_in_leaf': 50,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'lambda_l1': 0.0,
            'lambda_l2': 0.1,
            'verbose': -1,
            'force_col_wise': True
        }
    
    def _objective(self, trial, X_train, y_train, X_val, y_val, sample_weight_train, sample_weight_val):
        """Optuna objective function for hyperparameter tuning."""
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_boost_round': 3000,
            'early_stopping_rounds': 200,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 32, 256),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 1.0),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 1.0),
            'verbose': -1,
            'force_col_wise': True
        }
        
        train_data = lgb.Dataset(
            X_train, 
            label=y_train, 
            weight=sample_weight_train,
            free_raw_data=False
        )
        val_data = lgb.Dataset(
            X_val, 
            label=y_val, 
            weight=sample_weight_val,
            reference=train_data,
            free_raw_data=False
        )
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=200), lgb.log_evaluation(0)]
        )
        
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        
        return rmse
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: pd.DataFrame, y_val: pd.Series,
            sample_weight_train: Optional[pd.Series] = None,
            sample_weight_val: Optional[pd.Series] = None) -> 'LightGBMModel':
        """
        Train the LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            sample_weight_train: Sample weights for training
            sample_weight_val: Sample weights for validation
            
        Returns:
            Self
        """
        self.feature_names = X_train.columns.tolist()
        
        # Prepare sample weights
        if sample_weight_train is None:
            sample_weight_train = pd.Series(1.0, index=X_train.index)
        if sample_weight_val is None:
            sample_weight_val = pd.Series(1.0, index=X_val.index)
        
        # Hyperparameter tuning with Optuna
        if self.use_optuna:
            if not OPTUNA_AVAILABLE:
                raise ImportError(
                    "Optuna is required for hyperparameter tuning. "
                    "Install it with: pip install optuna>=3.0.0"
                )
            study = optuna.create_study(direction='minimize')
            study.optimize(
                lambda trial: self._objective(
                    trial, X_train, y_train, X_val, y_val,
                    sample_weight_train, sample_weight_val
                ),
                n_trials=self.n_trials,
                show_progress_bar=True
            )
            self.best_params = study.best_params
            params = {**self.default_params, **self.best_params}
        else:
            params = self.default_params
        
        # Prepare LightGBM datasets
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            weight=sample_weight_train,
            free_raw_data=False
        )
        val_data = lgb.Dataset(
            X_val,
            label=y_val,
            weight=sample_weight_val,
            reference=train_data,
            free_raw_data=False
        )
        
        # Train model
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=params['early_stopping_rounds']),
                lgb.log_evaluation(100)
            ]
        )
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        return self.model.predict(X, num_iteration=self.model.best_iteration)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance.
        
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        importance = self.model.feature_importance(importance_type='gain')
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, 
                      sample_weights: Optional[pd.Series] = None,
                      n_splits: int = 5) -> Tuple[np.ndarray, Dict]:
        """
        Perform time-series cross-validation and return OOF predictions.
        
        Args:
            X: Features
            y: Target
            sample_weights: Sample weights
            n_splits: Number of CV splits
            
        Returns:
            Tuple of (OOF predictions, metrics dictionary)
        """
        splits = time_series_cv_splits(X, n_splits=n_splits)
        oof_preds = np.zeros(len(X))
        
        metrics_list = []
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            print(f"LightGBM CV Fold {fold + 1}/{n_splits}")
            
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            if sample_weights is not None:
                sw_train = sample_weights.iloc[train_idx]
                sw_val = sample_weights.iloc[val_idx]
            else:
                sw_train = None
                sw_val = None
            
            # Train on fold
            self.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold, sw_train, sw_val)
            
            # Predict on validation fold
            val_preds = self.predict(X_val_fold)
            oof_preds[val_idx] = val_preds
            
            # Calculate metrics
            from utils import calculate_metrics
            fold_metrics = calculate_metrics(y_val_fold.values, val_preds)
            metrics_list.append(fold_metrics)
            print(f"  Fold {fold + 1} RMSE: {fold_metrics['RMSE']:.4f}, MAE: {fold_metrics['MAE']:.4f}, Bias: {fold_metrics['Bias']:.4f}")
        
        # Average metrics
        avg_metrics = {
            'RMSE': np.mean([m['RMSE'] for m in metrics_list]),
            'MAE': np.mean([m['MAE'] for m in metrics_list]),
            'Bias': np.mean([m['Bias'] for m in metrics_list]),
            'R2': np.mean([m['R2'] for m in metrics_list])
        }
        
        return oof_preds, avg_metrics

