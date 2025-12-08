"""
Stacking/Ensemble meta-learner for combining base model predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
from utils import calculate_metrics


class StackingEnsemble:
    """
    Stacking ensemble that combines base model predictions using a meta-learner.
    """
    
    def __init__(self, meta_model_type: str = 'ridge', 
                 ridge_alpha: float = 1.0,
                 lgbm_max_depth: int = 3,
                 lgbm_num_leaves: int = 7):
        """
        Initialize stacking ensemble.
        
        Args:
            meta_model_type: Type of meta-learner ('ridge' or 'lightgbm')
            ridge_alpha: Regularization strength for Ridge regression
            lgbm_max_depth: Max depth for LightGBM meta-learner
            lgbm_num_leaves: Number of leaves for LightGBM meta-learner
        """
        self.meta_model_type = meta_model_type
        self.ridge_alpha = ridge_alpha
        self.lgbm_max_depth = lgbm_max_depth
        self.lgbm_num_leaves = lgbm_num_leaves
        self.meta_model = None
        self.base_models = {}
    
    def fit(self, oof_predictions: Dict[str, np.ndarray], 
            y_true: np.ndarray,
            val_predictions: Optional[Dict[str, np.ndarray]] = None,
            y_val: Optional[np.ndarray] = None) -> 'StackingEnsemble':
        """
        Fit the meta-learner on OOF predictions.
        
        Args:
            oof_predictions: Dictionary of OOF predictions from base models
                Format: {'model_name': oof_preds_array}
            y_true: True target values for OOF predictions
            val_predictions: Optional validation predictions for tuning
            y_val: Optional validation target values
            
        Returns:
            Self
        """
        # Handle NaN values in OOF predictions (replace with fallback or mean)
        # This can happen if TFT training fails or produces NaN predictions
        oof_predictions_clean = {}
        for model_name, preds in oof_predictions.items():
            preds_clean = preds.copy()
            nan_mask = np.isnan(preds_clean)
            if np.any(nan_mask):
                nan_count = np.sum(nan_mask)
                print(f"Warning: {nan_count} NaN values found in {model_name} OOF predictions, replacing with mean")
                # Replace NaN with mean of non-NaN values
                mean_val = np.nanmean(preds_clean)
                preds_clean[nan_mask] = mean_val if not np.isnan(mean_val) else 0.0
            oof_predictions_clean[model_name] = preds_clean
        
        # Prepare meta-features from OOF predictions
        meta_X = np.column_stack([preds for preds in oof_predictions_clean.values()])
        
        # Final check: ensure no NaN in meta_X
        if np.any(np.isnan(meta_X)):
            print("Warning: NaN values still present in meta_X after cleaning, replacing with 0")
            meta_X = np.nan_to_num(meta_X, nan=0.0)
        
        # Fit meta-learner
        if self.meta_model_type == 'ridge':
            self.meta_model = Ridge(alpha=self.ridge_alpha)
            self.meta_model.fit(meta_X, y_true)
            
            # Tune alpha on validation set if provided
            if val_predictions is not None and y_val is not None:
                # Handle NaN in validation predictions
                val_predictions_clean = {}
                for model_name, preds in val_predictions.items():
                    preds_clean = preds.copy()
                    nan_mask = np.isnan(preds_clean)
                    if np.any(nan_mask):
                        nan_count = np.sum(nan_mask)
                        print(f"Warning: {nan_count} NaN values found in {model_name} validation predictions, replacing with mean")
                        mean_val = np.nanmean(preds_clean)
                        preds_clean[nan_mask] = mean_val if not np.isnan(mean_val) else 0.0
                    val_predictions_clean[model_name] = preds_clean
                
                best_alpha = self.ridge_alpha
                best_score = float('inf')
                
                for alpha in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
                    temp_model = Ridge(alpha=alpha)
                    temp_model.fit(meta_X, y_true)
                    val_meta_X = np.column_stack([preds for preds in val_predictions_clean.values()])
                    # Final check for NaN
                    if np.any(np.isnan(val_meta_X)):
                        val_meta_X = np.nan_to_num(val_meta_X, nan=0.0)
                    val_pred = temp_model.predict(val_meta_X)
                    score = np.sqrt(mean_squared_error(y_val, val_pred))
                    
                    if score < best_score:
                        best_score = score
                        best_alpha = alpha
                
                self.ridge_alpha = best_alpha
                self.meta_model = Ridge(alpha=best_alpha)
                self.meta_model.fit(meta_X, y_true)
                print(f"Best Ridge alpha: {best_alpha}, Val RMSE: {best_score:.4f}")
        
        elif self.meta_model_type == 'lightgbm':
            train_data = lgb.Dataset(meta_X, label=y_true)
            
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': self.lgbm_num_leaves,
                'max_depth': self.lgbm_max_depth,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 1,
                'verbose': -1,
                'force_col_wise': True
            }
            
            if val_predictions is not None and y_val is not None:
                # Handle NaN in validation predictions
                val_predictions_clean = {}
                for model_name, preds in val_predictions.items():
                    preds_clean = preds.copy()
                    nan_mask = np.isnan(preds_clean)
                    if np.any(nan_mask):
                        nan_count = np.sum(nan_mask)
                        print(f"Warning: {nan_count} NaN values found in {model_name} validation predictions, replacing with mean")
                        mean_val = np.nanmean(preds_clean)
                        preds_clean[nan_mask] = mean_val if not np.isnan(mean_val) else 0.0
                    val_predictions_clean[model_name] = preds_clean
                
                val_meta_X = np.column_stack([preds for preds in val_predictions_clean.values()])
                # Final check for NaN
                if np.any(np.isnan(val_meta_X)):
                    val_meta_X = np.nan_to_num(val_meta_X, nan=0.0)
                val_data = lgb.Dataset(val_meta_X, label=y_val, reference=train_data)
                
                self.meta_model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[val_data],
                    num_boost_round=500,
                    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
                )
            else:
                self.meta_model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=500
                )
        
        else:
            raise ValueError(f"Unknown meta_model_type: {self.meta_model_type}")
        
        return self
    
    def predict(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Make ensemble predictions from base model predictions.
        
        Args:
            predictions: Dictionary of predictions from base models
                Format: {'model_name': preds_array}
        
        Returns:
            Ensemble predictions
        """
        if self.meta_model is None:
            raise ValueError("Meta-model not trained. Call fit() first.")
        
        # Prepare meta-features
        meta_X = np.column_stack([preds for preds in predictions.values()])
        
        # Predict with meta-model
        if self.meta_model_type == 'ridge':
            ensemble_pred = self.meta_model.predict(meta_X)
        else:  # lightgbm
            ensemble_pred = self.meta_model.predict(meta_X, num_iteration=self.meta_model.best_iteration)
        
        return ensemble_pred
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from meta-learner (if available).
        
        Returns:
            DataFrame with feature importance
        """
        if self.meta_model is None:
            raise ValueError("Meta-model not trained. Call fit() first.")
        
        if self.meta_model_type == 'ridge':
            # Ridge coefficients as importance
            importance = np.abs(self.meta_model.coef_)
            feature_names = list(self.base_models.keys())
            return pd.DataFrame({
                'model': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        
        elif self.meta_model_type == 'lightgbm':
            # LightGBM feature importance
            importance = self.meta_model.feature_importance(importance_type='gain')
            feature_names = list(self.base_models.keys())
            return pd.DataFrame({
                'model': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        
        return pd.DataFrame()


class EnsembleModel:
    """
    High-level ensemble model that combines LightGBM and TFT.
    """
    
    def __init__(self, target: str = 'NO2',
                 lgbm_params: Optional[Dict] = None,
                 tft_params: Optional[Dict] = None,
                 stacking_params: Optional[Dict] = None):
        """
        Initialize ensemble model.
        
        Args:
            target: Target variable ('NO2' or 'O3')
            lgbm_params: Parameters for LightGBM model
            tft_params: Parameters for TFT model
            stacking_params: Parameters for stacking ensemble
        """
        self.target = target
        
        # Import models
        from lightgbm_model import LightGBMModel
        from tft_model import TFTModel
        
        # Initialize base models
        lgbm_params = lgbm_params or {}
        tft_params = tft_params or {}
        stacking_params = stacking_params or {}
        
        self.lgbm_model = LightGBMModel(target=target, **lgbm_params)
        self.tft_model = TFTModel(**tft_params)
        self.stacking = StackingEnsemble(**stacking_params)
        
        self.is_fitted = False
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame, y_val: pd.Series,
            sample_weight_train: Optional[pd.Series] = None,
            sample_weight_val: Optional[pd.Series] = None,
            static_train: Optional[pd.DataFrame] = None,
            static_val: Optional[pd.DataFrame] = None,
            known_future_train: Optional[pd.DataFrame] = None,
            known_future_val: Optional[pd.DataFrame] = None,
            use_cv: bool = True, n_splits: int = 5) -> 'EnsembleModel':
        """
        Train the ensemble model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            sample_weight_train: Training sample weights
            sample_weight_val: Validation sample weights
            static_train: Training static features
            static_val: Validation static features
            known_future_train: Training known future covariates
            known_future_val: Validation known future covariates
            use_cv: Whether to use cross-validation for OOF predictions
            n_splits: Number of CV splits if use_cv=True
            
        Returns:
            Self
        """
        if use_cv:
            # Generate OOF predictions using time-series CV
            print("Generating OOF predictions with time-series CV...")
            
            # LightGBM OOF predictions
            print("\n=== LightGBM Cross-Validation ===")
            from utils import time_series_cv_splits
            splits = time_series_cv_splits(X_train, n_splits=n_splits)
            
            # Initialize OOF arrays with NaN to detect missing folds
            n_train = len(X_train)
            lgbm_oof = np.full(n_train, np.nan, dtype=float)
            tft_oof = np.full(n_train, np.nan, dtype=float)
            
            # Defensive bounds check after splits
            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                # Handle empty arrays (can happen for first fold with minimal training data)
                max_train = np.max(train_idx) if len(train_idx) > 0 else -1
                max_val = np.max(val_idx) if len(val_idx) > 0 else -1
                
                if max_val >= n_train or (max_train >= n_train and len(train_idx) > 0):
                    raise IndexError(
                        f"CV indices out-of-bounds for fold {fold_idx + 1}: "
                        f"max(train_idx)={max_train}, max(val_idx)={max_val}, n_train={n_train}"
                    )
            
            for fold, (train_idx, val_idx) in enumerate(splits):
                print(f"\nFold {fold + 1}/{n_splits}")
                
                # Defensive sanity checks - fail early with clear message
                # Handle empty train_idx (can happen for first fold with minimal training data)
                if len(train_idx) > 0:
                    assert max(train_idx) < len(X_train), f"train_idx out of range: max={max(train_idx)} >= len(X_train)={len(X_train)}"
                assert max(val_idx) < len(X_train), f"val_idx out of range: max={max(val_idx)} >= len(X_train)={len(X_train)}"
                if static_train is not None:
                    assert len(static_train) == len(X_train), f"static_train length ({len(static_train)}) must equal X_train length ({len(X_train)}) for CV slicing"
                if known_future_train is not None:
                    assert len(known_future_train) == len(X_train), f"known_future_train length ({len(known_future_train)}) must equal X_train length ({len(X_train)}) for CV slicing"
                if sample_weight_train is not None:
                    assert len(sample_weight_train) == len(X_train), f"sample_weight_train length ({len(sample_weight_train)}) must equal X_train length ({len(X_train)}) for CV slicing"
                
                # Skip fold if training set is empty (can't train without data)
                if len(train_idx) == 0:
                    print(f"  Warning: Fold {fold + 1} has empty training set, skipping...")
                    # Use mean of target as fallback prediction
                    mean_pred = y_train.mean()
                    lgbm_oof[val_idx] = mean_pred
                    tft_oof[val_idx] = mean_pred
                    continue
                
                X_train_fold = X_train.iloc[train_idx]
                y_train_fold = y_train.iloc[train_idx]
                X_val_fold = X_train.iloc[val_idx]
                y_val_fold = y_train.iloc[val_idx]
                
                # Slice sample weights aligned to X_train if provided
                sw_train_fold = sample_weight_train.iloc[train_idx] if sample_weight_train is not None else None
                sw_val_fold = sample_weight_train.iloc[val_idx] if sample_weight_train is not None else None
                
                # Train LightGBM
                print("  Training LightGBM...")
                self.lgbm_model.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                                   sw_train_fold, sw_val_fold)
                lgbm_oof[val_idx] = self.lgbm_model.predict(X_val_fold)
                
                # Train TFT
                print("  Training TFT...")
                # CRITICAL: For CV folds, ALWAYS slice from the *_train tables that align to X_train
                # static_train and known_future_train are aligned with X_train (same index/length)
                # val_idx refers to indices in X_train, so we must slice static_train/known_future_train
                static_train_fold = static_train.iloc[train_idx] if static_train is not None else None
                static_val_fold = static_train.iloc[val_idx] if static_train is not None else None
                
                known_train_fold = known_future_train.iloc[train_idx] if known_future_train is not None else None
                known_val_fold = known_future_train.iloc[val_idx] if known_future_train is not None else None
                
                # Check if fold has enough samples for TFT (defensive)
                input_window = self.tft_model.input_window
                output_horizon = self.tft_model.output_horizon
                min_samples = input_window + output_horizon
                if len(X_train_fold) < min_samples:
                    print(f"  SKIP TFT fold {fold + 1}: not enough samples ({len(X_train_fold)} < {min_samples})")
                    # Use LightGBM predictions only for this fold
                    tft_oof[val_idx] = lgbm_oof[val_idx]  # Fallback to LightGBM predictions
                    continue
                
                try:
                    self.tft_model.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                                      static_train_fold, static_val_fold,
                                      known_train_fold, known_val_fold,
                                      sw_train_fold, sw_val_fold)
                except ValueError as e:
                    if "Not enough samples" in str(e):
                        print(f"  SKIP TFT fold {fold + 1}: {e}")
                        tft_oof[val_idx] = lgbm_oof[val_idx]  # Fallback to LightGBM predictions
                        continue
                    else:
                        raise
                
                # For TFT prediction, use the last part of training data as history
                # Use last input_window samples from training fold as history
                if len(y_train_fold) >= input_window:
                    y_history_for_pred = y_train_fold.iloc[-input_window:]
                else:
                    y_history_for_pred = y_train_fold
                
                # Ensure shapes are consistent for prediction
                assert len(X_val_fold) == len(static_val_fold) if static_val_fold is not None else True, \
                    f"X_val_fold length ({len(X_val_fold)}) must match static_val_fold length ({len(static_val_fold)})"
                assert len(X_val_fold) == len(known_val_fold) if known_val_fold is not None else True, \
                    f"X_val_fold length ({len(X_val_fold)}) must match known_val_fold length ({len(known_val_fold)})"
                
                tft_pred = self.tft_model.predict(X_val_fold, y_history_for_pred if len(y_history_for_pred) > 0 else None,
                                                  static_val_fold, known_val_fold)
                # Handle NaN predictions from TFT (fallback to LightGBM)
                nan_mask = np.isnan(tft_pred)
                if np.any(nan_mask):
                    nan_count = np.sum(nan_mask)
                    print(f"  Warning: {nan_count} NaN values in TFT predictions for fold {fold + 1}, replacing with LightGBM predictions")
                    tft_pred[nan_mask] = lgbm_oof[val_idx][nan_mask]
                tft_oof[val_idx] = tft_pred
            
            # Assert that all OOF predictions are filled (no NaN unless intentionally skipped)
            if np.any(np.isnan(lgbm_oof)):
                raise ValueError(f"LightGBM OOF has {np.sum(np.isnan(lgbm_oof))} NaN values - some folds were not filled")
            if np.any(np.isnan(tft_oof)):
                # TFT OOF may have NaNs if folds were skipped - replace with LightGBM
                nan_mask = np.isnan(tft_oof)
                print(f"Warning: TFT OOF has {np.sum(nan_mask)} NaN values, replacing with LightGBM predictions")
                tft_oof[nan_mask] = lgbm_oof[nan_mask]
            
            # Fit meta-learner on OOF predictions
            print("\n=== Training Meta-Learner ===")
            oof_predictions = {
                'lightgbm': lgbm_oof,
                'tft': tft_oof
            }
            
            # Get validation predictions for tuning
            print("Getting validation predictions...")
            self.lgbm_model.fit(X_train, y_train, X_val, y_val,
                               sample_weight_train, sample_weight_val)
            lgbm_val_pred = self.lgbm_model.predict(X_val)
            
            self.tft_model.fit(X_train, y_train, X_val, y_val,
                              static_train, static_val,
                              known_future_train, known_future_val,
                              sample_weight_train, sample_weight_val)
            # Use last part of training data as history for validation prediction
            y_history_val = y_train.iloc[-self.tft_model.input_window:] if len(y_train) >= self.tft_model.input_window else y_train
            tft_val_pred = self.tft_model.predict(X_val, y_history_val, static_val, known_future_val)
            
            # Handle NaN in TFT validation predictions (fallback to LightGBM)
            nan_mask = np.isnan(tft_val_pred)
            if np.any(nan_mask):
                nan_count = np.sum(nan_mask)
                print(f"Warning: {nan_count} NaN values in TFT validation predictions, replacing with LightGBM predictions")
                tft_val_pred[nan_mask] = lgbm_val_pred[nan_mask]
            
            val_predictions = {
                'lightgbm': lgbm_val_pred,
                'tft': tft_val_pred
            }
            
            self.stacking.fit(oof_predictions, y_train.values,
                            val_predictions, y_val.values)
        
        else:
            # Train on full training set (no CV)
            print("Training on full training set...")
            self.lgbm_model.fit(X_train, y_train, X_val, y_val,
                               sample_weight_train, sample_weight_val)
            self.tft_model.fit(X_train, y_train, X_val, y_val,
                              static_train, static_val,
                              known_future_train, known_future_val,
                              sample_weight_train, sample_weight_val)
            
            # For stacking without CV, use validation predictions
            lgbm_val_pred = self.lgbm_model.predict(X_val)
            tft_val_pred = self.tft_model.predict(X_val, y_train, static_val, known_future_val)
            
            # Handle NaN in TFT validation predictions (fallback to LightGBM)
            nan_mask = np.isnan(tft_val_pred)
            if np.any(nan_mask):
                nan_count = np.sum(nan_mask)
                print(f"Warning: {nan_count} NaN values in TFT validation predictions, replacing with LightGBM predictions")
                tft_val_pred[nan_mask] = lgbm_val_pred[nan_mask]
            
            val_predictions = {
                'lightgbm': lgbm_val_pred,
                'tft': tft_val_pred
            }
            
            # Use validation set as "OOF" for stacking
            self.stacking.fit(val_predictions, y_val.values)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame, y_history: Optional[pd.Series] = None,
                static: Optional[pd.DataFrame] = None,
                known_future: Optional[pd.DataFrame] = None,
                y_train_full: Optional[pd.Series] = None) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Features
            y_history: Historical target values (for TFT) - if None, will use y_train_full
            static: Static features (for TFT)
            known_future: Known future covariates (for TFT)
            y_train_full: Full training target (used if y_history is None)
            
        Returns:
            Ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Get predictions from base models
        lgbm_pred = self.lgbm_model.predict(X)
        
        # For TFT, use provided y_history or extract from y_train_full
        if y_history is None and y_train_full is not None:
            # Use last input_window samples from training data as history
            y_history = y_train_full.iloc[-self.tft_model.input_window:] if len(y_train_full) >= self.tft_model.input_window else y_train_full
        
        tft_pred = self.tft_model.predict(X, y_history, static, known_future)
        
        predictions = {
            'lightgbm': lgbm_pred,
            'tft': tft_pred
        }
        
        # Combine with meta-learner
        ensemble_pred = self.stacking.predict(predictions)
        
        return ensemble_pred

