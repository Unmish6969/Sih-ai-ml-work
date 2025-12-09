"""
Model loader utility for loading trained ML models
Supports loading models for 7 individual sites and 1 unified model
Now uses the new xgboost_per_site structure with separate .pkl files and metadata.json
"""

import os
import joblib
import json
import glob
from typing import Dict, Optional, Tuple
from pathlib import Path


class ModelLoader:
    """Loads and manages ML models for air pollution forecasting"""
    
    def __init__(self, models_dir: str = "Data_SIH_2025/models"):
        """
        Initialize model loader
        
        Args:
            models_dir: Path to directory containing model files (relative to backend root or absolute)
        """
        # Convert to Path and resolve relative to backend root if relative
        models_path = Path(models_dir)
        if not models_path.is_absolute():
            # Try relative to current working directory first, then relative to this file
            backend_root = Path(__file__).parent.parent.parent
            models_path = backend_root / models_dir
        self.models_dir = models_path
        self._models_cache: Dict[str, Dict] = {}
        self._feature_cols_cache: Dict[str, list] = {}
    
    def _find_latest_models(self, site_id: int) -> Tuple[Path, Path, Optional[Path]]:
        """
        Find the latest trained models for a site in xgboost_per_site structure
        
        Args:
            site_id: Site number (1-7)
            
        Returns:
            Tuple of (o3_model_path, no2_model_path, metadata_path)
        """
        site_dir = self.models_dir / "xgboost_per_site" / f"site_{site_id}"
        
        if not site_dir.exists():
            # Fallback to old structure
            old_model_file = self.models_dir / f"site_{site_id}_models.joblib"
            if old_model_file.exists():
                return old_model_file, old_model_file, None
            raise FileNotFoundError(f"Model directory not found: {site_dir}")
        
        o3_models = list(site_dir.glob("O3_model_*.pkl"))
        no2_models = list(site_dir.glob("NO2_model_*.pkl"))
        metadata_files = list(site_dir.glob("metadata_*.json"))
        
        if not o3_models or not no2_models:
            raise FileNotFoundError(f"No models found for site {site_id} in {site_dir}")
        
        o3_model_path = max(o3_models, key=lambda p: p.stat().st_mtime)
        no2_model_path = max(no2_models, key=lambda p: p.stat().st_mtime)
        metadata_path = max(metadata_files, key=lambda p: p.stat().st_mtime) if metadata_files else None
        
        return o3_model_path, no2_model_path, metadata_path
    
    def _get_model_path(self, site_id: Optional[int] = None) -> Path:
        """
        Get model file path for a site or unified model (legacy support)
        
        Args:
            site_id: Site number (1-7) or None for unified model
            
        Returns:
            Path to model file
        """
        if site_id is None:
            model_file = self.models_dir / "site_unified_models.joblib"
        else:
            if not 1 <= site_id <= 7:
                raise ValueError(f"Site ID must be between 1 and 7, got {site_id}")
            model_file = self.models_dir / f"site_{site_id}_models.joblib"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        return model_file
    
    def _get_features_path(self, site_id: Optional[int] = None) -> Path:
        """
        Get features file path for a site or unified model (legacy support)
        
        Args:
            site_id: Site number (1-7) or None for unified model
            
        Returns:
            Path to features file
        """
        if site_id is None:
            features_file = self.models_dir / "site_unified_features.txt"
        else:
            if not 1 <= site_id <= 7:
                raise ValueError(f"Site ID must be between 1 and 7, got {site_id}")
            features_file = self.models_dir / f"site_{site_id}_features.txt"
        
        if not features_file.exists():
            raise FileNotFoundError(f"Features file not found: {features_file}")
        
        return features_file
    
    def load_models(self, site_id: Optional[int] = None, use_cache: bool = True) -> Dict:
        """
        Load models for a specific site or unified model
        
        Args:
            site_id: Site number (1-7) or None for unified model
            use_cache: Whether to use cached models if already loaded
            
        Returns:
            Dictionary containing:
                - 'model_o3': Trained model for O3 prediction
                - 'model_no2': Trained model for NO2 prediction
                - 'feature_cols': List of feature column names
                - 'site_id': Site ID or 'unified'
        """
        cache_key = f"site_{site_id}" if site_id else "unified"
        
        # Return cached model if available
        if use_cache and cache_key in self._models_cache:
            return self._models_cache[cache_key]
        
        # Try new structure first (xgboost_per_site)
        if site_id is not None and 1 <= site_id <= 7:
            try:
                o3_model_path, no2_model_path, metadata_path = self._find_latest_models(site_id)
                
                # Load models
                model_o3 = joblib.load(o3_model_path)
                model_no2 = joblib.load(no2_model_path)
                
                # Load feature columns from metadata
                if metadata_path:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    feature_cols = metadata.get('features', [])
                    saved_at = metadata.get('timestamp', 'Unknown')
                else:
                    # Fallback: try to infer from old structure
                    features_path = self._get_features_path(site_id)
                    if features_path.exists():
                        with open(features_path, 'r') as f:
                            feature_cols = [line.strip() for line in f.readlines() if line.strip()]
                    else:
                        raise ValueError(f"Could not find feature list for site {site_id}")
                    saved_at = 'Unknown'
                
                result = {
                    'model_o3': model_o3,
                    'model_no2': model_no2,
                    'feature_cols': feature_cols,
                    'site_id': site_id,
                    'saved_at': saved_at
                }
                
                # Cache the result
                if use_cache:
                    self._models_cache[cache_key] = result
                    self._feature_cols_cache[cache_key] = feature_cols
                
                return result
            except (FileNotFoundError, ValueError) as e:
                # Fall back to old structure if new structure not found
                pass
        
        # Fallback to old structure (for unified model or if new structure not available)
        model_path = self._get_model_path(site_id)
        model_data = joblib.load(model_path)
        
        # Load feature columns
        features_path = self._get_features_path(site_id)
        with open(features_path, 'r') as f:
            feature_cols = [line.strip() for line in f.readlines() if line.strip()]
        
        # Extract models
        result = {
            'model_o3': model_data.get('model_o3'),
            'model_no2': model_data.get('model_no2'),
            'feature_cols': feature_cols,
            'site_id': site_id if site_id else 'unified',
            'saved_at': model_data.get('saved_at', 'Unknown')
        }
        
        # Cache the result
        if use_cache:
            self._models_cache[cache_key] = result
            self._feature_cols_cache[cache_key] = feature_cols
        
        return result
    
    def get_feature_columns(self, site_id: Optional[int] = None) -> list:
        """
        Get feature columns for a specific site or unified model
        
        Args:
            site_id: Site number (1-7) or None for unified model
            
        Returns:
            List of feature column names
        """
        cache_key = f"site_{site_id}" if site_id else "unified"
        
        if cache_key in self._feature_cols_cache:
            return self._feature_cols_cache[cache_key]
        
        # Try new structure first
        if site_id is not None and 1 <= site_id <= 7:
            try:
                _, _, metadata_path = self._find_latest_models(site_id)
                if metadata_path:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    feature_cols = metadata.get('features', [])
                    self._feature_cols_cache[cache_key] = feature_cols
                    return feature_cols
            except (FileNotFoundError, ValueError):
                pass
        
        # Fallback to old structure
        features_path = self._get_features_path(site_id)
        with open(features_path, 'r') as f:
            feature_cols = [line.strip() for line in f.readlines() if line.strip()]
        
        self._feature_cols_cache[cache_key] = feature_cols
        return feature_cols
    
    def clear_cache(self):
        """Clear the model cache"""
        self._models_cache.clear()
        self._feature_cols_cache.clear()


# Global model loader instance
_model_loader: Optional[ModelLoader] = None


def get_model_loader(models_dir: str = "Data_SIH_2025/models") -> ModelLoader:
    """
    Get or create global model loader instance
    
    Args:
        models_dir: Path to directory containing model files
        
    Returns:
        ModelLoader instance
    """
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader(models_dir)
    return _model_loader

