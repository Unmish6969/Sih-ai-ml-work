"""
Prediction service for air pollution forecasting
Handles feature preparation and model predictions
Uses the new XGBoost per-site model structure with improved feature engineering
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging

from app.models.loader import get_model_loader

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for making air pollution predictions"""
    
    def __init__(self, models_dir: str = "Data_SIH_2025/models", lat_lon_path: str = "Data_SIH_2025/lat_lon_sites.txt"):
        """
        Initialize prediction service
        
        Args:
            models_dir: Path to directory containing model files (relative to backend root or absolute)
            lat_lon_path: Path to lat_lon_sites.txt file with site coordinates
        """
        self.model_loader = get_model_loader(models_dir)
        self._site_coordinates = self._load_site_coordinates(lat_lon_path)
    
    def _load_site_coordinates(self, lat_lon_path: str) -> Dict[int, Tuple[float, float]]:
        """Load site coordinates from lat_lon_sites.txt file."""
        coordinates = {}
        lat_lon_file = Path(lat_lon_path)
        
        # Resolve relative to backend root if not absolute
        if not lat_lon_file.is_absolute():
            backend_root = Path(__file__).parent.parent.parent
            lat_lon_file = backend_root / lat_lon_path
        
        if not lat_lon_file.exists():
            logger.warning(f"Site coordinates file not found: {lat_lon_file}. SZA calculation will be skipped.")
            return coordinates
        
        try:
            with open(lat_lon_file, 'r') as f:
                lines = f.readlines()
                # Skip header line
                for line in lines[1:]:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        try:
                            site_id = int(parts[0])
                            lat = float(parts[1])
                            lon = float(parts[2])
                            coordinates[site_id] = (lat, lon)
                        except (ValueError, IndexError):
                            continue
            logger.info(f"Loaded coordinates for {len(coordinates)} sites")
        except Exception as e:
            logger.error(f"Error loading site coordinates: {e}")
        
        return coordinates
    
    def _compute_sza_from_coordinates(self, df: pd.DataFrame, site_id: Optional[int] = None) -> pd.DataFrame:
        """
        Compute SZA_deg from site coordinates and datetime (Asia/Kolkata timezone).
        Also computes other solar features (solar_declination_deg, hour_angle_deg, etc.)
        even if SZA_deg is already provided.
        
        Args:
            df: DataFrame with year, month, day, hour columns
            site_id: Site ID (1-7) to get coordinates from
            
        Returns:
            DataFrame with SZA_deg and related solar features added
        """
        # Check if we need to compute SZA_deg or just the other solar features
        sza_provided = 'SZA_deg' in df.columns and df['SZA_deg'].notna().any()
        solar_feature_cols = ['solar_declination_deg', 'hour_angle_deg', 'solar_time_hr', 'EoT_min']
        
        # Check if all solar features are present and valid (not all NaN)
        all_solar_features_valid = all(
            col in df.columns and df[col].notna().any()
            for col in solar_feature_cols
        )
        
        # If SZA and all solar features are provided and valid, skip computation
        if sza_provided and all_solar_features_valid:
            logger.debug(f"All solar features already present, skipping computation")
            return df
        
        if site_id is None or site_id not in self._site_coordinates:
            logger.warning(f"Cannot compute SZA: site_id {site_id} not found in coordinates. Available sites: {list(self._site_coordinates.keys())}")
            return df
        
        lat, lon = self._site_coordinates[site_id]
        
        try:
            # Create datetime with Asia/Kolkata timezone
            dt = pd.to_datetime(
                df[['year', 'month', 'day', 'hour']].astype(int, copy=False),
                errors='coerce'
            )
            dt = dt.dt.tz_localize('Asia/Kolkata', nonexistent='NaT', ambiguous='NaT')
            
            # Day of year
            N = dt.dt.dayofyear
            local_time = dt.dt.hour + dt.dt.minute / 60.0 + dt.dt.second / 3600.0
            
            # Equation of Time (minutes)
            B = 2 * np.pi * (N - 81) / 364.0
            EoT = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
            
            # Solar time correction (India standard meridian 82.5E)
            std_meridian = 82.5
            lon_correction = (lon - std_meridian) / 15.0  # hours
            solar_time = (local_time + lon_correction + (EoT / 60.0)) % 24.0
            hour_angle_deg = 15.0 * (solar_time - 12.0)
            
            # Declination (deg)
            decl_deg = 23.45 * np.sin(2 * np.pi * (284 + N) / 365.0)
            
            # Only compute SZA if not provided
            if not sza_provided:
                # SZA calculation
                lat_r = np.deg2rad(lat)
                decl_r = np.deg2rad(decl_deg)
                h_r = np.deg2rad(hour_angle_deg)
                cos_sza = np.sin(lat_r) * np.sin(decl_r) + np.cos(lat_r) * np.cos(decl_r) * np.cos(h_r)
                cos_sza = np.clip(cos_sza, -1.0, 1.0)
                sza_rad = np.arccos(cos_sza)
                sza_deg = np.rad2deg(sza_rad)
                df['SZA_deg'] = sza_deg
                logger.info(f"Computed SZA_deg for site {site_id} using coordinates ({lat}, {lon})")
            else:
                logger.info(f"SZA_deg already provided, computing other solar features for site {site_id}")
            
            # Always compute other solar features (they're needed even if SZA is provided)
            df['solar_declination_deg'] = decl_deg
            df['hour_angle_deg'] = hour_angle_deg
            df['solar_time_hr'] = solar_time
            df['EoT_min'] = EoT
            
        except Exception as e:
            logger.error(f"Error computing SZA: {e}")
        
        return df
    
    def _impute_daily_value(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Impute missing full-day satellite values by averaging previous/next day."""
        ordered = df.sort_values(["year", "month", "day", "hour"]).reset_index()
        
        # Get first non-null value for each day
        daily_first = (
            ordered.groupby(["year", "month", "day"])[column]
            .apply(lambda s: s.dropna().iloc[0] if not s.dropna().empty else pd.NA)
        )
        
        day_keys = list(daily_first.index)
        filled_values = {}
        
        for i, key in enumerate(day_keys):
            if pd.notna(daily_first.loc[key]):
                continue
            
            prev_val = None
            next_val = None
            
            # Look backward
            for j in range(i - 1, -1, -1):
                candidate = daily_first.iloc[j]
                if pd.notna(candidate):
                    prev_val = float(candidate)
                    break
            
            # Look forward
            for j in range(i + 1, len(day_keys)):
                candidate = daily_first.iloc[j]
                if pd.notna(candidate):
                    next_val = float(candidate)
                    break
            
            if prev_val is not None and next_val is not None:
                filled_values[key] = (prev_val + next_val) / 2.0
        
        # Assign imputed value to first row of each missing day
        for key, value in filled_values.items():
            mask = (
                (ordered["year"] == key[0]) &
                (ordered["month"] == key[1]) &
                (ordered["day"] == key[2])
            )
            if mask.any():
                first_idx = ordered.index[mask][0]
                ordered.loc[first_idx, column] = value
        
        return ordered.set_index("index").sort_index()
    
    def _fill_from_forecast(self, df: pd.DataFrame, forecast_col: str, sat_col: str, out_col: str) -> pd.DataFrame:
        """Normalize forecast by daily max and scale by daily satellite value."""
        filled = df.copy()
        
        def _fill_group(group):
            max_forecast = group[forecast_col].max(skipna=True)
            sat_values = group[sat_col].dropna().unique()
            sat_value = sat_values[0] if len(sat_values) else None
            
            if sat_value is None or pd.isna(max_forecast) or max_forecast <= 0:
                group[out_col] = pd.NA
                return group
            
            normalized = group[forecast_col] / max_forecast
            group[out_col] = normalized * sat_value
            return group
        
        filled = filled.groupby(["year", "month", "day"], group_keys=False).apply(_fill_group)
        return filled
    
    def _fill_satellite_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill satellite data: one value per day → all hours."""
        # Step 1: Impute missing daily satellite values
        if 'NO2_satellite' in df.columns:
            df = self._impute_daily_value(df, "NO2_satellite")
        if 'HCHO_satellite' in df.columns:
            df = self._impute_daily_value(df, "HCHO_satellite")
        
        # Step 2: Fill per-hour satellite series from forecasts
        # Only create filled versions if they don't already exist
        if 'NO2_satellite_filled' not in df.columns:
            if 'NO2_satellite' in df.columns and 'NO2_forecast' in df.columns:
                df = self._fill_from_forecast(df, "NO2_forecast", "NO2_satellite", "NO2_satellite_filled")
            elif 'NO2_satellite' in df.columns:
                # If we have satellite data but no forecast, just forward fill
                df['NO2_satellite_filled'] = df.groupby(['year', 'month', 'day'])['NO2_satellite'].transform(lambda x: x.ffill().bfill())
        
        if 'HCHO_satellite_filled' not in df.columns:
            if 'HCHO_satellite' in df.columns and 'O3_forecast' in df.columns:
                df = self._fill_from_forecast(df, "O3_forecast", "HCHO_satellite", "HCHO_satellite_filled")
            elif 'HCHO_satellite' in df.columns:
                # If we have satellite data but no forecast, just forward fill
                df['HCHO_satellite_filled'] = df.groupby(['year', 'month', 'day'])['HCHO_satellite'].transform(lambda x: x.ffill().bfill())
        
        # Step 3: Recompute ratio_satellite if needed
        if 'ratio_satellite' not in df.columns or df['ratio_satellite'].isna().all():
            def _compute_ratio(row):
                no2 = row.get("NO2_satellite_filled")
                hcho = row.get("HCHO_satellite_filled")
                if pd.isna(no2) or pd.isna(hcho) or no2 == 0:
                    return pd.NA
                return hcho / no2
            
            if 'NO2_satellite_filled' in df.columns and 'HCHO_satellite_filled' in df.columns:
                df["ratio_satellite"] = df.apply(_compute_ratio, axis=1)
        
        # Drop raw satellite columns (we use filled versions)
        df = df.drop(columns=["NO2_satellite", "HCHO_satellite"], errors="ignore")
        
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived forecast features."""
        # BLH features
        if 'blh_forecast' in df.columns:
            df['NO2_forecast_per_blh'] = df['NO2_forecast'] / (df['blh_forecast'] + 1e-6)
            df['O3_forecast_per_blh'] = df['O3_forecast'] / (df['blh_forecast'] + 1e-6)
            df['blh_delta_1h'] = df['blh_forecast'].diff()
            df['blh_rel_change'] = df['blh_forecast'].pct_change()
        
        # Wind features
        if 'u_forecast' in df.columns and 'v_forecast' in df.columns:
            df['wind_speed'] = np.sqrt(df['u_forecast']**2 + df['v_forecast']**2)
            df['wind_dir_deg'] = np.degrees(np.arctan2(df['v_forecast'], df['u_forecast'])) % 360
            df['wind_sector'] = (df['wind_dir_deg'] / 45).astype(int) % 8
        
        # Hour sin/cos (if not already present)
        if 'hour' in df.columns and 'sin_hour' not in df.columns:
            df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Solar features (if SZA_deg is available)
        if 'SZA_deg' in df.columns:
            sza_rad = df['SZA_deg'] * np.pi / 180.0
            df['cosSZA'] = np.cos(sza_rad)
            df['solar_elevation'] = 90.0 - df['SZA_deg']
            df['sunset_flag'] = (df['solar_elevation'] <= 0).astype(int)
        
        return df
    
    def _smooth_satellite_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Smooth satellite data and drop raw columns."""
        satellite_cols = ['NO2_satellite_filled', 'HCHO_satellite_filled', 'ratio_satellite']
        
        for col in satellite_cols:
            smooth_col = f'{col}_smooth'
            # Only create smooth version if it doesn't already exist
            if smooth_col not in df.columns:
                if col in df.columns:
                    # Create smooth version
                    df[smooth_col] = df[col].rolling(window=3, min_periods=1, center=True).mean()
                    # Fill any remaining NaN values with the original value
                    df[smooth_col] = df[smooth_col].fillna(df[col])
                    # Drop the original filled column (we use smooth version)
                    df = df.drop(col, axis=1)
                else:
                    # Column doesn't exist, create with NaN (will be filled later)
                    logger.warning(f"Column {col} not found for smoothing. Creating empty smooth column.")
                    df[smooth_col] = np.nan
            else:
                # Smooth version already exists, drop the filled version if it exists
                if col in df.columns:
                    df = df.drop(col, axis=1)
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features with sin/cos encoding."""
        if all(col in df.columns for col in ['year', 'month', 'day', 'hour']):
            # Create datetime
            df['datetime'] = pd.to_datetime(
                df[['year', 'month', 'day']].astype(int).astype(str).agg('-'.join, axis=1) + 
                ' ' + df['hour'].astype(int).astype(str) + ':00:00',
                errors='coerce'
            )
            
            # Day of week
            df['dayofweek'] = df['datetime'].dt.dayofweek
            
            # Season
            df['season'] = df['datetime'].dt.month % 12 // 3 + 1
            
            # Is weekend
            df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
            
            # Sin/Cos encoding for hour
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            
            # Sin/Cos encoding for month
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            # Drop datetime (but keep it temporarily for sorting)
            # We'll drop it later in prepare_features
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame, use_forecasts: bool = True) -> pd.DataFrame:
        """Add lag features. For prediction, use forecasts as proxy."""
        lag_periods = [1, 2, 3, 6, 12]
        
        if use_forecasts:
            # Use forecasts as proxy for targets (simpler but less accurate)
            for lag in lag_periods:
                if 'NO2_forecast' in df.columns:
                    df[f'NO2_target_lag{lag}'] = df['NO2_forecast'].shift(lag)
                if 'O3_forecast' in df.columns:
                    df[f'O3_target_lag{lag}'] = df['O3_forecast'].shift(lag)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window features."""
        # Rolling features for NO2
        if 'NO2_forecast' in df.columns:
            for window in [3, 6, 12]:
                df[f'NO2_roll{window}'] = df['NO2_forecast'].rolling(window=window, min_periods=1).mean()
        
        # Rolling features for O3
        if 'O3_forecast' in df.columns:
            for window in [3, 6, 12]:
                df[f'O3_roll{window}'] = df['O3_forecast'].rolling(window=window, min_periods=1).mean()
        
        # Rolling features for temperature
        if 'T_forecast' in df.columns:
            df['temp_roll3'] = df['T_forecast'].rolling(window=3, min_periods=1).mean()
        
        # Rolling features for relative humidity
        if 'q_forecast' in df.columns:
            df['rh_roll3'] = df['q_forecast'].rolling(window=3, min_periods=1).mean()
        
        # Rolling features for wind speed
        if 'wind_speed' in df.columns:
            df['ws_roll3'] = df['wind_speed'].rolling(window=3, min_periods=1).mean()
        
        return df
    
    def process_input_data(self, df: pd.DataFrame, site_id: Optional[int] = None) -> pd.DataFrame:
        """
        Process input data for prediction using the new feature engineering pipeline
        
        Args:
            df: Input dataframe with raw features
            site_id: Site ID (required for unified model)
            
        Returns:
            Processed DataFrame ready for feature preparation
        """
        df = df.copy()
        
        # Step 1: Ensure temporal columns are present and sorted
        if all(col in df.columns for col in ['year', 'month', 'day', 'hour']):
            # Convert to int
            df['year'] = df['year'].astype(int)
            df['month'] = df['month'].astype(int)
            df['day'] = df['day'].astype(int)
            df['hour'] = df['hour'].astype(int)
            
            # Sort by temporal order
            df = df.sort_values(by=['year', 'month', 'day', 'hour']).reset_index(drop=True)
        
        # Step 1.5: Compute SZA_deg and solar features (using site coordinates)
        # Always compute solar features even if SZA_deg is provided (needed for other features)
        df = self._compute_sza_from_coordinates(df, site_id)
        
        # Step 2: Fill satellite data (one value per day → all hours)
        df = self._fill_satellite_data(df)
        
        # Step 3: Add derived features
        df = self._add_derived_features(df)
        
        # Step 4: Feature engineering
        # 4.1: Smooth satellite data
        df = self._smooth_satellite_data(df)
        
        # 4.2: Add temporal features
        df = self._add_temporal_features(df)
        
        # 4.3: Add lag features (using forecasts as proxy)
        df = self._add_lag_features(df, use_forecasts=True)
        
        # 4.4: Add rolling window features
        df = self._add_rolling_features(df)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Prepare features for prediction matching training pipeline
        
        Args:
            df: Input dataframe with processed features
            feature_cols: List of required feature column names from model metadata
            
        Returns:
            DataFrame with prepared features in correct order
        """
        # Check for missing features
        missing_features = [f for f in feature_cols if f not in df.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Add missing features with NaN (will be filled with median)
            for f in missing_features:
                df[f] = np.nan
        
        # Select only the features used in training
        X = df[feature_cols].copy()
        
        # Handle missing values (use median from training if available, else compute)
        X = X.fillna(X.median())
        
        # Check for infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        return X
    
    def ensure_feature_columns(self, df: pd.DataFrame, required_features: List[str]) -> pd.DataFrame:
        """
        Ensure all required feature columns are present in the dataframe
        
        Args:
            df: Input dataframe
            required_features: List of required feature column names
            
        Returns:
            DataFrame with all required features (missing ones filled with 0)
        """
        df = df.copy()
        
        # Add missing features with default value 0
        for col in required_features:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder columns to match required order
        df = df[required_features]
        
        return df
    
    def predict(
        self,
        input_data: pd.DataFrame,
        site_id: Optional[int] = None,
        forecast_hours: int = 24
    ) -> pd.DataFrame:
        """
        Make predictions for O3 and NO2 using the new feature engineering pipeline
        
        Args:
            input_data: DataFrame with input features
            site_id: Site number (1-7) or None for unified model
            forecast_hours: Number of hours to forecast (24 or 48)
            
        Returns:
            DataFrame with predictions including:
                - year, month, day, hour
                - O3_target: Predicted O3 values
                - NO2_target: Predicted NO2 values
        """
        # Load models and get feature list
        model_data = self.model_loader.load_models(site_id)
        model_o3 = model_data['model_o3']
        model_no2 = model_data['model_no2']
        feature_cols = model_data['feature_cols']
        
        # Process input data using new feature engineering pipeline
        processed_df = self.process_input_data(input_data, site_id)
        
        # Prepare features matching training pipeline (uses feature_cols from metadata)
        X = self.prepare_features(processed_df, feature_cols)
        
        # Make predictions
        predictions_o3 = model_o3.predict(X)
        predictions_no2 = model_no2.predict(X)
        
        # Create output dataframe
        # Get original temporal columns from processed_df
        output_df = processed_df[['year', 'month', 'day', 'hour']].copy()
        output_df['O3_target'] = predictions_o3
        output_df['NO2_target'] = predictions_no2
        
        # Limit to requested forecast hours
        if len(output_df) > forecast_hours:
            output_df = output_df.head(forecast_hours)
        
        return output_df
    
    def predict_from_dict(
        self,
        input_data: List[Dict],
        site_id: Optional[int] = None,
        forecast_hours: int = 24
    ) -> List[Dict]:
        """
        Make predictions from dictionary input
        
        Args:
            input_data: List of dictionaries with input features
            site_id: Site number (1-7) or None for unified model
            forecast_hours: Number of hours to forecast (24 or 48)
            
        Returns:
            List of dictionaries with predictions
        """
        # Convert to DataFrame
        df = pd.DataFrame(input_data)
        
        # Add site_id if using unified model
        if site_id is None and 'site_id' not in df.columns:
            raise ValueError("site_id must be provided in input data for unified model")
        
        # Make predictions
        predictions_df = self.predict(df, site_id, forecast_hours)
        
        # Convert back to list of dictionaries
        return predictions_df.to_dict('records')
    
    def validate_input_data(self, input_data: List[Dict], site_id: Optional[int] = None) -> Tuple[bool, Optional[str]]:
        """
        Validate input data format
        
        Args:
            input_data: List of dictionaries with input features
            site_id: Site number (1-7) or None for unified model
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not input_data:
            return False, "Input data is empty"
        
        # Check required columns
        required_cols = ['year', 'month', 'day', 'hour']
        for i, record in enumerate(input_data):
            for col in required_cols:
                if col not in record:
                    return False, f"Missing required column '{col}' in record {i}"
        
        # Check unified model requirement
        if site_id is None:
            if 'site_id' not in input_data[0]:
                return False, "site_id is required in input data for unified model"
        
        return True, None

