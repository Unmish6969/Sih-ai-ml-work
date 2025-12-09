"""
Request schemas for prediction API
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class FeatureData(BaseModel):
    """Single feature data point for prediction
    
    Field Requirements:
    - REQUIRED: year, month, day, hour (must be provided)
    - HIGHLY RECOMMENDED: O3_forecast, NO2_forecast, T_forecast, q_forecast, u_forecast, v_forecast, w_forecast, blh_forecast, SZA_deg
    - OPTIONAL: All other fields (will be computed automatically if not provided)
    """
    # ========== REQUIRED FIELDS ==========
    year: float = Field(..., description="[REQUIRED] Year")
    month: float = Field(..., description="[REQUIRED] Month (1-12)")
    day: float = Field(..., description="[REQUIRED] Day of month (1-31)")
    hour: float = Field(..., description="[REQUIRED] Hour of day (0-23)")
    
    # ========== HIGHLY RECOMMENDED FIELDS ==========
    # These are essential for accurate predictions with the new models
    O3_forecast: Optional[float] = Field(None, description="[RECOMMENDED] O3 forecast value (used for predictions and lag features)")
    NO2_forecast: Optional[float] = Field(None, description="[RECOMMENDED] NO2 forecast value (used for predictions and lag features)")
    T_forecast: Optional[float] = Field(None, description="[RECOMMENDED] Temperature forecast")
    q_forecast: Optional[float] = Field(None, description="[RECOMMENDED] Specific humidity forecast")
    u_forecast: Optional[float] = Field(None, description="[RECOMMENDED] U wind component forecast")
    v_forecast: Optional[float] = Field(None, description="[RECOMMENDED] V wind component forecast")
    w_forecast: Optional[float] = Field(None, description="[RECOMMENDED] W wind component forecast")
    blh_forecast: Optional[float] = Field(None, description="[RECOMMENDED] Boundary layer height forecast (important for new models)")
    
    # ========== OPTIONAL FIELDS - Solar Features (Auto-computed) ==========
    SZA_deg: Optional[float] = Field(None, description="[OPTIONAL] Solar zenith angle in degrees (auto-computed from site coordinates and Asia/Kolkata timezone if not provided)")
    
    # ========== OPTIONAL FIELDS - Satellite Data ==========
    # Satellite data: one value per day, will be filled for all hours automatically
    NO2_satellite: Optional[float] = Field(None, description="[OPTIONAL] NO2 satellite data (one value per day, filled automatically for all hours)")
    HCHO_satellite: Optional[float] = Field(None, description="[OPTIONAL] HCHO satellite data (one value per day, filled automatically for all hours)")
    ratio_satellite: Optional[float] = Field(None, description="[OPTIONAL] Satellite ratio (will be recomputed automatically if satellite data provided)")
    
    # ========== OPTIONAL FIELDS - Solar Features (Auto-computed) ==========
    # All solar features are computed automatically from site coordinates and timezone (Asia/Kolkata)
    # if not provided. SZA_deg is computed first, then other features are derived from it.
    solar_declination_deg: Optional[float] = Field(None, description="[OPTIONAL] Solar declination angle in degrees (auto-computed with SZA_deg)")
    hour_angle_deg: Optional[float] = Field(None, description="[OPTIONAL] Hour angle in degrees (auto-computed with SZA_deg)")
    solar_time_hr: Optional[float] = Field(None, description="[OPTIONAL] Solar time in hours (auto-computed with SZA_deg)")
    EoT_min: Optional[float] = Field(None, description="[OPTIONAL] Equation of time in minutes (auto-computed with SZA_deg)")
    
    # ========== OPTIONAL FIELDS - Lag Features ==========
    # New models use lag1, lag2, lag3, lag6, lag12 (computed automatically from forecasts if not provided)
    O3_target_lag1: Optional[float] = Field(None, description="[OPTIONAL] O3 target lag 1 hour (computed from O3_forecast if not provided)")
    O3_target_lag2: Optional[float] = Field(None, description="[OPTIONAL] O3 target lag 2 hours (computed from O3_forecast if not provided)")
    O3_target_lag3: Optional[float] = Field(None, description="[OPTIONAL] O3 target lag 3 hours (computed from O3_forecast if not provided)")
    O3_target_lag6: Optional[float] = Field(None, description="[OPTIONAL] O3 target lag 6 hours (computed from O3_forecast if not provided)")
    O3_target_lag12: Optional[float] = Field(None, description="[OPTIONAL] O3 target lag 12 hours (computed from O3_forecast if not provided)")
    NO2_target_lag1: Optional[float] = Field(None, description="[OPTIONAL] NO2 target lag 1 hour (computed from NO2_forecast if not provided)")
    NO2_target_lag2: Optional[float] = Field(None, description="[OPTIONAL] NO2 target lag 2 hours (computed from NO2_forecast if not provided)")
    NO2_target_lag3: Optional[float] = Field(None, description="[OPTIONAL] NO2 target lag 3 hours (computed from NO2_forecast if not provided)")
    NO2_target_lag6: Optional[float] = Field(None, description="[OPTIONAL] NO2 target lag 6 hours (computed from NO2_forecast if not provided)")
    NO2_target_lag12: Optional[float] = Field(None, description="[OPTIONAL] NO2 target lag 12 hours (computed from NO2_forecast if not provided)")
    
    # ========== OPTIONAL FIELDS - Legacy (Not Used by New Models) ==========
    # Legacy satellite flags (not used in new pipeline but kept for compatibility)
    NO2_sat_flag: Optional[float] = Field(None, description="[OPTIONAL/LEGACY] NO2 satellite flag (not used by new models)")
    HCHO_sat_flag: Optional[float] = Field(None, description="[OPTIONAL/LEGACY] HCHO satellite flag (not used by new models)")
    ratio_sat_flag: Optional[float] = Field(None, description="[OPTIONAL/LEGACY] Ratio satellite flag (not used by new models)")
    
    # Legacy lag features (not used by new models, kept for backward compatibility)
    O3_target_lag24: Optional[float] = Field(None, description="[OPTIONAL/LEGACY] O3 target lag 24 hours (not used by new models)")
    O3_target_lag168: Optional[float] = Field(None, description="[OPTIONAL/LEGACY] O3 target lag 168 hours (not used by new models)")
    NO2_target_lag24: Optional[float] = Field(None, description="[OPTIONAL/LEGACY] NO2 target lag 24 hours (not used by new models)")
    NO2_target_lag168: Optional[float] = Field(None, description="[OPTIONAL/LEGACY] NO2 target lag 168 hours (not used by new models)")
    
    # Site ID (required for unified model, optional for site-specific endpoints)
    site_id: Optional[int] = Field(None, description="[OPTIONAL] Site ID (required for unified model endpoint, ignored for site-specific endpoints)", ge=1, le=7)
    
    @validator('month')
    def validate_month(cls, v):
        if not 1 <= v <= 12:
            raise ValueError('Month must be between 1 and 12')
        return v
    
    @validator('day')
    def validate_day(cls, v):
        if not 1 <= v <= 31:
            raise ValueError('Day must be between 1 and 31')
        return v
    
    @validator('hour')
    def validate_hour(cls, v):
        if not 0 <= v <= 23:
            raise ValueError('Hour must be between 0 and 23')
        return v


class PredictRequest(BaseModel):
    """Request model for prediction"""
    input_data: List[FeatureData] = Field(..., description="List of feature data points")
    site_id: Optional[int] = Field(None, description="Site ID (1-7) or None for unified model", ge=1, le=7)
    forecast_hours: int = Field(24, description="Number of hours to forecast (24 or 48)", ge=1, le=48)
    
    @validator('forecast_hours')
    def validate_forecast_hours(cls, v):
        if v not in [24, 48]:
            raise ValueError('forecast_hours must be 24 or 48')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "input_data": [
                    {
                        "year": 2024.0,
                        "month": 5.0,
                        "day": 5.0,
                        "hour": 0.0,
                        "O3_forecast": 7.93,
                        "NO2_forecast": 69.81,
                        "T_forecast": 20.71,
                        "q_forecast": 11.12,
                        "u_forecast": -0.17,
                        "v_forecast": -1.87,
                        "w_forecast": -1.56,
                        "blh_forecast": 500.0,
                        "SZA_deg": 45.2,
                        "NO2_satellite": 15.5,
                        "HCHO_satellite": 12.3
                    }
                ],
                "site_id": 1,
                "forecast_hours": 24
            }
        }

