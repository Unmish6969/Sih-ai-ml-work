"""
Response schemas for prediction API
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class PredictionPoint(BaseModel):
    """Single prediction data point"""
    year: float = Field(..., description="Year")
    month: float = Field(..., description="Month")
    day: float = Field(..., description="Day")
    hour: float = Field(..., description="Hour")
    O3_target: float = Field(..., description="Predicted O3 concentration")
    NO2_target: float = Field(..., description="Predicted NO2 concentration")


class LiveSourceMetadata(BaseModel):
    """Metadata about the live WAQI observation used for prediction."""
    station_id: Optional[int] = Field(None, description="WAQI station identifier")
    station_name: Optional[str] = Field(None, description="Station/showing location name")
    station_location: Optional[List[float]] = Field(
        None, description="Station latitude/longitude if provided by WAQI"
    )
    measurement_time: Optional[str] = Field(None, description="Original measurement timestamp from WAQI")
    timezone: Optional[str] = Field(None, description="Timezone string provided by WAQI")
    overall_aqi: Optional[float] = Field(None, description="Reported AQI at measurement time")
    observed_no2: Optional[float] = Field(None, description="Observed NO2 concentration (µg/m³)")
    observed_o3: Optional[float] = Field(None, description="Observed O3 concentration (µg/m³)")


class PredictResponse(BaseModel):
    """Response model for prediction"""
    success: bool = Field(..., description="Whether prediction was successful")
    site_id: Optional[int] = Field(None, description="Site ID used for prediction")
    forecast_hours: int = Field(..., description="Number of hours forecasted")
    predictions: List[PredictionPoint] = Field(..., description="List of predictions")
    message: Optional[str] = Field(None, description="Additional message")
    live_source: Optional[LiveSourceMetadata] = Field(
        None, description="Metadata about the live data source when applicable"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "site_id": 1,
                "forecast_hours": 24,
                "predictions": [
                    {
                        "year": 2024.0,
                        "month": 5.0,
                        "day": 5.0,
                        "hour": 0.0,
                        "O3_target": 45.2,
                        "NO2_target": 68.5
                    }
                ],
                "message": "Predictions generated successfully",
                "live_source": None
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = Field(False, description="Always false for errors")
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")


