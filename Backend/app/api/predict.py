"""
Prediction API routes for air pollution forecasting
"""

from fastapi import APIRouter, HTTPException, status, UploadFile, File
from typing import Optional, List, Dict, Any
import pandas as pd
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
import io

from app.schemas.predict_request import PredictRequest
from app.schemas.predict_response import PredictResponse, ErrorResponse, LiveSourceMetadata
from app.services.prediction_service import PredictionService
from app.services.live_data_service import LiveDataService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/predict", tags=["predictions"])

# Initialize services (paths resolved relative to backend root)
prediction_service = PredictionService(
    models_dir="Data_SIH_2025/models",
    lat_lon_path="Data_SIH_2025/lat_lon_sites.txt"
)
live_data_service = LiveDataService(lat_lon_path="Data_SIH_2025/lat_lon_sites.txt")
METRICS_FILE = Path(__file__).resolve().parent.parent / "data" / "model_metrics.json"
_cached_metrics: Optional[dict] = None


def _load_model_metrics() -> Optional[dict]:
    global _cached_metrics
    if _cached_metrics is not None:
        return _cached_metrics
    if not METRICS_FILE.exists():
        logger.error("Model metrics file not found: %s", METRICS_FILE)
        return None
    try:
        with open(METRICS_FILE, "r", encoding="utf-8") as handle:
            _cached_metrics = json.load(handle)
            return _cached_metrics
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse model metrics JSON: %s", exc)
        return None


@router.get(
    "/metrics",
    status_code=status.HTTP_200_OK,
    summary="Get model evaluation metrics",
    description="Return RMSE/R²/RIA metrics for each trained site model"
)
async def get_model_metrics():
    data = _load_model_metrics()
    if not data:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model metrics are unavailable. Please regenerate them."
        )
    return data


@router.get(
    "/site/{site_id}/historical",
    status_code=status.HTTP_200_OK,
    summary="Get historical observed data",
    description="Return last 30 days (or 24 hours) of observed O3 and NO2 data from training CSV files"
)
async def get_historical_data(site_id: int, days: int = 30, hours: int = None):
    """
    Get historical observed data for a site from the training CSV files.
    
    Args:
        site_id: Site number (1-7)
        days: Number of days to return (default: 30, max: 30). Ignored if hours is provided.
        hours: Number of hours to return (for 24-hour view). If provided, overrides days.
        
    Returns:
        Dictionary with historical observed data
    """
    try:
        if not 1 <= site_id <= 7:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Site ID must be between 1 and 7, got {site_id}"
            )
        
        # Determine hours to fetch
        if hours is not None:
            hours_to_fetch = min(hours, 24)  # Max 24 hours
            time_period = f"{hours_to_fetch} hours"
        else:
            # Validate days parameter: must be positive and within valid range
            if days <= 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"days parameter must be positive (1-30), got {days}"
                )
            if days > 30:
                days = 30
            hours_to_fetch = days * 24
            time_period = f"{days} days"
        
        # Load CSV file for the site
        csv_path = Path(__file__).resolve().parent.parent.parent / "Data_SIH_2025" / f"site_{site_id}_train_data.csv"
        
        if not csv_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Historical data file not found for site {site_id}"
            )
        
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Ensure we have required columns
        required_cols = ['year', 'month', 'day', 'hour', 'O3_target', 'NO2_target']
        if not all(col in df.columns for col in required_cols):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="CSV file missing required columns"
            )
        
        # Convert to datetime and sort
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        df = df.sort_values('datetime', ascending=False)
        
        # Get data based on time period
        if hours is not None:
            # For hours, get last N hours of data points
            df_recent = df.head(hours_to_fetch).sort_values('datetime', ascending=True)
        else:
            # For days, get all data from the last N calendar days from the most recent date in CSV
            # Use the most recent date in the CSV as the reference point (handles legacy data)
            from datetime import datetime
            
            # Find the most recent date in the CSV data
            if len(df) == 0:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No data found in CSV file for site {site_id}"
                )
            
            most_recent_date = df['datetime'].max()
            reference_date = most_recent_date.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Calculate cutoff: N days before the most recent date (at start of day)
            # For 7 days: if most recent is Dec 8, cutoff is Dec 1 (includes Dec 1-8, 7 days)
            # For 30 days: if most recent is Dec 8, cutoff is Nov 8 (includes Nov 8 - Dec 8, 30 days)
            cutoff_date = reference_date - pd.Timedelta(days=days - 1)  # days-1 because we want inclusive
            
            # Filter to only include data from the last N calendar days from the most recent date (inclusive)
            df_recent = df[(df['datetime'] >= cutoff_date) & (df['datetime'] <= most_recent_date)].sort_values('datetime', ascending=True)
            
            # Additional validation: ensure we only have the last N unique calendar days from the reference date
            if len(df_recent) > 0:
                # Get unique dates and sort
                unique_dates = sorted(df_recent['datetime'].dt.date.unique(), reverse=True)
                
                # Take only the last N calendar days from the reference date
                if len(unique_dates) > days:
                    # Get the N most recent dates (closest to reference date)
                    unique_dates_to_keep = unique_dates[:days]
                    # Filter to only those dates
                    df_recent = df_recent[df_recent['datetime'].dt.date.isin(unique_dates_to_keep)]
                
                unique_dates_final = sorted(df_recent['datetime'].dt.date.unique())
                logger.info(f"Historical data: {len(df_recent)} points, {len(unique_dates_final)} unique days from {unique_dates_final[0] if unique_dates_final else 'N/A'} to {unique_dates_final[-1] if unique_dates_final else 'N/A'} (requested: last {days} days from most recent date {reference_date.date()})")
            else:
                logger.warning(f"No data found for the last {days} days from the most recent date {reference_date.date()} in CSV. This may indicate insufficient data in the requested range.")
        
        # Convert to list of dictionaries
        historical_data = []
        for _, row in df_recent.iterrows():
            historical_data.append({
                "year": int(row['year']),
                "month": int(row['month']),
                "day": int(row['day']),
                "hour": int(row['hour']),
                "O3_observed": float(row['O3_target']) if pd.notna(row['O3_target']) else None,
                "NO2_observed": float(row['NO2_target']) if pd.notna(row['NO2_target']) else None,
                "datetime": row['datetime'].strftime("%Y-%m-%d %H:%M:%S")
            })
        
        return {
            "success": True,
            "site_id": site_id,
            "time_period": time_period,
            "data_points": len(historical_data),
            "data": historical_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching historical data: {str(e)}"
        )


@router.get(
    "/site/{site_id}/live",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Live prediction using WAQI data",
    description="Fetch live WAQI measurements for a site and run the site-specific model"
)
async def predict_live_site(site_id: int):
    """
    Fetch live WAQI measurements for the given site and run the model with those features.
    Only a single-hour prediction is returned because WAQI exposes the current observation.
    """
    try:
        if not 1 <= site_id <= 7:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Site ID must be between 1 and 7, got {site_id}"
            )

        live_payload = live_data_service.get_site_features(site_id)
        features = live_payload["features"]
        metadata = live_payload["metadata"]

        predictions = prediction_service.predict_from_dict(
            input_data=[features],
            site_id=site_id,
            forecast_hours=1
        )

        live_source = LiveSourceMetadata(
            station_id=metadata.get("station_id"),
            station_name=metadata.get("station_name"),
            station_location=metadata.get("station_location"),
            measurement_time=metadata.get("measurement_time"),
            timezone=metadata.get("timezone"),
            overall_aqi=metadata.get("overall_aqi"),
            observed_no2=metadata.get("observed_no2"),
            observed_o3=metadata.get("observed_o3"),
        )

        station_name = metadata.get("station_name", "WAQI station")
        return PredictResponse(
            success=True,
            site_id=site_id,
            forecast_hours=1,
            predictions=predictions,
            message=f"Live prediction generated using {station_name}",
            live_source=live_source
        )

    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model for site {site_id} not found. Please ensure models are trained and available."
        )
    except Exception as e:
        logger.error(f"Error generating live prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating live prediction: {str(e)}"
        )


@router.get(
    "/site/{site_id}/live/24h",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="24-hour forecast using live WAQI data",
    description="Fetch live WAQI measurements for a site and generate 24-hour forecast predictions"
)
async def predict_live_24h_site(site_id: int):
    """
    Fetch live WAQI measurements for the given site and generate 24-hour forecast.
    Uses the current observation as the base and generates predictions for the next 24 hours.
    """
    try:
        if not 1 <= site_id <= 7:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Site ID must be between 1 and 7, got {site_id}"
            )

        live_payload = live_data_service.get_site_features(site_id)
        base_features = live_payload["features"]
        metadata = live_payload["metadata"]

        # Generate input data for 24 hours
        # Use the current observation as base and increment time for each hour
        from datetime import datetime, timedelta
        
        input_data = []
        base_datetime = datetime(
            int(base_features.get("year", datetime.now().year)),
            int(base_features.get("month", datetime.now().month)),
            int(base_features.get("day", datetime.now().day)),
            int(base_features.get("hour", datetime.now().hour))
        )
        
        for hour_offset in range(24):
            forecast_datetime = base_datetime + timedelta(hours=hour_offset)
            hour_features = base_features.copy()
            hour_features["year"] = forecast_datetime.year
            hour_features["month"] = forecast_datetime.month
            hour_features["day"] = forecast_datetime.day
            hour_features["hour"] = forecast_datetime.hour
            input_data.append(hour_features)

        predictions = prediction_service.predict_from_dict(
            input_data=input_data,
            site_id=site_id,
            forecast_hours=24
        )

        live_source = LiveSourceMetadata(
            station_id=metadata.get("station_id"),
            station_name=metadata.get("station_name"),
            station_location=metadata.get("station_location"),
            measurement_time=metadata.get("measurement_time"),
            timezone=metadata.get("timezone"),
            overall_aqi=metadata.get("overall_aqi"),
            observed_no2=metadata.get("observed_no2"),
            observed_o3=metadata.get("observed_o3"),
        )

        station_name = metadata.get("station_name", "WAQI station")
        return PredictResponse(
            success=True,
            site_id=site_id,
            forecast_hours=24,
            predictions=predictions,
            message=f"24-hour forecast generated using live data from {station_name}",
            live_source=live_source
        )

    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model for site {site_id} not found. Please ensure models are trained and available."
        )
    except Exception as e:
        logger.error(f"Error generating 24-hour forecast: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating 24-hour forecast: {str(e)}"
        )


@router.post(
    "/site/{site_id}",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict air pollution for a specific site",
    description="Make predictions for O3 and NO2 for a specific monitoring site (1-7)"
)
async def predict_site(
    site_id: int,
    request: PredictRequest
):
    """
    Predict air pollution for a specific site
    
    Args:
        site_id: Site number (1-7)
        request: Prediction request with input data and forecast hours
        
    Returns:
        PredictResponse with predictions
    """
    try:
        # Validate site_id
        if not 1 <= site_id <= 7:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Site ID must be between 1 and 7, got {site_id}"
            )
        
        # Override site_id in request if provided
        if request.site_id is not None and request.site_id != site_id:
            logger.warning(f"Site ID in request ({request.site_id}) differs from path ({site_id}), using path value")
        
        # Convert input data to list of dicts
        input_data = [item.dict() for item in request.input_data]
        
        # Validate input data
        is_valid, error_msg = prediction_service.validate_input_data(input_data, site_id=site_id)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )
        
        # Make predictions
        predictions = prediction_service.predict_from_dict(
            input_data=input_data,
            site_id=site_id,
            forecast_hours=request.forecast_hours
        )
        
        return PredictResponse(
            success=True,
            site_id=site_id,
            forecast_hours=request.forecast_hours,
            predictions=predictions,
            message=f"Successfully generated {len(predictions)} predictions for site {site_id}"
        )
        
    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model for site {site_id} not found. Please ensure models are trained and available."
        )
    except Exception as e:
        logger.error(f"Error making predictions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making predictions: {str(e)}"
        )


@router.post(
    "/site/{site_id}/csv",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict air pollution from CSV file",
    description="Upload a CSV file with input data and get predictions for O3 and NO2"
)
async def predict_site_from_csv(
    site_id: int,
    file: UploadFile = File(..., description="CSV file with input data"),
    forecast_hours: Optional[int] = 24
):
    """
    Predict air pollution from CSV file upload
    
    Args:
        site_id: Site number (1-7)
        file: CSV file with input data (must contain: year, month, day, hour, and forecast fields)
        forecast_hours: Number of hours to forecast (default: 24, max: 48)
        
    Returns:
        PredictResponse with predictions
    """
    logger.info(f"Received CSV upload request for site {site_id}, file: {file.filename}, size: {file.size if hasattr(file, 'size') else 'unknown'}")
    try:
        # Validate site_id
        if not 1 <= site_id <= 7:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Site ID must be between 1 and 7, got {site_id}"
            )
        
        # Validate forecast_hours (if provided)
        if forecast_hours is not None and (forecast_hours < 1 or forecast_hours > 48):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="forecast_hours must be between 1 and 48"
            )
        
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be a CSV file"
            )
        
        # Read CSV file
        logger.info(f"Reading CSV file content...")
        contents = await file.read()
        logger.info(f"Read {len(contents)} bytes from CSV file")
        
        try:
            # Try to decode as UTF-8
            csv_content = contents.decode('utf-8')
        except UnicodeDecodeError:
            # Try other encodings
            try:
                csv_content = contents.decode('latin-1')
            except UnicodeDecodeError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="CSV file encoding not supported. Please use UTF-8 or Latin-1 encoding."
                )
        
        # Parse CSV
        logger.info(f"Parsing CSV content ({len(csv_content)} characters)...")
        try:
            df = pd.read_csv(io.StringIO(csv_content))
            logger.info(f"CSV parsed successfully: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            logger.error(f"Failed to parse CSV: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to parse CSV file: {str(e)}"
            )
        
        # Validate required columns
        required_cols = ['year', 'month', 'day', 'hour']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"CSV file missing required columns: {', '.join(missing_cols)}"
            )
        
        # Convert DataFrame to list of dictionaries
        # Handle case-insensitive column names
        df.columns = df.columns.str.strip()
        input_data = []
        
        for _, row in df.iterrows():
            data_point = {
                'year': int(row['year']),
                'month': int(row['month']),
                'day': int(row['day']),
                'hour': int(row['hour']),
            }
            
            # Add optional forecast fields (case-insensitive)
            optional_fields = {
                'O3_forecast': ['O3_forecast', 'o3_forecast', 'O3', 'o3'],
                'NO2_forecast': ['NO2_forecast', 'no2_forecast', 'NO2', 'no2'],
                'T_forecast': ['T_forecast', 't_forecast', 'temperature', 'Temperature'],
                'q_forecast': ['q_forecast', 'Q_forecast', 'humidity', 'Humidity'],
                'u_forecast': ['u_forecast', 'U_forecast', 'u_wind', 'U_wind'],
                'v_forecast': ['v_forecast', 'V_forecast', 'v_wind', 'V_wind'],
                'w_forecast': ['w_forecast', 'W_forecast', 'w_wind', 'W_wind'],
                'blh_forecast': ['blh_forecast', 'BLH_forecast', 'blh', 'BLH'],
                'NO2_satellite': ['NO2_satellite', 'no2_satellite'],
                'HCHO_satellite': ['HCHO_satellite', 'hcho_satellite'],
                'ratio_satellite': ['ratio_satellite', 'Ratio_satellite'],
                'SZA_deg': ['SZA_deg', 'sza_deg', 'SZA', 'sza'],
            }
            
            for field_name, possible_names in optional_fields.items():
                for name in possible_names:
                    if name in df.columns:
                        value = row[name]
                        if pd.notna(value):
                            data_point[field_name] = float(value)
                        break
            
            input_data.append(data_point)
        
        if len(input_data) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="CSV file contains no valid data rows"
            )
        
        # Validate input data
        is_valid, error_msg = prediction_service.validate_input_data(input_data, site_id=site_id)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )
        
        # Limit forecast hours if needed
        # If forecast_hours is provided, limit to that; otherwise use all rows (but cap at 48 for safety)
        if forecast_hours is not None:
            actual_forecast_hours = forecast_hours
            if len(input_data) > actual_forecast_hours:
                input_data = input_data[:actual_forecast_hours]
        else:
            # If no forecast_hours specified, use all rows but warn if > 48
            actual_forecast_hours = len(input_data)
            if actual_forecast_hours > 48:
                logger.warning(f"CSV contains {actual_forecast_hours} rows, which exceeds recommended 48 hours. Processing all rows.")
        
        # Make predictions
        predictions = prediction_service.predict_from_dict(
            input_data=input_data,
            site_id=site_id,
            forecast_hours=actual_forecast_hours
        )
        
        return PredictResponse(
            success=True,
            site_id=site_id,
            forecast_hours=len(predictions),
            predictions=predictions,
            message=f"Successfully generated {len(predictions)} predictions from CSV file for site {site_id}"
        )
        
    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model for site {site_id} not found. Please ensure models are trained and available."
        )
    except Exception as e:
        logger.error(f"Error processing CSV file: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing CSV file: {str(e)}"
        )


@router.post(
    "/unified",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict air pollution using unified model",
    description="Make predictions for O3 and NO2 using the unified model (requires site_id in input data)"
)
async def predict_unified(
    request: PredictRequest
):
    """
    Predict air pollution using unified model
    
    Args:
        request: Prediction request with input data (must include site_id) and forecast hours
        
    Returns:
        PredictResponse with predictions
    """
    try:
        # Convert input data to list of dicts
        input_data = [item.dict() for item in request.input_data]
        
        # Validate input data (unified model requires site_id)
        is_valid, error_msg = prediction_service.validate_input_data(input_data, site_id=None)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )
        
        # Make predictions
        predictions = prediction_service.predict_from_dict(
            input_data=input_data,
            site_id=None,  # None means unified model
            forecast_hours=request.forecast_hours
        )
        
        return PredictResponse(
            success=True,
            site_id=None,
            forecast_hours=request.forecast_hours,
            predictions=predictions,
            message=f"Successfully generated {len(predictions)} predictions using unified model"
        )
        
    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.error(f"Unified model file not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Unified model not found. Please ensure models are trained and available."
        )
    except Exception as e:
        logger.error(f"Error making predictions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making predictions: {str(e)}"
        )


@router.post(
    "/",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict air pollution (auto-detect site or unified)",
    description="Make predictions using site_id from request or unified model if site_id is None"
)
async def predict(
    request: PredictRequest
):
    """
    Predict air pollution (auto-detect model based on site_id)
    
    Args:
        request: Prediction request with input data and forecast hours
        
    Returns:
        PredictResponse with predictions
    """
    try:
        # Convert input data to list of dicts
        input_data = [item.dict() for item in request.input_data]
        
        # Determine which model to use
        site_id = request.site_id
        
        # If site_id is None, check if it's in input data (for unified model)
        if site_id is None:
            if input_data and 'site_id' in input_data[0]:
                # Use unified model
                site_id = None
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Either site_id must be provided in request or in input_data for unified model"
                )
        
        # Validate input data
        is_valid, error_msg = prediction_service.validate_input_data(input_data, site_id=site_id)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )
        
        # Make predictions
        predictions = prediction_service.predict_from_dict(
            input_data=input_data,
            site_id=site_id,
            forecast_hours=request.forecast_hours
        )
        
        model_type = f"site {site_id}" if site_id else "unified"
        return PredictResponse(
            success=True,
            site_id=site_id,
            forecast_hours=request.forecast_hours,
            predictions=predictions,
            message=f"Successfully generated {len(predictions)} predictions using {model_type} model"
        )
        
    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found. Please ensure models are trained and available."
        )
    except Exception as e:
        logger.error(f"Error making predictions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making predictions: {str(e)}"
        )
