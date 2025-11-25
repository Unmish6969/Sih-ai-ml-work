# Air Pollution Forecasting API Usage Guide

## Overview

This API provides endpoints for short-term forecasting (24-48 hours) of surface O3 and NO2 air pollution concentrations for Delhi monitoring sites.

## Endpoints

### 1. Health Check
- `GET /api/v1/health/` - Basic health check
- `GET /api/v1/health/models` - Check which models are available
- `GET /api/v1/health/models/{site_id}` - Check specific model details

### 2. Predictions

#### Predict for Specific Site
- `POST /api/v1/predict/site/{site_id}`
  - `site_id`: 1-7 (monitoring site number)
  - Request body: See PredictRequest schema below

#### Live Prediction using WAQI feed
- `GET /api/v1/predict/site/{site_id}/live`
  - `site_id`: 1-7 (monitoring site number)
  - Fetches the latest WAQI observation for that site's coordinates,
    maps the available fields (`year`, `month`, `day`, `hour`, NO₂/O₃/T/humidity/wind)
    into the model features, and runs the trained site model. Returns a single-hour prediction plus
    metadata about the source station/measurement.

#### Predict using Unified Model
- `POST /api/v1/predict/unified`
  - Request body: Must include `site_id` in input_data for unified model

#### Auto-detect Model
- `POST /api/v1/predict/`
  - Uses `site_id` from request or unified model if not specified

## Request Format

```json
{
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
      "NO2_satellite": null,
      "HCHO_satellite": null,
      "ratio_satellite": null,
      "NO2_sat_flag": 0.0,
      "HCHO_sat_flag": 0.0,
      "ratio_sat_flag": 0.0,
      "O3_target_lag1": null,
      "O3_target_lag24": null,
      "O3_target_lag168": null,
      "NO2_target_lag1": null,
      "NO2_target_lag24": null,
      "NO2_target_lag168": null
    }
  ],
  "site_id": 1,
  "forecast_hours": 24
}
```

## Response Format

```json
{
  "success": true,
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
  "message": "Successfully generated 24 predictions for site 1"
}
```

### Live Prediction Response Example

```json
{
  "success": true,
  "site_id": 1,
  "forecast_hours": 1,
  "predictions": [
    {
      "year": 2025,
      "month": 11,
      "day": 25,
      "hour": 13,
      "O3_target": 58.3,
      "NO2_target": 21.4
    }
  ],
  "message": "Live prediction generated using Satyawati College, Delhi, Delhi, India",
  "live_source": {
    "station_id": 10115,
    "station_name": "Satyawati College, Delhi, Delhi, India",
    "station_location": [28.69572, 77.181295],
    "measurement_time": "2025-11-25 13:00:00",
    "timezone": "+05:30",
    "overall_aqi": 217,
    "observed_no2": 13.7,
    "observed_o3": 42.8
  }
}
```

## Example Usage

### Using cURL

```bash
# Predict for site 1
curl -X POST "http://localhost:8000/api/v1/predict/site/1" \
  -H "Content-Type: application/json" \
  -d '{
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
        "w_forecast": -1.56
      }
    ],
    "forecast_hours": 24
  }'
```

### Using Python

```python
import requests

url = "http://localhost:8000/api/v1/predict/site/1"
payload = {
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
            "w_forecast": -1.56
        }
    ],
    "forecast_hours": 24
}

response = requests.post(url, json=payload)
print(response.json())
```

## Notes

1. **Lag Features**: If lag features (O3_target_lag1, etc.) are not provided, they will be automatically created from forecast values.

2. **Site IDs**: 
   - Site 1-7: Individual site models
   - Unified model: Requires `site_id` in input_data

3. **Forecast Hours**: Must be 24 or 48

4. **Required Fields**: 
   - `year`, `month`, `day`, `hour` are required
   - Other fields can be null/omitted (will be filled with 0)

5. **API Documentation**: Visit `http://localhost:8000/docs` for interactive API documentation


