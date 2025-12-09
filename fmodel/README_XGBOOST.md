# XGBoost Model Training for O3 and NO2 Prediction

This directory contains scripts to train XGBoost models for predicting `O3_target` and `NO2_target` using data from all 7 sites.

## Files

- `train_xgboost.py` - Main training script
- `predict_xgboost.py` - Prediction script for unseen data
- `requirements_xgboost.txt` - Required Python packages

## Installation

1. Install required packages:
```bash
pip install -r requirements_xgboost.txt
```

## Usage

### Training

To train the models on all 7 sites' training data:

```bash
python train_xgboost.py
```

This will:
- Load training data from all 7 sites
- Train separate XGBoost models for O3 and NO2
- Save models to `models/xgboost/`
- Generate feature importance reports
- Display training and validation metrics

### Prediction

To make predictions on unseen data:

```bash
python predict_xgboost.py
```

This will:
- Load the latest trained models
- Make predictions on unseen data for all 7 sites
- Save predictions to `predictions/xgboost/`

## Model Outputs

After training, the following files are created in `models/xgboost/`:

- `O3_model_YYYYMMDD_HHMMSS.pkl` - Trained O3 model
- `NO2_model_YYYYMMDD_HHMMSS.pkl` - Trained NO2 model
- `O3_metadata_YYYYMMDD_HHMMSS.json` - Model metadata and metrics
- `NO2_metadata_YYYYMMDD_HHMMSS.json` - Model metadata and metrics
- `O3_importance_YYYYMMDD_HHMMSS.csv` - Feature importance rankings
- `NO2_importance_YYYYMMDD_HHMMSS.csv` - Feature importance rankings

## Model Configuration

The models use the following hyperparameters (can be modified in `train_xgboost.py`):

**O3 Model:**
- n_estimators: 500
- max_depth: 8
- learning_rate: 0.05
- subsample: 0.8
- colsample_bytree: 0.8
- Early stopping: 50 rounds

**NO2 Model:**
- n_estimators: 500
- max_depth: 8
- learning_rate: 0.05
- subsample: 0.8
- colsample_bytree: 0.8
- Early stopping: 50 rounds

## Data Requirements

The training script expects data files in:
```
Data_SIH_2025_with_blh/Data_SIH_2025_with_blh/with_satellite/
  - site_1_train_data_with_satellite.csv
  - site_2_train_data_with_satellite.csv
  - ...
  - site_7_train_data_with_satellite.csv
```

The prediction script expects unseen data files:
```
Data_SIH_2025_with_blh/Data_SIH_2025_with_blh/with_satellite/
  - site_1_unseen_input_data_with_satellite.csv
  - site_2_unseen_input_data_with_satellite.csv
  - ...
  - site_7_unseen_input_data_with_satellite.csv
```

## Features

The models use all available features except:
- Target variables (O3_target, NO2_target)
- Time identifiers (year, month, day, hour)

Total features: ~30+ including:
- Forecast variables (O3_forecast, NO2_forecast, T_forecast, etc.)
- Satellite data (ratio_satellite, NO2_satellite_filled, HCHO_satellite_filled)
- BLH features (NO2_forecast_per_blh, O3_forecast_per_blh, etc.)
- Time features (sin_hour, cos_hour)
- Solar features (SZA_deg, cosSZA, solar_elevation, etc.)
- Wind features (wind_speed, wind_dir_deg, wind_sector)

## Evaluation Metrics

The models are evaluated using:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)

Metrics are calculated for both training and validation sets.

