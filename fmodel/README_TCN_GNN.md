# TCN-GNN Model for Air Quality Prediction

This implementation provides a Temporal Convolutional Network (TCN) combined with Graph Neural Network (GNN) for predicting NO₂ and O₃ concentrations.

## Architecture Overview

The model consists of four main components:

1. **TCN Encoder**: Processes hourly time-series sequences (24-48 hours) to extract temporal patterns
   - Diurnal cycles
   - Photochemistry cycles
   - Lag effects
   - Forecast corrections

2. **Feature Combination Layer**: Combines temporal embeddings with:
   - Daily satellite features (NO₂_satellite, HCHO_satellite, ratio_satellite)
   - Static site features (latitude, longitude)

3. **GNN Layer**: Graph Attention Network (GAT) that learns spatial relationships between sites
   - Regional transport
   - Chemical coupling
   - Spatial smoothing
   - Satellite footprint influence

4. **MLP Prediction Head**: Maps GNN output to NO₂ and O₃ predictions

## Installation

Install required dependencies:

```bash
pip install -r requirements_tcn_gnn.txt
```

## Data Structure

The model expects data in the following format:

- **Hourly sequence features**: Time-series data with shape `[seq_length, num_hourly_features]`
  - Features include: NO₂_forecast, O₃_forecast, Temperature, Wind speed/direction, BLH, hour_sin, hour_cos, etc.
  
- **Daily satellite features**: Flat vectors (not repeated hourly)
  - NO₂_satellite_filled
  - HCHO_satellite_filled
  - ratio_satellite

- **Static site features**: Per-site constants
  - latitude
  - longitude

## Training

### Basic Usage

```bash
python train_tcn_gnn.py \
    --data_dir "Data_SIH_2025_with_blh/Data_SIH_2025_with_blh/with_satellite" \
    --site_ids 1 2 3 4 5 6 7 \
    --seq_length 24 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --num_epochs 100 \
    --loss_type mae \
    --patience 10 \
    --save_dir "models/tcn_gnn"
```

### Parameters

- `--data_dir`: Directory containing CSV files with satellite data
- `--site_ids`: List of site IDs to train on (default: all 7 sites)
- `--seq_length`: Length of sequence window in hours (default: 24)
- `--prediction_horizon`: Hours ahead to predict (default: 1)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate for AdamW optimizer (default: 1e-3)
- `--num_epochs`: Maximum number of epochs (default: 100)
- `--loss_type`: Loss function - 'mae' or 'huber' (default: 'mae')
- `--patience`: Early stopping patience (default: 10)
- `--save_dir`: Directory to save model and results

### Training Features

- **Blocked Time-Series Split**: Train on older dates, validate on newer dates
- **Early Stopping**: Stops training when validation loss stops improving
- **Learning Rate Scheduling**: Reduces learning rate on plateau
- **Gradient Clipping**: Prevents gradient explosion
- **Model Checkpointing**: Saves best model based on validation loss

## Model Outputs

After training, the following files are saved:

- `best_model.pt`: Best model weights and configuration
- `training_history.json`: Training/validation loss and metrics
- `training_curves.png`: Visualization of training progress
- `args.json`: Training arguments used

## Model Architecture Details

### TCN Encoder
- Input: `[batch, seq_len, hourly_feature_dim]`
- Output: `[batch, temporal_hidden_dim]` (default: 128)
- Layers: Initial Conv1D + 3 Dilated Conv1D layers with increasing dilation

### Feature Combiner
- Combines: Temporal embedding (128) + Daily satellite (3) + Static (2)
- Output: `[batch, gnn_hidden_dim]` (default: 128)

### GNN Layer
- Type: Graph Attention Network (GAT)
- Heads: 4 attention heads in first layer, 1 in second layer
- Output: `[batch, gnn_hidden_dim]` (default: 128)

### MLP Head
- Layers: Dense(128) → ReLU → Dense(64) → ReLU → Dense(2)
- Output: `[batch, 2]` → [NO₂_pred, O₃_pred]

## Data Flow

```
Hourly Data → TCN → Temporal Embedding
                                    ↓
Daily Satellite + Static Features → Feature Combiner → Node Embedding
                                                              ↓
                                                          GNN Layer
                                                              ↓
                                                          MLP Head
                                                              ↓
                                                    [NO₂_pred, O₃_pred]
```

## Why This Architecture Works

1. **TCN**: Handles hourly pollution & meteorology changes, captures temporal dependencies
2. **GNN**: Models spatial interactions and regional chemistry between sites
3. **Satellite Data**: Enhances chemical regime understanding
4. **MLP**: Performs final mapping to predictions

## Notes

- The model uses blocked time-series cross-validation to prevent data leakage
- Graph edges are created dynamically within each batch (fully connected)
- The GAT attention mechanism learns which spatial connections are important
- All features are standardized using StandardScaler

## Example Usage

See `train_tcn_gnn.py` for complete training implementation.

For inference, load the saved model:

```python
import torch
from tcn_gnn_model import TCNGNNModel

checkpoint = torch.load('models/tcn_gnn/best_model.pt')
model = TCNGNNModel(**checkpoint['model_config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```
