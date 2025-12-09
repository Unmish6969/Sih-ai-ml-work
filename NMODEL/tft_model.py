"""
Temporal Fusion Transformer (TFT) model implementation for air quality prediction.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesDataset(Dataset):
    """Dataset for time-series data."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, 
                 static_features: Optional[np.ndarray] = None,
                 known_future: Optional[np.ndarray] = None,
                 sample_weights: Optional[np.ndarray] = None,
                 input_window: int = 72,
                 output_horizon: int = 24):
        """
        Initialize dataset.
        
        Args:
            X: Observed features (n_samples, n_features)
            y: Target values (n_samples,)
            static_features: Static features (n_samples, n_static)
            known_future: Known future covariates (n_samples, n_known_future)
            sample_weights: Sample weights (n_samples,)
            input_window: Input sequence length
            output_horizon: Output prediction horizon
        """
        self.X = X
        self.y = y
        self.static_features = static_features
        self.known_future = known_future
        self.sample_weights = sample_weights
        self.input_window = input_window
        self.output_horizon = output_horizon
        
        # Calculate number of samples
        # We need at least input_window samples for input, and output_horizon samples for target
        # For single-step (output_horizon=1), we need input_window + 1 total samples
        # For idx=0, we use X[0:input_window] and need y[input_window] for target
        # So we need len(X) > input_window (at least input_window + 1 elements)
        if len(X) <= input_window:
            self.n_samples = 0
        else:
            # Maximum idx such that idx + input_window < len(X) (for accessing y[end_idx])
            # So idx < len(X) - input_window
            # Valid idx range: 0 to len(X) - input_window - 1 (inclusive)
            # Number of valid indices: len(X) - input_window
            # But we also need output_horizon samples after end_idx
            # So we need: idx + input_window + output_horizon <= len(X)
            # Which means: idx <= len(X) - input_window - output_horizon
            # Valid idx range: 0 to len(X) - input_window - output_horizon (inclusive)
            # Number of valid indices: len(X) - input_window - output_horizon + 1
            self.n_samples = max(0, len(X) - input_window - output_horizon + 1)
    
    def __len__(self):
        return max(0, self.n_samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Returns:
            X_seq: (input_window, n_features) - input sequence
            y_seq: (input_window,) - historical target sequence
            y_target: (output_horizon,) - target to predict
            static: static features
            known_future: (output_horizon, n_known_features) - known future covariates
            weight: sample weight
        """
        # Compute start/end indices for input window
        start_idx = idx
        end_idx = idx + self.input_window  # exclusive
        
        # Input sequence - always return exactly input_window samples
        if end_idx <= len(self.X):
            # Normal case: enough data available
            X_seq = self.X[start_idx:end_idx].copy()
            y_seq = self.y[start_idx:end_idx].copy()
        else:
            # Not enough data - pad with zeros at the beginning
            available = self.X[start_idx:len(self.X)].copy()
            y_available = self.y[start_idx:len(self.y)].copy()
            pad_size = end_idx - len(self.X)
            
            # Pad with zeros matching the dtype and shape
            X_pad = np.zeros((pad_size, self.X.shape[1]), dtype=self.X.dtype)
            y_pad = np.zeros(pad_size, dtype=self.y.dtype)
            
            X_seq = np.vstack([X_pad, available])
            y_seq = np.concatenate([y_pad, y_available])
        
        # Ensure exactly input_window samples (safety check)
        if len(X_seq) != self.input_window:
            raise ValueError(f"X_seq length {len(X_seq)} != input_window {self.input_window}")
        if len(y_seq) != self.input_window:
            raise ValueError(f"y_seq length {len(y_seq)} != input_window {self.input_window}")
        
        # Target: prediction target starts at end_idx and spans output_horizon
        target_start = end_idx
        target_end = target_start + self.output_horizon
        
        # Target: extract target values for prediction
        if target_end <= len(self.y):
            # Normal case: enough target data available
            y_target = self.y[target_start:target_end].copy()
        else:
            # Not enough target data - pad with last value (or zeros)
            if target_start < len(self.y):
                available = self.y[target_start:len(self.y)].copy()
                pad_size = target_end - len(self.y)
                # Pad with last available value (or zero if no data)
                pad_value = self.y[-1] if len(self.y) > 0 else 0.0
                y_pad = np.full(pad_size, pad_value, dtype=self.y.dtype)
                y_target = np.concatenate([available, y_pad])
            else:
                # No target data available - use zeros
                y_target = np.zeros(self.output_horizon, dtype=self.y.dtype)
        
        # Ensure exactly output_horizon target values
        if len(y_target) != self.output_horizon:
            # Truncate or pad to match output_horizon
            if len(y_target) < self.output_horizon:
                pad_size = self.output_horizon - len(y_target)
                pad_value = y_target[-1] if len(y_target) > 0 else 0.0
                y_pad = np.full(pad_size, pad_value, dtype=y_target.dtype)
                y_target = np.concatenate([y_target, y_pad])
            else:
                y_target = y_target[:self.output_horizon]
        
        # Static features (use the value at end_idx - 1, or last available)
        if self.static_features is not None:
            static_idx = min(end_idx - 1, len(self.static_features) - 1) if end_idx > 0 else 0
            static = self.static_features[static_idx]
            # Ensure it's a 1D array
            if isinstance(static, (int, float, np.integer, np.floating)):
                static = np.array([static])
            elif len(static.shape) == 0:
                static = np.array([static])
            elif len(static.shape) > 1:
                static = static.flatten()
        else:
            static = np.array([])
        
        # Known future (for the prediction period)
        # Always return shape (output_horizon, n_known_features)
        if self.known_future is not None and len(self.known_future) > 0:
            if target_start < len(self.known_future) and target_end <= len(self.known_future):
                known = self.known_future[target_start:target_end]
                # Ensure 2D shape: (output_horizon, n_features)
                if len(known.shape) == 1:
                    # If it's 1D, reshape to (output_horizon, 1)
                    known = known.reshape(-1, 1)
                elif len(known.shape) == 0:
                    # Scalar case
                    known = np.array([[known]])
                # If already 2D, ensure correct shape
                if len(known.shape) == 2 and known.shape[0] != self.output_horizon:
                    # Pad or truncate to output_horizon
                    if known.shape[0] < self.output_horizon:
                        pad_size = self.output_horizon - known.shape[0]
                        known = np.vstack([known, np.zeros((pad_size, known.shape[1]))])
                    else:
                        known = known[:self.output_horizon]
            else:
                # Not enough data, use zeros with correct shape
                n_features = self.known_future.shape[1] if len(self.known_future.shape) > 1 else 1
                known = np.zeros((self.output_horizon, n_features))
        else:
            # No known future features - return zeros with shape (output_horizon, 0)
            known = np.zeros((self.output_horizon, 0))
        
        # Sample weight (safe indexing - use end_idx - 1)
        if self.sample_weights is not None:
            weight_idx = min(end_idx - 1, len(self.sample_weights) - 1) if end_idx > 0 else 0
            weight = self.sample_weights[weight_idx]
        else:
            weight = 1.0
        
        # Convert to tensors, handling empty arrays
        static_tensor = torch.FloatTensor(static) if len(static) > 0 else torch.FloatTensor([0.0])
        
        # Known future tensor - always shape (output_horizon, n_features)
        if known.shape[1] > 0:
            known_tensor = torch.FloatTensor(known)
        else:
            # Zero features case - return zeros with shape (output_horizon, 0)
            known_tensor = torch.zeros(self.output_horizon, 0)
        
        return {
            'X': torch.FloatTensor(X_seq),
            'y_history': torch.FloatTensor(y_seq),
            'static': static_tensor,
            'known_future': known_tensor,
            'y_target': torch.FloatTensor(y_target),
            'weight': torch.FloatTensor([weight])
        }


class TemporalFusionTransformer(nn.Module):
    """
    Simplified Temporal Fusion Transformer for time-series forecasting.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 96, 
                 num_layers: int = 2, num_heads: int = 4,
                 dropout: float = 0.1, output_horizon: int = 24,
                 static_size: int = 0, known_future_size: int = 0):
        """
        Initialize TFT model.
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden dimension
            num_layers: Number of LSTM/attention layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            output_horizon: Prediction horizon
            static_size: Number of static features
            known_future_size: Number of known future covariates
        """
        super(TemporalFusionTransformer, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_horizon = output_horizon
        self.static_size = static_size
        self.known_future_size = known_future_size
        
        # Input embedding
        self.input_embedding = nn.Linear(input_size, hidden_size)
        
        # Historical target embedding (fix: define in __init__, not in forward)
        self.y_embed = nn.Linear(1, hidden_size)
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, 
            num_layers=num_layers, 
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        
        # Static feature processing
        if static_size > 0:
            self.static_encoder = nn.Linear(static_size, hidden_size)
        
        # Known future processing
        if known_future_size > 0:
            self.known_future_encoder = nn.Linear(known_future_size, hidden_size)
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_horizon)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights to prevent NaN
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights to prevent NaN/Inf issues."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization with moderate gain (0.1 was too small)
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LSTM):
                # Initialize LSTM weights
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data, gain=0.5)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data, gain=0.5)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0.0)
                        # Set forget gate bias to 1 (helps with gradient flow)
                        n = param.size(0)
                        start, end = n // 4, n // 2
                        param.data[start:end].fill_(1.0)
    
    def forward(self, X, y_history, static=None, known_future=None):
        """
        Forward pass.
        
        Args:
            X: Input features (batch, seq_len, input_size)
            y_history: Historical target values (batch, seq_len)
            static: Static features (batch, static_size)
            known_future: Known future covariates (batch, output_horizon, known_future_size)
            
        Returns:
            Predictions (batch, output_horizon)
        """
        batch_size = X.size(0)
        
        # Validate inputs for NaN/Inf
        if torch.any(torch.isnan(X)) or torch.any(torch.isinf(X)):
            raise ValueError("Input X contains NaN/Inf values")
        if torch.any(torch.isnan(y_history)) or torch.any(torch.isinf(y_history)):
            raise ValueError("Input y_history contains NaN/Inf values")
        
        # Embed input features
        X_emb = self.input_embedding(X)  # (batch, seq_len, hidden_size)
        X_emb = self.dropout(X_emb)
        
        # Check for NaN after embedding
        if torch.any(torch.isnan(X_emb)) or torch.any(torch.isinf(X_emb)):
            raise ValueError("X_emb contains NaN/Inf after embedding")
        
        # Add historical target as additional feature (fix: use self.y_embed defined in __init__)
        y_emb = y_history.unsqueeze(-1)  # (batch, seq_len, 1)
        y_emb = self.y_embed(y_emb)  # (batch, seq_len, hidden_size)
        
        # Check for NaN after y embedding
        if torch.any(torch.isnan(y_emb)) or torch.any(torch.isinf(y_emb)):
            raise ValueError("y_emb contains NaN/Inf after embedding")
        
        X_emb = X_emb + y_emb
        
        # Check for NaN after addition
        if torch.any(torch.isnan(X_emb)) or torch.any(torch.isinf(X_emb)):
            raise ValueError("X_emb contains NaN/Inf after adding y_emb")
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(X_emb)
        
        # Check for NaN after LSTM
        if torch.any(torch.isnan(lstm_out)) or torch.any(torch.isinf(lstm_out)):
            raise ValueError("lstm_out contains NaN/Inf after LSTM")
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # Multi-head attention on sequence
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Check for NaN after attention
        if torch.any(torch.isnan(attn_out)) or torch.any(torch.isinf(attn_out)):
            raise ValueError("attn_out contains NaN/Inf after attention")
        
        attn_last = attn_out[:, -1, :]  # (batch, hidden_size)
        
        # Combine LSTM and attention outputs
        combined = torch.cat([last_hidden, attn_last], dim=1)  # (batch, hidden_size * 2)
        
        # Check for NaN after combining
        if torch.any(torch.isnan(combined)) or torch.any(torch.isinf(combined)):
            raise ValueError("combined contains NaN/Inf after concatenation")
        
        # Add static features if available
        if static is not None and self.static_size > 0:
            if torch.any(torch.isnan(static)) or torch.any(torch.isinf(static)):
                raise ValueError("static features contain NaN/Inf")
            static_emb = self.static_encoder(static)  # (batch, hidden_size)
            if torch.any(torch.isnan(static_emb)) or torch.any(torch.isinf(static_emb)):
                raise ValueError("static_emb contains NaN/Inf after encoding")
            # Expand static_emb to match combined size and concatenate
            static_expanded = torch.cat([static_emb, static_emb], dim=1)  # (batch, hidden_size * 2)
            combined = combined + static_expanded
            if torch.any(torch.isnan(combined)) or torch.any(torch.isinf(combined)):
                raise ValueError("combined contains NaN/Inf after adding static")
        
        # Add known future features if available
        if known_future is not None and self.known_future_size > 0:
            if torch.any(torch.isnan(known_future)) or torch.any(torch.isinf(known_future)):
                raise ValueError("known_future features contain NaN/Inf")
            known_emb = self.known_future_encoder(known_future)  # (batch, output_horizon, hidden_size)
            if torch.any(torch.isnan(known_emb)) or torch.any(torch.isinf(known_emb)):
                raise ValueError("known_emb contains NaN/Inf after encoding")
            # Average over horizon and expand to match combined size
            known_avg = known_emb.mean(dim=1)  # (batch, hidden_size)
            if torch.any(torch.isnan(known_avg)) or torch.any(torch.isinf(known_avg)):
                raise ValueError("known_avg contains NaN/Inf after averaging")
            known_expanded = torch.cat([known_avg, known_avg], dim=1)  # (batch, hidden_size * 2)
            combined = combined + known_expanded
            if torch.any(torch.isnan(combined)) or torch.any(torch.isinf(combined)):
                raise ValueError("combined contains NaN/Inf after adding known_future")
        
        # Output prediction
        output = self.output_layer(combined)  # (batch, output_horizon)
        
        # Final check
        if torch.any(torch.isnan(output)) or torch.any(torch.isinf(output)):
            raise ValueError("output contains NaN/Inf after output layer")
        
        return output


class TFTModel:
    """
    Temporal Fusion Transformer model wrapper.
    """
    
    def __init__(self, input_window: int = 72, output_horizon: int = 24,
                 hidden_size: int = 96, num_layers: int = 2, num_heads: int = 4,
                 dropout: float = 0.1, batch_size: int = 128, learning_rate: float = 1e-3,
                 epochs: int = 100, early_stopping_patience: int = 10,
                 device: Optional[str] = None):
        """
        Initialize TFT model.
        
        Args:
            input_window: Input sequence length
            output_horizon: Prediction horizon
            hidden_size: Hidden dimension
            num_layers: Number of LSTM layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            batch_size: Batch size
            learning_rate: Learning rate
            epochs: Maximum number of epochs
            early_stopping_patience: Early stopping patience
            device: Device ('cuda' or 'cpu')
        """
        self.input_window = input_window
        self.output_horizon = output_horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.static_scaler = StandardScaler()
        self.known_future_scaler = StandardScaler()
    
    def _prepare_data(self, X: pd.DataFrame, y: pd.Series,
                     static_features: Optional[pd.DataFrame] = None,
                     known_future: Optional[pd.DataFrame] = None,
                     sample_weights: Optional[pd.Series] = None,
                     fit_scalers: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare and scale data."""
        # Validate input data for NaN/Inf
        if X.isna().any().any():
            raise ValueError("Input features X contain NaN values")
        if np.isinf(X.values).any():
            raise ValueError("Input features X contain Inf values")
        if y.isna().any():
            raise ValueError("Target y contains NaN values")
        if np.isinf(y.values).any():
            raise ValueError("Target y contains Inf values")
        
        # Scale features
        if fit_scalers:
            X_scaled = self.scaler_X.fit_transform(X.values)
            y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
        else:
            X_scaled = self.scaler_X.transform(X.values)
            y_scaled = self.scaler_y.transform(y.values.reshape(-1, 1)).flatten()
        
        # Check for NaN/Inf after scaling
        if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
            raise ValueError("Scaled features X_scaled contain NaN/Inf values")
        if np.any(np.isnan(y_scaled)) or np.any(np.isinf(y_scaled)):
            raise ValueError("Scaled target y_scaled contains NaN/Inf values")
        
        # Scale static features
        static_scaled = None
        if static_features is not None and len(static_features.columns) > 0:
            # Validate static features
            if static_features.isna().any().any():
                raise ValueError("Static features contain NaN values")
            if np.isinf(static_features.values).any():
                raise ValueError("Static features contain Inf values")
            
            if fit_scalers:
                static_scaled = self.static_scaler.fit_transform(static_features.values)
            else:
                static_scaled = self.static_scaler.transform(static_features.values)
            
            # Check for NaN/Inf after scaling
            if np.any(np.isnan(static_scaled)) or np.any(np.isinf(static_scaled)):
                raise ValueError("Scaled static features contain NaN/Inf values")
        
        # Scale known future features
        known_scaled = None
        if known_future is not None and len(known_future.columns) > 0:
            # Validate known future features
            if known_future.isna().any().any():
                raise ValueError("Known future features contain NaN values")
            if np.isinf(known_future.values).any():
                raise ValueError("Known future features contain Inf values")
            
            if fit_scalers:
                known_scaled = self.known_future_scaler.fit_transform(known_future.values)
            else:
                known_scaled = self.known_future_scaler.transform(known_future.values)
            
            # Check for NaN/Inf after scaling
            if np.any(np.isnan(known_scaled)) or np.any(np.isinf(known_scaled)):
                raise ValueError("Scaled known future features contain NaN/Inf values")
        
        # Sample weights
        weights = sample_weights.values if sample_weights is not None else None
        
        return X_scaled, y_scaled, static_scaled, known_scaled, weights
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame, y_val: pd.Series,
            static_train: Optional[pd.DataFrame] = None,
            static_val: Optional[pd.DataFrame] = None,
            known_future_train: Optional[pd.DataFrame] = None,
            known_future_val: Optional[pd.DataFrame] = None,
            sample_weight_train: Optional[pd.Series] = None,
            sample_weight_val: Optional[pd.Series] = None) -> 'TFTModel':
        """
        Train the TFT model.
        
        Args:
            X_train: Training observed features
            y_train: Training target
            X_val: Validation observed features
            y_val: Validation target
            static_train: Training static features
            static_val: Validation static features
            known_future_train: Training known future covariates
            known_future_val: Validation known future covariates
            sample_weight_train: Training sample weights
            sample_weight_val: Validation sample weights
            
        Returns:
            Self
        """
        # Prepare training data
        X_train_scaled, y_train_scaled, static_train_scaled, known_train_scaled, weights_train = \
            self._prepare_data(X_train, y_train, static_train, known_future_train, 
                             sample_weight_train, fit_scalers=True)
        
        # Prepare validation data
        X_val_scaled, y_val_scaled, static_val_scaled, known_val_scaled, weights_val = \
            self._prepare_data(X_val, y_val, static_val, known_future_val,
                             sample_weight_val, fit_scalers=False)
        
        # Create datasets
        train_dataset = TimeSeriesDataset(
            X_train_scaled, y_train_scaled,
            static_train_scaled, known_train_scaled, weights_train,
            self.input_window, self.output_horizon
        )
        val_dataset = TimeSeriesDataset(
            X_val_scaled, y_val_scaled,
            static_val_scaled, known_val_scaled, weights_val,
            self.input_window, self.output_horizon
        )
        
        # Defensive check: skip if not enough samples for time-series window
        if len(train_dataset) == 0:
            raise ValueError(
                f"Not enough samples for TFT training: need at least "
                f"input_window={self.input_window} + output_horizon={self.output_horizon} samples, "
                f"but got {len(X_train_scaled)} samples"
            )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Initialize model
        static_size = static_train_scaled.shape[1] if static_train_scaled is not None else 0
        known_future_size = known_train_scaled.shape[1] if known_train_scaled is not None else 0
        
        self.model = TemporalFusionTransformer(
            input_size=X_train_scaled.shape[1],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            output_horizon=self.output_horizon,
            static_size=static_size,
            known_future_size=known_future_size
        ).to(self.device)
        
        # Optimizer and scheduler
        # Lower learning rate to prevent NaN (reduce from 1e-3 to 5e-4)
        effective_lr = min(self.learning_rate, 5e-4)  # Cap at 5e-4
        optimizer = torch.optim.Adam(self.model.parameters(), lr=effective_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        criterion = nn.L1Loss()  # MAE loss
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                
                X_batch = batch['X'].to(self.device)
                y_history = batch['y_history'].to(self.device)
                static = batch['static'].to(self.device) if static_train_scaled is not None else None
                known_future = batch['known_future'].to(self.device) if known_train_scaled is not None else None
                y_target = batch['y_target'].to(self.device)
                weights = batch['weight'].to(self.device)
                
                # Forward pass
                pred = self.model(X_batch, y_history, static, known_future)
                
                # Check for NaN in predictions before computing loss
                if torch.any(torch.isnan(pred)) or torch.any(torch.isinf(pred)):
                    print(f"Warning: NaN/Inf in model predictions at epoch {epoch + 1}, skipping batch")
                    continue
                
                # For single-step prediction
                if self.output_horizon == 1:
                    loss = criterion(pred.squeeze(), y_target.squeeze())
                else:
                    loss = criterion(pred, y_target)
                
                # Apply sample weights
                loss = (loss * weights.squeeze()).mean()
                
                # Check for NaN/Inf loss before backpropagation
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss detected at epoch {epoch + 1}, skipping batch")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                loss_value = loss.item()
                # Check for NaN in loss value
                if np.isnan(loss_value) or np.isinf(loss_value):
                    print(f"Warning: NaN/Inf in loss.item() at epoch {epoch + 1}, skipping batch")
                    continue
                
                train_loss += loss_value
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_batch_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    X_batch = batch['X'].to(self.device)
                    y_history = batch['y_history'].to(self.device)
                    static = batch['static'].to(self.device) if static_val_scaled is not None else None
                    known_future = batch['known_future'].to(self.device) if known_val_scaled is not None else None
                    y_target = batch['y_target'].to(self.device)
                    
                    pred = self.model(X_batch, y_history, static, known_future)
                    
                    # Check for NaN in validation predictions
                    if torch.any(torch.isnan(pred)) or torch.any(torch.isinf(pred)):
                        print(f"Warning: NaN/Inf in validation predictions at epoch {epoch + 1}, skipping batch")
                        continue
                    
                    if self.output_horizon == 1:
                        loss = criterion(pred.squeeze(), y_target.squeeze())
                    else:
                        loss = criterion(pred, y_target)
                    
                    # Check for NaN in validation loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: NaN/Inf in validation loss at epoch {epoch + 1}, skipping batch")
                        continue
                    
                    loss_value = loss.item()
                    if np.isnan(loss_value) or np.isinf(loss_value):
                        print(f"Warning: NaN/Inf in validation loss.item() at epoch {epoch + 1}, skipping batch")
                        continue
                    
                    val_loss += loss_value
                    val_batch_count += 1
            
            # Handle case where all validation batches were skipped
            if val_batch_count == 0:
                print(f"Error: All validation batches had NaN/Inf at epoch {epoch + 1}. Stopping training.")
                break
            
            # Check for NaN in accumulated losses
            if np.isnan(train_loss) or np.isinf(train_loss):
                print(f"Error: NaN/Inf in train_loss at epoch {epoch + 1}. Stopping training.")
                break
            
            if np.isnan(val_loss) or np.isinf(val_loss):
                print(f"Error: NaN/Inf in val_loss at epoch {epoch + 1}. Stopping training.")
                break
            
            train_loss /= len(train_loader)
            val_loss /= val_batch_count  # Use actual count, not len(val_loader) in case batches were skipped
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        return self
    
    def predict(self, X: pd.DataFrame, y_history: Optional[pd.Series] = None,
                static: Optional[pd.DataFrame] = None,
                known_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Observed features
            y_history: Historical target values
            static: Static features
            known_future: Known future covariates
            
        Returns:
            Predictions (unscaled)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Prepare data
        X_scaled, _, static_scaled, known_scaled, _ = \
            self._prepare_data(X, pd.Series([0] * len(X)), static, known_future, None, fit_scalers=False)
        
        # Use provided y_history or create dummy (use mean of training data)
        # CRITICAL: y_scaled must have the same length as X_scaled
        # X_scaled has length len(X) at this point
        if y_history is not None and len(y_history) > 0:
            y_scaled = self.scaler_y.transform(y_history.values.reshape(-1, 1)).flatten()
            # Ensure y_scaled has the same length as X_scaled (which is len(X))
            if len(y_scaled) != len(X_scaled):
                # If y_history is shorter, pad with last value; if longer, truncate
                if len(y_scaled) < len(X_scaled):
                    pad_size = len(X_scaled) - len(y_scaled)
                    last_val = y_scaled[-1] if len(y_scaled) > 0 else 0.0
                    y_scaled = np.concatenate([np.full(pad_size, last_val), y_scaled])
                else:
                    y_scaled = y_scaled[:len(X_scaled)]
        else:
            # Use zero (which corresponds to mean after scaling)
            y_scaled = np.zeros(len(X_scaled))
        
        # For prediction, we need to handle the case where we don't have enough history
        # We'll pad with zeros if needed
        if len(X_scaled) < self.input_window:
            # Pad with zeros at the beginning
            pad_size = self.input_window - len(X_scaled)
            X_scaled = np.vstack([np.zeros((pad_size, X_scaled.shape[1])), X_scaled])
            y_scaled = np.concatenate([np.zeros(pad_size), y_scaled])
            if static_scaled is not None:
                static_scaled = np.vstack([static_scaled[:1].repeat(pad_size, axis=0), static_scaled])
            if known_scaled is not None:
                known_scaled = np.vstack([known_scaled[:1].repeat(pad_size, axis=0), known_scaled])
        
        # Final safety check: ensure X_scaled and y_scaled have the same length
        if len(X_scaled) != len(y_scaled):
            raise ValueError(
                f"X_scaled and y_scaled must have the same length: "
                f"X_scaled={len(X_scaled)}, y_scaled={len(y_scaled)}, "
                f"X_len={len(X)}, y_history_len={len(y_history) if y_history is not None else 0}"
            )
        
        # Create dataset
        dataset = TimeSeriesDataset(
            X_scaled, y_scaled, static_scaled, known_scaled, None,
            self.input_window, self.output_horizon
        )
        
        if len(dataset) == 0:
            # If no valid sequences, return zeros
            return np.zeros(len(X))
        
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        # Predict
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in loader:
                X_batch = batch['X'].to(self.device)
                y_hist = batch['y_history'].to(self.device)
                static_batch = batch['static'].to(self.device) if static_scaled is not None else None
                known_batch = batch['known_future'].to(self.device) if known_scaled is not None else None
                
                pred = self.model(X_batch, y_hist, static_batch, known_batch)
                predictions.append(pred.cpu().numpy())
        
        if len(predictions) == 0:
            return np.zeros(len(X))
        
        predictions = np.concatenate(predictions, axis=0)
        
        # Unscale predictions
        if self.output_horizon == 1:
            predictions = self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
        else:
            # For multi-horizon, we'll return the first step prediction
            predictions = self.scaler_y.inverse_transform(predictions[:, 0].reshape(-1, 1)).flatten()
        
        # If we padded, we need to adjust the output length
        if len(predictions) != len(X):
            # For sequence models, we might have fewer predictions than input samples
            # Pad with the last prediction or repeat
            if len(predictions) < len(X):
                last_pred = predictions[-1] if len(predictions) > 0 else 0.0
                predictions = np.concatenate([predictions, np.full(len(X) - len(predictions), last_pred)])
            else:
                predictions = predictions[:len(X)]
        
        return predictions

