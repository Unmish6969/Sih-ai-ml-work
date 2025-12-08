"""
Data Loader for TCN-GNN Model

Handles loading and preprocessing of:
1. Hourly sequence features (time-series, last 24-48 hours)
2. Daily satellite features (once per day)
3. Static site features (latitude, longitude)
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class AirQualityDataset(Dataset):
    """
    Dataset for air quality prediction with TCN-GNN model
    
    Handles:
    - Hourly sequence features (24-48 hour windows)
    - Daily satellite features (extracted once per day)
    - Static site features (latitude, longitude)
    """
    
    def __init__(
        self,
        data_dir: Path,
        site_ids: List[int],
        seq_length: int = 24,
        prediction_horizon: int = 1,
        mode: str = 'train',
        scaler_hourly: StandardScaler = None,
        scaler_daily: StandardScaler = None,
        scaler_static: StandardScaler = None
    ):
        """
        Args:
            data_dir: Directory containing CSV files
            site_ids: List of site IDs to include (e.g., [1, 2, 3, 4, 5, 6, 7])
            seq_length: Length of hourly sequence window (default 24 hours)
            prediction_horizon: Hours ahead to predict (default 1)
            mode: 'train' or 'test'
            scaler_hourly: Pre-fitted scaler for hourly features (if None, fit new)
            scaler_daily: Pre-fitted scaler for daily features (if None, fit new)
            scaler_static: Pre-fitted scaler for static features (if None, fit new)
        """
        self.data_dir = Path(data_dir)
        self.site_ids = site_ids
        self.seq_length = seq_length
        self.prediction_horizon = prediction_horizon
        self.mode = mode
        
        # Site coordinates (latitude, longitude)
        self.site_coords = {
            1: (28.69536, 77.18168),
            2: (28.5718, 77.07125),
            3: (28.58278, 77.23441),
            4: (28.82286, 77.10197),
            5: (28.53077, 77.27123),
            6: (28.72954, 77.09601),
            7: (28.71052, 77.24951),
        }
        
        # Define feature groups
        self.hourly_features = [
            'NO2_forecast', 'O3_forecast', 'T_forecast',
            'wind_speed', 'wind_dir_deg', 'blh_forecast',
            'sin_hour', 'cos_hour',
            'NO2_forecast_per_blh', 'O3_forecast_per_blh',
            'blh_delta_1h', 'blh_rel_change',
            'cosSZA', 'solar_elevation', 'sunset_flag'
        ]
        
        self.daily_satellite_features = [
            'NO2_satellite_filled',
            'HCHO_satellite_filled',
            'ratio_satellite'
        ]
        
        self.static_features = ['latitude', 'longitude']
        
        # Load and preprocess data
        self.data = self._load_data()
        self.samples = self._create_samples()
        
        # Fit or use provided scalers
        if scaler_hourly is None:
            self.scaler_hourly = StandardScaler()
            self.scaler_daily = StandardScaler()
            self.scaler_static = StandardScaler()
            self._fit_scalers()
        else:
            self.scaler_hourly = scaler_hourly
            self.scaler_daily = scaler_daily
            self.scaler_static = scaler_static
    
    def _load_data(self) -> Dict[int, pd.DataFrame]:
        """Load data for all sites"""
        data = {}
        for site_id in self.site_ids:
            if self.mode == 'train':
                filename = f'site_{site_id}_train_data_with_satellite.csv'
            else:
                filename = f'site_{site_id}_unseen_input_data_with_satellite.csv'
            
            filepath = self.data_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(f"Data file not found: {filepath}")
            
            df = pd.read_csv(filepath)
            
            # Sort by date/time
            df = df.sort_values(['year', 'month', 'day', 'hour']).reset_index(drop=True)
            
            # Add static features
            lat, lon = self.site_coords[site_id]
            df['latitude'] = lat
            df['longitude'] = lon
            
            # Ensure all required columns exist
            missing_cols = set(self.hourly_features + self.daily_satellite_features) - set(df.columns)
            if missing_cols:
                print(f"Warning: Missing columns in site {site_id}: {missing_cols}")
                # Fill missing columns with zeros
                for col in missing_cols:
                    df[col] = 0.0
            
            data[site_id] = df
        
        return data
    
    def _create_samples(self) -> List[Dict]:
        """Create samples with sequences (handles cross-day sequences)"""
        samples = []
        
        for site_id, df in self.data.items():
            # Sort by date/time
            df = df.sort_values(['year', 'month', 'day', 'hour']).reset_index(drop=True)
            df['date'] = pd.to_datetime(df[['year', 'month', 'day']].astype(int))
            
            # Create sequences across the entire dataset
            for i in range(len(df)):
                # Skip if not enough history
                if i < self.seq_length - 1:
                    continue
                
                # Get sequence window (may span multiple days)
                start_idx = i - self.seq_length + 1
                seq_data = df.iloc[start_idx:i+1].copy()
                
                # Extract daily satellite features (use current day's values)
                current_date = df.iloc[i]['date']
                current_day_data = df[df['date'] == current_date]
                if len(current_day_data) > 0:
                    # Use first hour of current day for daily satellite features
                    daily_satellite = current_day_data[self.daily_satellite_features].iloc[0].values
                else:
                    # Fallback: use last available daily satellite values
                    daily_satellite = seq_data[self.daily_satellite_features].iloc[-1].values
                
                # Extract static features
                static = df.iloc[i][self.static_features].values
                
                # Extract hourly sequence
                hourly_seq = seq_data[self.hourly_features].values
                
                # Ensure sequence length matches (should always be seq_length)
                if len(hourly_seq) < self.seq_length:
                    # Pad with zeros if needed (shouldn't happen, but safety check)
                    padding = np.zeros((self.seq_length - len(hourly_seq), len(self.hourly_features)))
                    hourly_seq = np.vstack([padding, hourly_seq])
                
                # Get target (if available)
                if 'NO2_target' in df.columns and 'O3_target' in df.columns:
                    if i + self.prediction_horizon < len(df):
                        target_idx = i + self.prediction_horizon
                        target = df[['NO2_target', 'O3_target']].iloc[target_idx].values
                    else:
                        target = np.array([np.nan, np.nan])
                else:
                    target = np.array([np.nan, np.nan])
                
                samples.append({
                    'site_id': site_id,
                    'hourly_seq': hourly_seq,
                    'daily_satellite': daily_satellite,
                    'static': static,
                    'target': target,
                    'date': current_date,
                    'hour': df.iloc[i]['hour']
                })
        
        return samples
    
    def _fit_scalers(self):
        """Fit scalers on training data"""
        hourly_data = []
        daily_data = []
        static_data = []
        
        for sample in self.samples:
            hourly_data.append(sample['hourly_seq'])
            daily_data.append(sample['daily_satellite'])
            static_data.append(sample['static'])
        
        hourly_data = np.vstack(hourly_data)
        daily_data = np.vstack(daily_data)
        static_data = np.vstack(static_data)
        
        self.scaler_hourly.fit(hourly_data.reshape(-1, hourly_data.shape[-1]))
        self.scaler_daily.fit(daily_data)
        self.scaler_static.fit(static_data)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Scale features
        hourly_seq = sample['hourly_seq']
        hourly_seq_scaled = self.scaler_hourly.transform(
            hourly_seq.reshape(-1, hourly_seq.shape[-1])
        ).reshape(hourly_seq.shape)
        
        daily_satellite_scaled = self.scaler_daily.transform(
            sample['daily_satellite'].reshape(1, -1)
        ).flatten()
        
        static_scaled = self.scaler_static.transform(
            sample['static'].reshape(1, -1)
        ).flatten()
        
        # Convert to tensors
        hourly_seq_tensor = torch.FloatTensor(hourly_seq_scaled)
        daily_satellite_tensor = torch.FloatTensor(daily_satellite_scaled)
        static_tensor = torch.FloatTensor(static_scaled)
        
        # Target
        if not np.isnan(sample['target']).any():
            target_tensor = torch.FloatTensor(sample['target'])
        else:
            target_tensor = torch.FloatTensor([0.0, 0.0])  # Placeholder for test data
        
        return {
            'hourly_seq': hourly_seq_tensor,
            'daily_satellite': daily_satellite_tensor,
            'static': static_tensor,
            'target': target_tensor,
            'site_id': sample['site_id']
        }


def collate_fn(batch):
    """
    Custom collate function for batching samples
    
    Handles batching of sequences and creates graph structure.
    Creates edges between all nodes in the batch (fully connected within batch).
    The GNN will learn to weight these connections based on node features.
    """
    hourly_seqs = [item['hourly_seq'] for item in batch]
    daily_satellites = [item['daily_satellite'] for item in batch]
    statics = [item['static'] for item in batch]
    targets = [item['target'] for item in batch]
    site_ids = [item['site_id'] for item in batch]
    
    # Stack tensors
    hourly_seqs = torch.stack(hourly_seqs)
    daily_satellites = torch.stack(daily_satellites)
    statics = torch.stack(statics)
    targets = torch.stack(targets)
    
    # Create edge index for graph (fully connected within batch)
    # The GAT attention mechanism will learn which connections are important
    n_nodes = len(batch)
    edge_index = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                edge_index.append([i, j])
    
    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    return {
        'hourly_seq': hourly_seqs,
        'daily_satellite': daily_satellites,
        'static': statics,
        'target': targets,
        'edge_index': edge_index,
        'site_ids': site_ids
    }


def create_data_loaders(
    data_dir: Path,
    site_ids: List[int],
    seq_length: int = 24,
    prediction_horizon: int = 1,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.1,
    use_time_split: bool = True
):
    """
    Create train, validation, and test data loaders
    
    Args:
        data_dir: Directory containing CSV files
        site_ids: List of site IDs
        seq_length: Length of sequence window
        prediction_horizon: Hours ahead to predict
        batch_size: Batch size
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        use_time_split: If True, split by time (blocked time-series split)
    """
    # Create full dataset
    full_dataset = AirQualityDataset(
        data_dir=data_dir,
        site_ids=site_ids,
        seq_length=seq_length,
        prediction_horizon=prediction_horizon,
        mode='train'
    )
    
    # Split dataset
    n_samples = len(full_dataset)
    
    if use_time_split:
        # Blocked time-series split: train on older dates, validate on newer dates
        # Sort samples by date
        sorted_indices = sorted(
            range(n_samples),
            key=lambda i: (full_dataset.samples[i]['date'], full_dataset.samples[i]['hour'])
        )
        
        train_end = int(n_samples * train_split)
        val_end = int(n_samples * (train_split + val_split))
        
        train_indices = sorted_indices[:train_end]
        val_indices = sorted_indices[train_end:val_end]
        test_indices = sorted_indices[val_end:]
    else:
        # Random split
        indices = np.random.permutation(n_samples)
        train_end = int(n_samples * train_split)
        val_end = int(n_samples * (train_split + val_split))
        
        train_indices = indices[:train_end].tolist()
        val_indices = indices[train_end:val_end].tolist()
        test_indices = indices[val_end:].tolist()
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    
    return train_loader, val_loader, test_loader, full_dataset.scaler_hourly, full_dataset.scaler_daily, full_dataset.scaler_static
