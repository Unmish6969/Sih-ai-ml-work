"""
Data Processing Pipeline for SIH PS-10
Processes all sites: Site 1 through Site 7
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def load_and_explore_data(file_path, site_name):
    """STEP 2: Load dataset and perform basic EDA"""
    print(f"\n{'='*60}")
    print(f"Loading {site_name}")
    print(f"{'='*60}")
    
    df = pd.read_csv(file_path)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    
    print(f"\nData Types:")
    print(df.dtypes)
    
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nBasic Statistics:")
    print(df.describe())
    
    return df

def create_datetime_column(df):
    """STEP 3: Create datetime column from year, month, day, hour"""
    print("\nCreating datetime column...")
    
    # Convert to integer first to remove decimals
    df['year'] = df['year'].astype(int)
    df['month'] = df['month'].astype(int)
    df['day'] = df['day'].astype(int)
    df['hour'] = df['hour'].astype(int)
    
    # Create datetime column
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    
    # Sort by datetime
    df = df.sort_values('datetime').reset_index(drop=True)
    
    print(f"Datetime range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    return df

def handle_satellite_data(df):
    """STEP 4: Handle daily satellite data - forward fill within each day"""
    print("\nHandling satellite data...")
    
    satellite_cols = ['NO2_satellite', 'HCHO_satellite', 'ratio_satellite']
    
    # Create date column for grouping
    df['date'] = df['datetime'].dt.date
    
    for col in satellite_cols:
        if col in df.columns:
            # Group by date and forward-fill within each day
            # Forward-fill and back-fill within each day group
            df[col] = df.groupby('date')[col].transform(lambda x: x.ffill().bfill())
            
            # Create flag column to indicate if satellite data is present
            flag_col = col.replace('_satellite', '_sat_flag')
            df[flag_col] = (~df[col].isnull()).astype(int)
            
            print(f"{col}: {df[col].notna().sum()} non-null values out of {len(df)}")
            print(f"{flag_col}: {df[flag_col].sum()} days with data")
    
    # Drop temporary date column
    df = df.drop('date', axis=1)
    
    return df

def add_lag_features(df):
    """STEP 5: Add lag features (1-hour, 24-hour, 168-hour lags)"""
    print("\nAdding lag features...")
    
    target_cols = ['O3_target', 'NO2_target']
    lag_periods = [1, 24, 168]  # 1 hour, 24 hours (1 day), 168 hours (7 days)
    
    for target_col in target_cols:
        if target_col in df.columns:
            for lag in lag_periods:
                lag_col = f"{target_col}_lag{lag}"
                df[lag_col] = df[target_col].shift(lag)
                print(f"Created {lag_col}: {df[lag_col].notna().sum()} non-null values")
    
    return df

def split_by_date(df, train_ratio=0.75):
    """STEP 6: Split data by date (not randomly) - 75% train, 25% test"""
    print("\nSplitting data by date...")
    
    # Get unique dates
    df['date'] = df['datetime'].dt.date
    unique_dates = sorted(df['date'].unique())
    
    print(f"Total unique dates: {len(unique_dates)}")
    print(f"Date range: {unique_dates[0]} to {unique_dates[-1]}")
    
    # Split dates
    split_idx = int(len(unique_dates) * train_ratio)
    train_dates = set(unique_dates[:split_idx])
    test_dates = set(unique_dates[split_idx:])
    
    sorted_train_dates = sorted(list(train_dates))
    sorted_test_dates = sorted(list(test_dates))
    
    print(f"Train dates: {len(train_dates)} ({sorted_train_dates[0] if sorted_train_dates else 'N/A'} to {sorted_train_dates[-1] if sorted_train_dates else 'N/A'})")
    print(f"Test dates: {len(test_dates)} ({sorted_test_dates[0] if sorted_test_dates else 'N/A'} to {sorted_test_dates[-1] if sorted_test_dates else 'N/A'})")
    
    # Create train and test masks
    train_mask = df['date'].isin(train_dates)
    test_mask = df['date'].isin(test_dates)
    
    train_df = df[train_mask].copy().reset_index(drop=True)
    test_df = df[test_mask].copy().reset_index(drop=True)
    
    print(f"\nTrain set: {len(train_df)} rows ({len(train_df)/len(df)*100:.2f}%)")
    print(f"Test set: {len(test_df)} rows ({len(test_df)/len(df)*100:.2f}%)")
    
    # Drop temporary date column
    train_df = train_df.drop('date', axis=1)
    test_df = test_df.drop('date', axis=1)
    
    return train_df, test_df

def process_site(site_num, data_dir='Data_SIH_2025'):
    """Process a single site through all steps"""
    site_name = f"site_{site_num}"
    
    # File paths
    train_file = os.path.join(data_dir, f"{site_name}_train_data.csv")
    
    if not os.path.exists(train_file):
        print(f"Warning: {train_file} not found!")
        return None
    
    # STEP 2: Load and explore
    df = load_and_explore_data(train_file, site_name)
    
    # STEP 3: Create datetime column
    df = create_datetime_column(df)
    
    # STEP 4: Handle satellite data
    df = handle_satellite_data(df)
    
    # STEP 5: Add lag features
    df = add_lag_features(df)
    
    # STEP 6: Split by date
    train_df, test_df = split_by_date(df, train_ratio=0.75)
    
    # Save processed data
    output_dir = os.path.join(data_dir, 'processed')
    os.makedirs(output_dir, exist_ok=True)
    
    train_output = os.path.join(output_dir, f"{site_name}_train_processed.csv")
    test_output = os.path.join(output_dir, f"{site_name}_test_processed.csv")
    
    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)
    
    print(f"\n[OK] Saved processed data:")
    print(f"  Train: {train_output}")
    print(f"  Test: {test_output}")
    
    return train_df, test_df

def main():
    """Process all sites"""
    print("="*60)
    print("SIH PS-10 Data Processing Pipeline")
    print("Processing all sites: Site 1 through Site 7")
    print("="*60)
    
    data_dir = 'Data_SIH_2025'
    
    # Process all sites
    for site_num in range(1, 8):
        try:
            train_df, test_df = process_site(site_num, data_dir)
            print(f"\n{'='*60}")
            print(f"[OK] Successfully processed Site {site_num}")
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"\n[ERROR] Error processing Site {site_num}: {str(e)}\n")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Data processing complete!")
    print("="*60)

if __name__ == "__main__":
    main()

