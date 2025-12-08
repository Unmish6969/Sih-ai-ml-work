"""
Check for potential dataset issues that could affect CV splits.
"""
import pandas as pd
import numpy as np

def check_dataset(site_id=3, data_dir='F_cleaned'):
    """Check for dataset issues."""
    print(f"Checking dataset for site {site_id} in {data_dir}/")
    print("="*60)
    
    # Load data
    train_path = f"{data_dir}/site{site_id}_train.csv"
    df = pd.read_csv(train_path)
    
    print(f"\n1. Basic Info:")
    print(f"   Total rows: {len(df):,}")
    print(f"   Total columns: {len(df.columns)}")
    
    # Check datetime
    print(f"\n2. Datetime Column:")
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        print(f"   ✓ Datetime column exists")
        print(f"   NaN values: {df['datetime'].isna().sum()}")
        print(f"   Duplicated values: {df['datetime'].duplicated().sum()}")
        print(f"   Is sorted: {df['datetime'].is_monotonic_increasing}")
        print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        
        # Check for gaps
        df_sorted = df.sort_values('datetime').reset_index(drop=True)
        time_diffs = df_sorted['datetime'].diff()
        large_gaps = (time_diffs > pd.Timedelta(days=1)).sum()
        print(f"   Large gaps (>1 day): {large_gaps}")
    else:
        print(f"   ✗ Datetime column missing!")
    
    # Check target columns
    print(f"\n3. Target Columns:")
    for target in ['NO2_target', 'O3_target']:
        if target in df.columns:
            nan_count = df[target].isna().sum()
            nan_pct = (nan_count / len(df)) * 100
            print(f"   {target}:")
            print(f"     NaN values: {nan_count} ({nan_pct:.2f}%)")
            print(f"     Min: {df[target].min():.2f}, Max: {df[target].max():.2f}")
            print(f"     Mean: {df[target].mean():.2f}")
        else:
            print(f"   ✗ {target} missing!")
    
    # Check for rows that might be dropped during prepare_features
    print(f"\n4. Potential Issues in prepare_features():")
    target_col = 'NO2_target'
    if target_col in df.columns:
        nan_count = df[target_col].isna().sum()
        nan_pct = (nan_count / len(df)) * 100
        if nan_pct > 10:
            print(f"   ⚠ WARNING: {nan_pct:.1f}% NaN in {target_col} - rows will be DROPPED!")
            print(f"     This will reduce dataset size from {len(df):,} to ~{len(df) - nan_count:,}")
        else:
            print(f"   ✓ Only {nan_pct:.1f}% NaN - rows will be filled, not dropped")
    
    # Check index continuity
    print(f"\n5. Index Continuity:")
    print(f"   Index range: {df.index.min()} to {df.index.max()}")
    print(f"   Index is continuous: {df.index.is_monotonic_increasing}")
    
    # Check after prepare_features simulation
    print(f"\n6. After prepare_features() simulation:")
    from utils import prepare_features
    try:
        X, y, sw = prepare_features(df.copy(), target='NO2')
        print(f"   ✓ prepare_features() succeeded")
        print(f"   Final X shape: {X.shape}")
        print(f"   Final y length: {len(y)}")
        print(f"   Rows dropped: {len(df) - len(X)}")
        
        if len(df) != len(X):
            print(f"   ⚠ WARNING: Dataset size changed from {len(df):,} to {len(X):,}")
            print(f"     This could cause CV split issues if not accounted for!")
    except Exception as e:
        print(f"   ✗ Error in prepare_features(): {e}")
    
    # Check CV split coverage
    print(f"\n7. CV Split Coverage Test:")
    from utils import time_series_cv_splits
    try:
        # Test with actual X after prepare_features
        X, y, sw = prepare_features(df.copy(), target='NO2')
        splits = time_series_cv_splits(X, n_splits=3)
        
        print(f"   Number of splits generated: {len(splits)}")
        
        # Check coverage
        all_val_indices = set()
        for fold, (train_idx, val_idx) in enumerate(splits):
            print(f"   Fold {fold + 1}: train={len(train_idx):,}, val={len(val_idx):,}")
            all_val_indices.update(val_idx)
        
        total_covered = len(all_val_indices)
        total_samples = len(X)
        missing = total_samples - total_covered
        
        print(f"   Total samples: {total_samples:,}")
        print(f"   Samples covered by CV: {total_covered:,}")
        print(f"   Missing samples: {missing:,}")
        
        if missing > 0:
            print(f"   ⚠ WARNING: {missing:,} samples not covered by CV splits!")
            print(f"     This will cause NaN in OOF predictions!")
        else:
            print(f"   ✓ All samples covered by CV splits")
            
    except Exception as e:
        print(f"   ✗ Error in CV split test: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)

if __name__ == '__main__':
    check_dataset(site_id=3, data_dir='F_cleaned')
