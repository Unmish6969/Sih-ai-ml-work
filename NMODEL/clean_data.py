"""
Script to create cleaned CSV files with redundant satellite columns dropped.
"""
import pandas as pd
from pathlib import Path
import sys

def clean_csv_files(source_dir: str = 'F', output_dir: str = 'F_cleaned'):
    """
    Create cleaned CSV files by dropping redundant satellite columns.
    
    Args:
        source_dir: Source directory containing original CSV files
        output_dir: Output directory for cleaned CSV files
    """
    # Columns to drop (redundant satellite columns)
    redundant_satellite_cols = [
        'NO2_satellite_daily',      # Use NO2_satellite_daily_filled_broadcast instead
        'HCHO_satellite_daily',      # Use HCHO_satellite_daily_filled_broadcast instead
        'ratio_satellite_daily',     # Use ratio_satellite_daily_filled_broadcast instead
        'NO2_satellite_delta'        # Has NaN issues, no filled version available
    ]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"Created output directory: {output_dir}\n")
    
    # Get all CSV files in source directory
    source_path = Path(source_dir)
    csv_files = list(source_path.glob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {source_dir}/")
        return
    
    print(f"Found {len(csv_files)} CSV file(s) to process\n")
    
    total_dropped = 0
    total_rows = 0
    
    for csv_file in csv_files:
        print(f"Processing: {csv_file.name}")
        
        try:
            # Read CSV
            df = pd.read_csv(csv_file)
            original_cols = len(df.columns)
            original_rows = len(df)
            total_rows += original_rows
            
            # Check which redundant columns exist
            cols_to_drop = [col for col in redundant_satellite_cols if col in df.columns]
            
            if cols_to_drop:
                # Drop redundant columns
                df_cleaned = df.drop(columns=cols_to_drop)
                dropped_count = len(cols_to_drop)
                total_dropped += dropped_count
                
                print(f"  Original columns: {original_cols}")
                print(f"  Dropped {dropped_count} columns: {', '.join(cols_to_drop)}")
                print(f"  Remaining columns: {len(df_cleaned.columns)}")
            else:
                df_cleaned = df
                print(f"  No redundant columns found (already clean)")
            
            # Save cleaned CSV
            output_file = output_path / csv_file.name
            df_cleaned.to_csv(output_file, index=False)
            print(f"  ✓ Saved to: {output_file}")
            print()
            
        except Exception as e:
            print(f"  ✗ Error processing {csv_file.name}: {e}\n")
            continue
    
    print(f"{'='*60}")
    print(f"Summary:")
    print(f"  Total files processed: {len(csv_files)}")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Total columns dropped: {total_dropped}")
    print(f"  Output directory: {output_dir}/")
    print(f"{'='*60}\n")
    
    # Verify the cleaned files
    print("Verifying cleaned files...")
    cleaned_files = list(output_path.glob('*.csv'))
    for cleaned_file in cleaned_files:
        df_check = pd.read_csv(cleaned_file, nrows=1)
        print(f"  ✓ {cleaned_file.name}: {len(df_check.columns)} columns")
    
    print("\n✓ All files cleaned and saved successfully!")

if __name__ == '__main__':
    clean_csv_files()


