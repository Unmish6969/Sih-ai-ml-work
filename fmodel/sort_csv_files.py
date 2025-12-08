"""
Script to sort all CSV files in the specified directory according to year, month, day, and hour.
"""

import os
import pandas as pd
from pathlib import Path
import glob

def sort_csv_file(file_path):
    """
    Sort a CSV file by year, month, day, and hour columns.
    
    Args:
        file_path: Path to the CSV file to sort
    """
    try:
        print(f"Processing: {file_path}")
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if required columns exist
        required_columns = ['year', 'month', 'day', 'hour']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"  Warning: Missing columns {missing_columns} in {file_path}. Skipping...")
            return False
        
        # Sort by year, month, day, and hour
        df_sorted = df.sort_values(by=['year', 'month', 'day', 'hour'], 
                                   ascending=[True, True, True, True])
        
        # Save the sorted dataframe back to the same file
        df_sorted.to_csv(file_path, index=False)
        
        print(f"  ✓ Successfully sorted {len(df_sorted)} rows")
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {file_path}: {str(e)}")
        return False

def main():
    """
    Main function to sort all CSV files in the target directory.
    """
    # Target directory
    target_dir = Path("Data_SIH_2025_with_blh/Data_SIH_2025_with_blh/with_satellite")
    
    # Check if directory exists
    if not target_dir.exists():
        print(f"Error: Directory '{target_dir}' does not exist!")
        return
    
    # Find all CSV files in the directory
    csv_files = list(target_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in '{target_dir}'")
        return
    
    print(f"Found {len(csv_files)} CSV file(s) to process\n")
    
    # Process each CSV file
    success_count = 0
    failed_count = 0
    
    for csv_file in csv_files:
        if sort_csv_file(csv_file):
            success_count += 1
        else:
            failed_count += 1
        print()  # Empty line for readability
    
    # Summary
    print("=" * 60)
    print(f"Summary:")
    print(f"  Successfully sorted: {success_count} file(s)")
    print(f"  Failed: {failed_count} file(s)")
    print(f"  Total processed: {len(csv_files)} file(s)")
    print("=" * 60)

if __name__ == "__main__":
    main()

