"""
Fix negative predictions by converting them to positive values.
Replaces negative O3_predicted and NO2_predicted values with their absolute values.
"""

import pandas as pd
import os
import glob

# Configuration
PREDICTIONS_DIR = "predictions/per_site"
BACKUP_DIR = "predictions/per_site_backup"

def fix_negative_predictions(predictions_dir, backup=True):
    """Fix negative predictions in all site prediction files."""
    
    # Create backup directory if needed
    if backup:
        os.makedirs(BACKUP_DIR, exist_ok=True)
        print(f"Backup directory: {BACKUP_DIR}")
    
    # Find all prediction files
    prediction_files = glob.glob(os.path.join(predictions_dir, "site_*_predictions.csv"))
    
    if not prediction_files:
        print(f"No prediction files found in {predictions_dir}")
        return
    
    print(f"Found {len(prediction_files)} prediction files")
    print(f"\nFixing negative predictions...")
    print(f"{'='*70}")
    
    total_fixed_o3 = 0
    total_fixed_no2 = 0
    
    for file_path in sorted(prediction_files):
        # Get site ID from filename
        filename = os.path.basename(file_path)
        site_id = filename.replace('site_', '').replace('_predictions.csv', '')
        
        # Read predictions
        df = pd.read_csv(file_path)
        
        # Count negative values before fixing
        negative_o3_before = (df['O3_predicted'] < 0).sum()
        negative_no2_before = (df['NO2_predicted'] < 0).sum()
        
        # Backup original file
        if backup:
            backup_path = os.path.join(BACKUP_DIR, filename)
            df.to_csv(backup_path, index=False)
        
        # Fix negative values by taking absolute value
        df['O3_predicted'] = df['O3_predicted'].abs()
        df['NO2_predicted'] = df['NO2_predicted'].abs()
        
        # Count negative values after fixing (should be 0)
        negative_o3_after = (df['O3_predicted'] < 0).sum()
        negative_no2_after = (df['NO2_predicted'] < 0).sum()
        
        # Save fixed file
        df.to_csv(file_path, index=False)
        
        total_fixed_o3 += negative_o3_before
        total_fixed_no2 += negative_no2_before
        
        print(f"Site {site_id}:")
        print(f"  O3:  {negative_o3_before:>5} negative values → fixed")
        print(f"  NO2: {negative_no2_before:>5} negative values → fixed")
        print(f"  File: {file_path}")
        if backup:
            print(f"  Backup: {backup_path}")
        print()
    
    print(f"{'='*70}")
    print(f"Summary:")
    print(f"  Total O3 negative values fixed:  {total_fixed_o3}")
    print(f"  Total NO2 negative values fixed: {total_fixed_no2}")
    print(f"  Files processed: {len(prediction_files)}")
    if backup:
        print(f"  Backups saved to: {BACKUP_DIR}")
    print(f"\n✅ All negative predictions fixed!")


def fix_combined_predictions():
    """Also fix the combined predictions file."""
    combined_file = "predictions/all_sites/all_sites_predictions.csv"
    
    if os.path.exists(combined_file):
        print(f"\nFixing combined predictions file...")
        print(f"{'='*70}")
        
        # Backup
        backup_file = combined_file.replace('.csv', '_backup.csv')
        df = pd.read_csv(combined_file)
        df.to_csv(backup_file, index=False)
        print(f"  Backup created: {backup_file}")
        
        # Count negative values
        negative_o3_before = (df['O3_predicted'] < 0).sum()
        negative_no2_before = (df['NO2_predicted'] < 0).sum()
        
        # Fix negative values
        df['O3_predicted'] = df['O3_predicted'].abs()
        df['NO2_predicted'] = df['NO2_predicted'].abs()
        
        # Save fixed file
        df.to_csv(combined_file, index=False)
        
        print(f"  O3:  {negative_o3_before:>5} negative values → fixed")
        print(f"  NO2: {negative_no2_before:>5} negative values → fixed")
        print(f"  File: {combined_file}")
        print(f"  ✅ Combined file fixed!")
    else:
        print(f"  Combined file not found: {combined_file}")


if __name__ == "__main__":
    print("="*70)
    print("Fixing Negative Predictions")
    print("="*70)
    
    # Fix per-site predictions
    fix_negative_predictions(PREDICTIONS_DIR, backup=True)
    
    # Fix combined predictions
    fix_combined_predictions()
    
    print(f"\n{'='*70}")
    print("✅ All done!")
    print(f"{'='*70}")

