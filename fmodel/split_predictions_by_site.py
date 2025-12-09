"""
Split combined predictions CSV into separate files for each site.
"""

import pandas as pd
import os

# Configuration
INPUT_FILE = "predictions/all_sites/all_sites_predictions.csv"
OUTPUT_DIR = "predictions/per_site"

def split_predictions_by_site(input_file, output_dir):
    """Split predictions CSV into separate files for each site."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read combined predictions
    print(f"Reading predictions from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"  Total predictions: {len(df)}")
    
    # Check if site_id column exists
    if 'site_id' not in df.columns:
        print("Error: 'site_id' column not found in predictions file!")
        return
    
    # Get unique site IDs
    site_ids = sorted(df['site_id'].unique())
    print(f"  Found {len(site_ids)} sites: {site_ids}")
    
    # Split by site and save
    print(f"\nSplitting predictions by site...")
    for site_id in site_ids:
        site_data = df[df['site_id'] == site_id].copy()
        
        # Remove site_id column from output (optional, or keep it)
        # site_data = site_data.drop('site_id', axis=1)  # Uncomment to remove site_id
        
        # Save to separate file
        output_file = os.path.join(output_dir, f"site_{int(site_id)}_predictions.csv")
        site_data.to_csv(output_file, index=False)
        
        print(f"  Site {int(site_id)}: {len(site_data)} predictions → {output_file}")
    
    print(f"\n✅ All predictions split successfully!")
    print(f"   Output directory: {output_dir}")
    print(f"   Files created: {len(site_ids)}")


if __name__ == "__main__":
    split_predictions_by_site(INPUT_FILE, OUTPUT_DIR)



