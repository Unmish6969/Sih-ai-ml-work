"""
Predict for all unseen data and create scatter plots.
Processes all sites and generates visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import glob
import os
from pathlib import Path
from predict_with_model import (
    fill_satellite_data, add_derived_features, smooth_satellite_data,
    add_temporal_features, add_lag_features, add_rolling_features,
    prepare_features_for_prediction, find_latest_models
)

# Configuration
DATA_DIR = "Data_SIH_2025_with_blh/Data_SIH_2025_with_blh/with_satellite"
MODEL_DIR = "models/xgboost_per_site"
OUTPUT_DIR = "predictions/all_sites"
PLOT_DIR = "plots"
N_SITES = 7

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def predict_site_unseen(site_id, data_dir, model_dir):
    """Predict for a single site's unseen data."""
    unseen_file = os.path.join(data_dir, f"site_{site_id}_unseen_input_data_with_satellite.csv")
    
    if not os.path.exists(unseen_file):
        print(f"Warning: {unseen_file} not found. Skipping site {site_id}.")
        return None
    
    print(f"\n{'='*70}")
    print(f"Processing Site {site_id}")
    print(f"{'='*70}")
    
    # Load data
    df = pd.read_csv(unseen_file)
    df = df.sort_values(by=['year', 'month', 'day', 'hour']).reset_index(drop=True)
    print(f"  Loaded {len(df)} samples")
    
    # Check if satellite data needs filling (if raw columns exist, fill them)
    if 'NO2_satellite' in df.columns or 'HCHO_satellite' in df.columns:
        print(f"  Filling satellite data...")
        from predict_with_model import fill_satellite_data
        df = fill_satellite_data(df)
    
    # Feature engineering (same as predict_with_model.py)
    df = add_derived_features(df)
    df = smooth_satellite_data(df)
    df = add_temporal_features(df)
    df = add_lag_features(df, use_forecasts=True)
    df = add_rolling_features(df)
    
    # Load models
    try:
        o3_model_path, no2_model_path, metadata_path = find_latest_models(model_dir, site_id)
        o3_model = joblib.load(o3_model_path)
        no2_model = joblib.load(no2_model_path)
        
        if metadata_path:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            feature_cols = metadata['features']
        else:
            EXCLUDE_FEATURES = ['O3_target', 'NO2_target', 'site_id']
            feature_cols = [col for col in df.columns if col not in EXCLUDE_FEATURES]
    except Exception as e:
        print(f"  Error loading models: {e}")
        return None
    
    # Prepare features
    X = prepare_features_for_prediction(df, feature_cols)
    
    # Make predictions
    o3_predictions = o3_model.predict(X)
    no2_predictions = no2_model.predict(X)
    
    # Fix negative predictions (convert to positive using absolute value)
    negative_o3_count = (o3_predictions < 0).sum()
    negative_no2_count = (no2_predictions < 0).sum()
    
    if negative_o3_count > 0:
        print(f"  Warning: {negative_o3_count} negative O3 predictions found, converting to positive")
        o3_predictions = np.abs(o3_predictions)
    
    if negative_no2_count > 0:
        print(f"  Warning: {negative_no2_count} negative NO2 predictions found, converting to positive")
        no2_predictions = np.abs(no2_predictions)
    
    # Create output dataframe
    output_df = df[['year', 'month', 'day', 'hour']].copy()
    output_df['site_id'] = site_id
    output_df['O3_predicted'] = o3_predictions
    output_df['NO2_predicted'] = no2_predictions
    
    # Add datetime for plotting
    output_df['datetime'] = pd.to_datetime(
        output_df[['year', 'month', 'day']].astype(int).astype(str).agg('-'.join, axis=1) + 
        ' ' + output_df['hour'].astype(int).astype(str) + ':00:00',
        errors='coerce'
    )
    
    print(f"  Predictions made: O3 mean={o3_predictions.mean():.2f}, NO2 mean={no2_predictions.mean():.2f}")
    
    return output_df


def create_scatter_plots(all_predictions_df, plot_dir):
    """Create various scatter plots for predictions."""
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Creating Scatter Plots")
    print(f"{'='*70}")
    
    # 1. O3 vs NO2 scatter plot (all sites)
    print(f"\n1. Creating O3 vs NO2 scatter plot (all sites)")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by site
    sites = sorted(all_predictions_df['site_id'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(sites)))
    
    for site_id, color in zip(sites, colors):
        site_data = all_predictions_df[all_predictions_df['site_id'] == site_id]
        ax.scatter(site_data['NO2_predicted'], site_data['O3_predicted'], 
                  alpha=0.6, s=20, label=f'Site {site_id}', color=color)
    
    ax.set_xlabel('NO2 Predicted (ppb)', fontsize=12)
    ax.set_ylabel('O3 Predicted (ppb)', fontsize=12)
    ax.set_title('O3 vs NO2 Predictions (All Sites)', fontsize=14, fontweight='bold')
    ax.legend(title='Site ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'scatter_O3_vs_NO2_all_sites.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: scatter_O3_vs_NO2_all_sites.png")
    
    # 2. O3 vs NO2 scatter plot (per site)
    print(f"\n2. Creating O3 vs NO2 scatter plots (per site)")
    n_sites = len(sites)
    n_cols = 3
    n_rows = (n_sites + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_sites > 1 else [axes]
    
    for idx, site_id in enumerate(sites):
        site_data = all_predictions_df[all_predictions_df['site_id'] == site_id]
        
        axes[idx].scatter(site_data['NO2_predicted'], site_data['O3_predicted'], 
                         alpha=0.6, s=15, color=colors[idx])
        axes[idx].set_xlabel('NO2 Predicted (ppb)', fontsize=10)
        axes[idx].set_ylabel('O3 Predicted (ppb)', fontsize=10)
        axes[idx].set_title(f'Site {site_id} (n={len(site_data)})', fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_sites, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'scatter_O3_vs_NO2_per_site.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: scatter_O3_vs_NO2_per_site.png")
    
    # 3. Distribution scatter (O3 distribution)
    print(f"\n3. Creating O3 distribution scatter")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # O3 distribution by site
    for site_id, color in zip(sites, colors):
        site_data = all_predictions_df[all_predictions_df['site_id'] == site_id]
        axes[0].scatter(range(len(site_data)), site_data['O3_predicted'], 
                      alpha=0.5, s=10, label=f'Site {site_id}', color=color)
    
    axes[0].set_xlabel('Sample Index', fontsize=12)
    axes[0].set_ylabel('O3 Predicted (ppb)', fontsize=12)
    axes[0].set_title('O3 Predictions Distribution (All Sites)', fontsize=14, fontweight='bold')
    axes[0].legend(title='Site ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # NO2 distribution by site
    for site_id, color in zip(sites, colors):
        site_data = all_predictions_df[all_predictions_df['site_id'] == site_id]
        axes[1].scatter(range(len(site_data)), site_data['NO2_predicted'], 
                      alpha=0.5, s=10, label=f'Site {site_id}', color=color)
    
    axes[1].set_xlabel('Sample Index', fontsize=12)
    axes[1].set_ylabel('NO2 Predicted (ppb)', fontsize=12)
    axes[1].set_title('NO2 Predictions Distribution (All Sites)', fontsize=14, fontweight='bold')
    axes[1].legend(title='Site ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'scatter_distribution_O3_NO2.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: scatter_distribution_O3_NO2.png")
    
    # 4. Time series scatter (O3 and NO2 over time)
    print(f"\n4. Creating time series scatter plots")
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # O3 over time
    for site_id, color in zip(sites, colors):
        site_data = all_predictions_df[all_predictions_df['site_id'] == site_id].sort_values('datetime')
        axes[0].scatter(site_data['datetime'], site_data['O3_predicted'], 
                       alpha=0.6, s=15, label=f'Site {site_id}', color=color)
    
    axes[0].set_xlabel('Date', fontsize=12)
    axes[0].set_ylabel('O3 Predicted (ppb)', fontsize=12)
    axes[0].set_title('O3 Predictions Over Time (All Sites)', fontsize=14, fontweight='bold')
    axes[0].legend(title='Site ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    # NO2 over time
    for site_id, color in zip(sites, colors):
        site_data = all_predictions_df[all_predictions_df['site_id'] == site_id].sort_values('datetime')
        axes[1].scatter(site_data['datetime'], site_data['NO2_predicted'], 
                       alpha=0.6, s=15, label=f'Site {site_id}', color=color)
    
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_ylabel('NO2 Predicted (ppb)', fontsize=12)
    axes[1].set_title('NO2 Predictions Over Time (All Sites)', fontsize=14, fontweight='bold')
    axes[1].legend(title='Site ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'scatter_time_series_O3_NO2.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: scatter_time_series_O3_NO2.png")
    
    # 5. Correlation heatmap
    print(f"\n5. Creating correlation heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate correlation matrix
    corr_data = all_predictions_df[['O3_predicted', 'NO2_predicted', 'site_id']].copy()
    corr_matrix = corr_data.groupby('site_id')[['O3_predicted', 'NO2_predicted']].corr().unstack().iloc[:, 1]
    
    # Create correlation matrix for all sites
    site_corrs = []
    for site_id in sites:
        site_data = all_predictions_df[all_predictions_df['site_id'] == site_id]
        corr = site_data[['O3_predicted', 'NO2_predicted']].corr().iloc[0, 1]
        site_corrs.append(corr)
    
    # Create heatmap data
    heatmap_data = pd.DataFrame({
        'Site': [f'Site {s}' for s in sites],
        'O3-NO2 Correlation': site_corrs
    }).set_index('Site')
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('O3-NO2 Correlation by Site', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'heatmap_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: heatmap_correlation.png")
    
    # 6. Box plots for distribution comparison
    print(f"\n6. Creating box plots")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # O3 box plot
    o3_data = [all_predictions_df[all_predictions_df['site_id'] == s]['O3_predicted'].values 
               for s in sites]
    axes[0].boxplot(o3_data, labels=[f'Site {s}' for s in sites])
    axes[0].set_ylabel('O3 Predicted (ppb)', fontsize=12)
    axes[0].set_title('O3 Predictions Distribution by Site', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].tick_params(axis='x', rotation=45)
    
    # NO2 box plot
    no2_data = [all_predictions_df[all_predictions_df['site_id'] == s]['NO2_predicted'].values 
                for s in sites]
    axes[1].boxplot(no2_data, labels=[f'Site {s}' for s in sites])
    axes[1].set_ylabel('NO2 Predicted (ppb)', fontsize=12)
    axes[1].set_title('NO2 Predictions Distribution by Site', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'boxplot_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: boxplot_distribution.png")
    
    print(f"\n✅ All plots created successfully!")


def main():
    """Main function to predict all sites and create plots."""
    print("="*70)
    print("Predicting All Unseen Data and Creating Scatter Plots")
    print("="*70)
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # Predict for all sites
    all_predictions = []
    
    for site_id in range(1, N_SITES + 1):
        try:
            predictions = predict_site_unseen(site_id, DATA_DIR, MODEL_DIR)
            if predictions is not None:
                all_predictions.append(predictions)
        except Exception as e:
            print(f"Error processing site {site_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_predictions:
        print("\n❌ No predictions generated. Exiting.")
        return
    
    # Combine all predictions
    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    
    # Final check: Ensure no negative predictions remain (safety check)
    negative_o3_final = (all_predictions_df['O3_predicted'] < 0).sum()
    negative_no2_final = (all_predictions_df['NO2_predicted'] < 0).sum()
    
    if negative_o3_final > 0 or negative_no2_final > 0:
        print(f"\n⚠️  Final check: Found {negative_o3_final} negative O3 and {negative_no2_final} negative NO2 predictions")
        print(f"   Fixing remaining negative values...")
        all_predictions_df['O3_predicted'] = all_predictions_df['O3_predicted'].abs()
        all_predictions_df['NO2_predicted'] = all_predictions_df['NO2_predicted'].abs()
        print(f"   ✅ All negative values fixed!")
    
    # Save combined predictions
    output_file = os.path.join(OUTPUT_DIR, "all_sites_predictions.csv")
    all_predictions_df.to_csv(output_file, index=False)
    print(f"\n{'='*70}")
    print(f"Predictions Summary")
    print(f"{'='*70}")
    print(f"Total predictions: {len(all_predictions_df)}")
    print(f"Predictions saved to: {output_file}")
    
    # Summary statistics
    print(f"\nOverall Statistics:")
    print(f"  O3 - Mean: {all_predictions_df['O3_predicted'].mean():.4f}, "
          f"Std: {all_predictions_df['O3_predicted'].std():.4f}, "
          f"Min: {all_predictions_df['O3_predicted'].min():.4f}, "
          f"Max: {all_predictions_df['O3_predicted'].max():.4f}")
    print(f"  NO2 - Mean: {all_predictions_df['NO2_predicted'].mean():.4f}, "
          f"Std: {all_predictions_df['NO2_predicted'].std():.4f}, "
          f"Min: {all_predictions_df['NO2_predicted'].min():.4f}, "
          f"Max: {all_predictions_df['NO2_predicted'].max():.4f}")
    
    print(f"\nPer-Site Statistics:")
    print(f"{'Site':<6} {'Samples':<10} {'O3 Mean':<12} {'O3 Std':<12} {'NO2 Mean':<12} {'NO2 Std':<12}")
    print("-" * 70)
    for site_id in sorted(all_predictions_df['site_id'].unique()):
        site_data = all_predictions_df[all_predictions_df['site_id'] == site_id]
        print(f"Site {site_id:<3} {len(site_data):<10} "
              f"{site_data['O3_predicted'].mean():<12.4f} {site_data['O3_predicted'].std():<12.4f} "
              f"{site_data['NO2_predicted'].mean():<12.4f} {site_data['NO2_predicted'].std():<12.4f}")
    
    # Create scatter plots
    create_scatter_plots(all_predictions_df, PLOT_DIR)
    
    print(f"\n{'='*70}")
    print(f"✅ All done! Predictions and plots saved.")
    print(f"{'='*70}")
    print(f"Predictions: {output_file}")
    print(f"Plots: {PLOT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

