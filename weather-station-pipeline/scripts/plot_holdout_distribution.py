#!/usr/bin/env python3
"""
plot_holdout_distribution.py
----------------------------
Visualize the geographic distribution of hold-out vs training stations.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_holdout_vs_training(coverage_csv, holdout_json, out_file):
    # Load coverage data
    coverage_df = pd.read_csv(coverage_csv)

    # Load hold-out stations
    with open(holdout_json, 'r') as f:
        holdout_mapping = json.load(f)

    holdout_ids = set(holdout_mapping.keys())

    # Mark stations as holdout or training
    coverage_df['is_holdout'] = coverage_df['station_id'].isin(holdout_ids)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Geographic distribution (map view)
    ax = axes[0, 0]
    training = coverage_df[~coverage_df['is_holdout']]
    holdout = coverage_df[coverage_df['is_holdout']]

    ax.scatter(training['lon_orig'], training['lat_orig'],
              c='lightblue', s=5, alpha=0.3, label=f'Training ({len(training):,})')
    ax.scatter(holdout['lon_orig'], holdout['lat_orig'],
              c='red', s=20, alpha=0.7, marker='*', label=f'Hold-out ({len(holdout):,})')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Geographic Distribution: Hold-out vs Training Stations')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 2. Coverage comparison
    ax = axes[0, 1]
    bins = np.linspace(0, 100, 30)
    ax.hist(training['all_var_valid_mean_pct'], bins=bins, alpha=0.5, label='Training', color='blue', edgecolor='black')
    ax.hist(holdout['all_var_valid_mean_pct'], bins=bins, alpha=0.7, label='Hold-out', color='red', edgecolor='black')
    ax.axvline(training['all_var_valid_mean_pct'].mean(), color='blue', linestyle='--',
              label=f'Train mean: {training["all_var_valid_mean_pct"].mean():.1f}%')
    ax.axvline(holdout['all_var_valid_mean_pct'].mean(), color='red', linestyle='--',
              label=f'Holdout mean: {holdout["all_var_valid_mean_pct"].mean():.1f}%')
    ax.set_xlabel('Mean Coverage (%)')
    ax.set_ylabel('Number of Stations')
    ax.set_title('Coverage Distribution: Hold-out vs Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Grid heatmap showing holdout stations
    ax = axes[1, 0]
    grid_holdout = np.zeros((90, 180))
    for _, row in holdout.iterrows():
        r, c = int(row['row']), int(row['col'])
        grid_holdout[r, c] = 1

    im = ax.imshow(grid_holdout, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto', origin='upper')
    ax.set_xlabel('Grid Column')
    ax.set_ylabel('Grid Row')
    ax.set_title('Hold-out Stations in 90×180 Grid')
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.set_ticklabels(['Training', 'Hold-out'])

    # 4. Statistics comparison table
    ax = axes[1, 1]
    ax.axis('off')

    stats_text = f"""
    HOLD-OUT VS TRAINING COMPARISON
    {'='*50}

    TRAINING SET:
      Stations:       {len(training):,}
      Coverage mean:  {training['all_var_valid_mean_pct'].mean():.2f}%
      Coverage std:   {training['all_var_valid_mean_pct'].std():.2f}%
      Lat range:      {training['lat_orig'].min():.1f}° to {training['lat_orig'].max():.1f}°
      Lon range:      {training['lon_orig'].min():.1f}° to {training['lon_orig'].max():.1f}°

    HOLD-OUT SET:
      Stations:       {len(holdout):,}
      Coverage mean:  {holdout['all_var_valid_mean_pct'].mean():.2f}%
      Coverage std:   {holdout['all_var_valid_mean_pct'].std():.2f}%
      Lat range:      {holdout['lat_orig'].min():.1f}° to {holdout['lat_orig'].max():.1f}°
      Lon range:      {holdout['lon_orig'].min():.1f}° to {holdout['lon_orig'].max():.1f}°

    COVERAGE THRESHOLDS:
      Hold-out ≥90%:  {(holdout['all_var_valid_mean_pct'] >= 90).sum():,} ({(holdout['all_var_valid_mean_pct'] >= 90).sum()/len(holdout)*100:.1f}%)
      Hold-out ≥80%:  {(holdout['all_var_valid_mean_pct'] >= 80).sum():,} ({(holdout['all_var_valid_mean_pct'] >= 80).sum()/len(holdout)*100:.1f}%)
      Hold-out ≥70%:  {(holdout['all_var_valid_mean_pct'] >= 70).sum():,} ({(holdout['all_var_valid_mean_pct'] >= 70).sum()/len(holdout)*100:.1f}%)

    HEMISPHERE DISTRIBUTION:
      Northern (training): {(training['lat_orig'] > 0).sum():,} ({(training['lat_orig'] > 0).sum()/len(training)*100:.1f}%)
      Northern (holdout):  {(holdout['lat_orig'] > 0).sum():,} ({(holdout['lat_orig'] > 0).sum()/len(holdout)*100:.1f}%)
      Southern (training): {(training['lat_orig'] <= 0).sum():,} ({(training['lat_orig'] <= 0).sum()/len(training)*100:.1f}%)
      Southern (holdout):  {(holdout['lat_orig'] <= 0).sum():,} ({(holdout['lat_orig'] <= 0).sum()/len(holdout)*100:.1f}%)
    """

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved visualization to {out_file}")


if __name__ == "__main__":
    plot_holdout_vs_training(
        coverage_csv='station_coverage_2023.csv',
        holdout_json='final_holdout_mapping_90x180.json',
        out_file='coverage_plots/holdout_distribution.png'
    )
