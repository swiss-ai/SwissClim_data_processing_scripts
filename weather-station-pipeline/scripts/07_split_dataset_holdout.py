#!/usr/bin/env python3
"""
split_dataset_holdout.py
-------------------------
Split the full dataset into training and holdout sets using the holdout mask.

The script extracts only the relevant stations to save disk space:
- Training: [T, V, n_training_stations]
- Holdout: [T, V, n_holdout_stations]

Metadata is updated to reflect only the included stations.

Usage:
    python split_dataset_holdout.py \
        --data_npy stations_11k_var9_data.npy \
        --meta_npz stations_11k_var9_meta.npz \
        --holdout_mask holdout_mask.npz \
        --out_dir ESFM_Dataset
"""

import argparse
import numpy as np
from pathlib import Path


def split_dataset(data_npy, meta_npz, holdout_mask_npz, out_dir):
    print("=" * 80)
    print("SPLITTING DATASET INTO TRAINING AND HOLDOUT")
    print("=" * 80)

    out_dir = Path(out_dir)

    print(f"\nInput files:")
    print(f"  Data:         {data_npy}")
    print(f"  Metadata:     {meta_npz}")
    print(f"  Holdout mask: {holdout_mask_npz}")
    print(f"  Output dir:   {out_dir}")

    # Load mask
    print("\nLoading holdout mask...")
    mask_data = np.load(holdout_mask_npz)
    holdout_mask = mask_data['holdout_mask']  # [90, 180]
    training_mask = ~holdout_mask

    n_holdout = np.sum(holdout_mask)
    n_training = np.sum(training_mask)

    print(f"  Holdout stations:  {n_holdout}")
    print(f"  Training stations: {n_training}")

    # Load metadata
    print("\nLoading metadata...")
    meta = np.load(meta_npz, allow_pickle=True)
    timestamps = meta['timestamps']
    variables = meta['variables']
    mapping = meta['mapping']
    lon_grid = meta['lon_grid']
    lat_grid = meta['lat_grid']
    locations = meta['locations']
    scales = meta['scales']

    print(f"  Timestamps: {len(timestamps)}")
    print(f"  Variables:  {len(variables)}")
    print(f"  Mapping:    {len(mapping)} stations")

    # Load data (use memory mapping for efficiency)
    print("\nLoading data (memory-mapped)...")
    data = np.load(data_npy, mmap_mode='r')
    T, V, H, W = data.shape

    print(f"  Data shape: {data.shape}")
    print(f"  Data size:  {data.nbytes / 1e9:.2f} GB")

    # Create holdout data (keep same shape, mask training stations with NaN)
    print("\nCreating holdout data...")
    print("  Keeping spatial structure [T, V, 90, 180]")
    print("  Setting training stations to NaN")
    holdout_data = data.copy()  # Make a full copy
    holdout_data[:, :, training_mask] = np.nan  # Mask out training stations

    print(f"  Holdout data shape: {holdout_data.shape}")
    print(f"  Holdout data size:  {holdout_data.nbytes / 1e9:.2f} GB")
    print(f"  Valid stations:     {n_holdout} (rest are NaN)")

    # Create training data (keep same shape, mask holdout stations with NaN)
    print("\nCreating training data...")
    print("  Keeping spatial structure [T, V, 90, 180]")
    print("  Setting holdout stations to NaN")
    training_data = data.copy()  # Make a full copy
    training_data[:, :, holdout_mask] = np.nan  # Mask out holdout stations

    print(f"  Training data shape: {training_data.shape}")
    print(f"  Training data size:  {training_data.nbytes / 1e9:.2f} GB")
    print(f"  Valid stations:      {n_training} (rest are NaN)")

    # Save holdout dataset
    holdout_data_path = out_dir / 'stations_11k_data_holdout.npy'
    holdout_meta_path = out_dir / 'stations_11k_meta_holdout.npz'

    print(f"\nSaving holdout dataset...")
    print(f"  Saving data to {holdout_data_path}...")
    np.save(holdout_data_path, holdout_data)

    print(f"  Saving metadata to {holdout_meta_path}...")
    np.savez_compressed(
        holdout_meta_path,
        timestamps=timestamps,
        variables=variables,
        mapping=mapping,  # Keep all station mappings (same as original)
        lon_grid=lon_grid,
        lat_grid=lat_grid,
        locations=locations,
        scales=scales,
        holdout_mask=holdout_mask,  # Include mask to know which stations are valid
    )

    # Save training dataset
    training_data_path = out_dir / 'stations_11k_data_training.npy'
    training_meta_path = out_dir / 'stations_11k_meta_training.npz'

    print(f"\nSaving training dataset...")
    print(f"  Saving data to {training_data_path}...")
    np.save(training_data_path, training_data)

    print(f"  Saving metadata to {training_meta_path}...")
    np.savez_compressed(
        training_meta_path,
        timestamps=timestamps,
        variables=variables,
        mapping=mapping,  # Keep all station mappings (same as original)
        lon_grid=lon_grid,
        lat_grid=lat_grid,
        locations=locations,
        scales=scales,
        training_mask=training_mask,  # Include mask to know which stations are valid
    )

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nOriginal dataset:")
    print(f"  Data:        {data.shape} ({data.nbytes / 1e9:.2f} GB)")
    print(f"  Stations:    {len(mapping)}")

    print(f"\nHoldout dataset:")
    print(f"  Data:        {holdout_data.shape} ({holdout_data.nbytes / 1e9:.2f} GB)")
    print(f"  Valid stations: {n_holdout} (others are NaN)")
    print(f"  Files:")
    print(f"    - {holdout_data_path}")
    print(f"    - {holdout_meta_path}")

    print(f"\nTraining dataset:")
    print(f"  Data:        {training_data.shape} ({training_data.nbytes / 1e9:.2f} GB)")
    print(f"  Valid stations: {n_training} (others are NaN)")
    print(f"  Files:")
    print(f"    - {training_data_path}")
    print(f"    - {training_meta_path}")

    print(f"\nNote:")
    print(f"  Both datasets have the same shape and keep the spatial structure.")
    print(f"  Holdout data has training stations masked as NaN.")
    print(f"  Training data has holdout stations masked as NaN.")

    print("\n" + "=" * 80)
    print("USAGE")
    print("=" * 80)

    print("""
# Load holdout data
holdout_data = np.load('stations_11k_data_holdout.npy')  # Shape: [T, V, 90, 180]
holdout_meta = np.load('stations_11k_meta_holdout.npz', allow_pickle=True)

# Load training data
training_data = np.load('stations_11k_data_training.npy')  # Shape: [T, V, 90, 180]
training_meta = np.load('stations_11k_meta_training.npz', allow_pickle=True)

# Access metadata (same structure for both)
timestamps = training_meta['timestamps']
variables = training_meta['variables']
mapping = training_meta['mapping']  # All stations (same for both datasets)
locations = training_meta['locations']  # Normalization stats (same for both)
scales = training_meta['scales']

# Get masks to know which stations are valid
training_mask = training_meta['training_mask']  # Shape: [90, 180], True for training
holdout_mask = holdout_meta['holdout_mask']    # Shape: [90, 180], True for holdout

# Example: Extract only valid data (flattening spatial dimension)
valid_training = training_data[:, :, training_mask]  # Shape: [T, V, 15614]
valid_holdout = holdout_data[:, :, holdout_mask]     # Shape: [T, V, 586]

# Or work directly with the full grid (NaN values indicate masked stations)
# This preserves spatial structure for operations that need it
""")

    print("=" * 80)
    print("✅ DONE!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split dataset into training and holdout using mask."
    )
    parser.add_argument("--data_npy", required=True,
                        help="Input data NPY file")
    parser.add_argument("--meta_npz", required=True,
                        help="Input metadata NPZ file")
    parser.add_argument("--holdout_mask", required=True,
                        help="Holdout mask NPZ file")
    parser.add_argument("--out_dir", required=True,
                        help="Output directory")

    args = parser.parse_args()

    split_dataset(
        data_npy=args.data_npy,
        meta_npz=args.meta_npz,
        holdout_mask_npz=args.holdout_mask,
        out_dir=args.out_dir,
    )
