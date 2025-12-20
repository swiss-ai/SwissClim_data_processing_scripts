#!/usr/bin/env python3
"""
Usage Examples for Weather Station Dataset

This file contains example code for loading and using the processed dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


# ============================================================================
# Example 1: Loading Training Data
# ============================================================================

def load_training_data():
    """Load and inspect training dataset."""
    print("Loading training data...")

    # Load data (memory-mapped for efficiency)
    train_data = np.load('ESFM_Dataset/stations_11k_data_training.npy', mmap_mode='r')
    train_meta = np.load('ESFM_Dataset/stations_11k_meta_training.npz', allow_pickle=True)

    # Print shape and metadata
    print(f"Training data shape: {train_data.shape}")  # [T, V, 90, 180]
    print(f"Timestamps: {len(train_meta['timestamps'])}")
    print(f"Variables: {train_meta['variables']}")

    # Get training mask
    training_mask = train_meta['training_mask']  # [90, 180]
    n_training = np.sum(training_mask)
    print(f"Number of training stations: {n_training}")

    return train_data, train_meta


# ============================================================================
# Example 2: Loading Holdout Data
# ============================================================================

def load_holdout_data():
    """Load and inspect holdout dataset."""
    print("Loading holdout data...")

    # Load data
    holdout_data = np.load('ESFM_Dataset/stations_11k_data_holdout.npy', mmap_mode='r')
    holdout_meta = np.load('ESFM_Dataset/stations_11k_meta_holdout.npz', allow_pickle=True)

    # Print shape and metadata
    print(f"Holdout data shape: {holdout_data.shape}")

    # Get holdout mask
    holdout_mask = holdout_meta['holdout_mask']  # [90, 180]
    n_holdout = np.sum(holdout_mask)
    print(f"Number of holdout stations: {n_holdout}")

    return holdout_data, holdout_meta


# ============================================================================
# Example 3: Extracting Valid Data (Filtering NaN)
# ============================================================================

def extract_valid_data(data, mask):
    """
    Extract only valid station data (remove NaN-masked stations).

    Args:
        data: Full data array [T, V, 90, 180]
        mask: Boolean mask [90, 180]

    Returns:
        Valid data [T, V, n_valid_stations]
    """
    # Extract data for valid stations only
    valid_data = data[:, :, mask]
    print(f"Original shape: {data.shape}")
    print(f"Filtered shape: {valid_data.shape}")

    return valid_data


# ============================================================================
# Example 4: Working with Specific Variables
# ============================================================================

def get_variable_data(data, meta, variable_name):
    """
    Extract data for a specific variable.

    Args:
        data: Full data array [T, V, 90, 180]
        meta: Metadata dictionary
        variable_name: Variable short name (e.g., '2t', '10u', 'msl')

    Returns:
        Variable data [T, 90, 180]
    """
    variables = meta['variables']
    var_idx = np.where(variables == variable_name)[0][0]

    var_data = data[:, var_idx, :, :]
    print(f"Extracted {variable_name}: shape = {var_data.shape}")

    return var_data


# ============================================================================
# Example 5: Denormalization
# ============================================================================

def denormalize_data(normalized_data, meta):
    """
    Denormalize data back to original scale.

    Args:
        normalized_data: Normalized data [T, V, ...]
        meta: Metadata with 'locations' and 'scales'

    Returns:
        Denormalized data
    """
    locations = meta['locations']  # [V]
    scales = meta['scales']        # [V]

    # Broadcast to match data shape
    # For data shape [T, V, H, W]:
    loc = locations[np.newaxis, :, np.newaxis, np.newaxis]
    scl = scales[np.newaxis, :, np.newaxis, np.newaxis]

    denormalized = normalized_data * scl + loc

    return denormalized


# ============================================================================
# Example 6: Time Series for a Specific Station
# ============================================================================

def get_station_timeseries(data, meta, row, col):
    """
    Extract time series for a specific grid cell.

    Args:
        data: Full data array [T, V, 90, 180]
        meta: Metadata dictionary
        row: Grid row index (0-89)
        col: Grid column index (0-179)

    Returns:
        Time series [T, V] and timestamps
    """
    timeseries = data[:, :, row, col]
    timestamps = meta['timestamps']

    print(f"Station at ({row}, {col}):")
    print(f"  Time series shape: {timeseries.shape}")
    print(f"  Date range: {timestamps[0]} to {timestamps[-1]}")

    return timeseries, timestamps


# ============================================================================
# Example 7: Calculating Statistics
# ============================================================================

def calculate_statistics(data, mask=None):
    """
    Calculate statistics over valid stations.

    Args:
        data: Data array [T, V, 90, 180]
        mask: Optional mask [90, 180] to filter stations

    Returns:
        Dictionary of statistics
    """
    if mask is not None:
        # Extract valid data
        valid_data = data[:, :, mask]
    else:
        # Flatten spatial dimensions
        valid_data = data.reshape(data.shape[0], data.shape[1], -1)

    # Calculate statistics (ignoring NaN)
    stats = {
        'mean': np.nanmean(valid_data, axis=(0, 2)),      # [V]
        'std': np.nanstd(valid_data, axis=(0, 2)),        # [V]
        'min': np.nanmin(valid_data, axis=(0, 2)),        # [V]
        'max': np.nanmax(valid_data, axis=(0, 2)),        # [V]
        'median': np.nanmedian(valid_data, axis=(0, 2)),  # [V]
    }

    return stats


# ============================================================================
# Example 8: Plotting Temperature Map
# ============================================================================

def plot_temperature_map(data, meta, timestep=0):
    """
    Plot a spatial map of temperature at a specific time.

    Args:
        data: Data array [T, V, 90, 180]
        meta: Metadata dictionary
        timestep: Time index to plot
    """
    # Get temperature variable index
    variables = meta['variables']
    temp_idx = np.where(variables == '2t')[0][0]

    # Extract temperature at timestep
    temp_map = data[timestep, temp_idx, :, :]

    # Denormalize if needed
    temp_map = temp_map * meta['scales'][temp_idx] + meta['locations'][temp_idx]

    # Convert from Kelvin to Celsius
    temp_map_celsius = temp_map - 273.15

    # Create plot
    plt.figure(figsize=(12, 6))
    plt.imshow(temp_map_celsius, cmap='RdYlBu_r', origin='upper', aspect='auto')
    plt.colorbar(label='Temperature (°C)')
    plt.title(f'Temperature Map - {meta["timestamps"][timestep]}')
    plt.xlabel('Longitude Index')
    plt.ylabel('Latitude Index')
    plt.tight_layout()
    plt.savefig('temperature_map.png', dpi=150, bbox_inches='tight')
    print("Saved temperature_map.png")


# ============================================================================
# Example 9: Data Generator for Training
# ============================================================================

class WeatherDataGenerator:
    """
    Simple data generator for batch training.
    """

    def __init__(self, data_path, meta_path, batch_size=32, shuffle=True):
        """
        Initialize generator.

        Args:
            data_path: Path to .npy data file
            meta_path: Path to .npz metadata file
            batch_size: Batch size
            shuffle: Whether to shuffle data
        """
        self.data = np.load(data_path, mmap_mode='r')
        self.meta = np.load(meta_path, allow_pickle=True)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.T = self.data.shape[0]
        self.indices = np.arange(self.T)

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        """Number of batches per epoch."""
        return int(np.ceil(self.T / self.batch_size))

    def __getitem__(self, idx):
        """Get a batch of data."""
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.T)

        batch_indices = self.indices[start_idx:end_idx]
        batch_data = self.data[batch_indices]

        return batch_data

    def on_epoch_end(self):
        """Shuffle indices at end of epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)


# ============================================================================
# Example 10: Computing Train/Holdout Metrics Separately
# ============================================================================

def evaluate_on_splits(predictions, targets, train_mask, holdout_mask):
    """
    Evaluate predictions separately on train and holdout sets.

    Args:
        predictions: Model predictions [T, V, 90, 180]
        targets: Ground truth [T, V, 90, 180]
        train_mask: Training station mask [90, 180]
        holdout_mask: Holdout station mask [90, 180]

    Returns:
        Dictionary with train and holdout metrics
    """
    # Extract train predictions and targets
    train_pred = predictions[:, :, train_mask]
    train_true = targets[:, :, train_mask]

    # Extract holdout predictions and targets
    holdout_pred = predictions[:, :, holdout_mask]
    holdout_true = targets[:, :, holdout_mask]

    # Calculate MSE (ignoring NaN)
    train_mse = np.nanmean((train_pred - train_true) ** 2)
    holdout_mse = np.nanmean((holdout_pred - holdout_true) ** 2)

    # Calculate MAE
    train_mae = np.nanmean(np.abs(train_pred - train_true))
    holdout_mae = np.nanmean(np.abs(holdout_pred - holdout_true))

    metrics = {
        'train': {
            'mse': train_mse,
            'mae': train_mae,
            'rmse': np.sqrt(train_mse),
        },
        'holdout': {
            'mse': holdout_mse,
            'mae': holdout_mae,
            'rmse': np.sqrt(holdout_mse),
        }
    }

    print(f"Training   - MSE: {train_mse:.4f}, MAE: {train_mae:.4f}")
    print(f"Holdout    - MSE: {holdout_mse:.4f}, MAE: {holdout_mae:.4f}")

    return metrics


# ============================================================================
# Main Demo
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Weather Station Dataset - Usage Examples")
    print("=" * 80)

    # Example 1: Load data
    print("\n1. Loading training data...")
    train_data, train_meta = load_training_data()

    print("\n2. Loading holdout data...")
    holdout_data, holdout_meta = load_holdout_data()

    # Example 3: Extract valid data
    print("\n3. Extracting valid training data...")
    valid_train = extract_valid_data(train_data, train_meta['training_mask'])

    # Example 4: Get specific variable
    print("\n4. Extracting temperature data...")
    temp_data = get_variable_data(train_data, train_meta, '2t')

    # Example 5: Denormalize
    print("\n5. Denormalizing data...")
    denorm_data = denormalize_data(train_data[:10], train_meta)  # First 10 timesteps

    # Example 6: Station time series
    print("\n6. Getting station time series...")
    ts, timestamps = get_station_timeseries(train_data, train_meta, row=45, col=90)

    # Example 7: Calculate statistics
    print("\n7. Calculating statistics...")
    stats = calculate_statistics(train_data, train_meta['training_mask'])
    print(f"Temperature stats (normalized): mean={stats['mean'][2]:.3f}, std={stats['std'][2]:.3f}")

    print("\n" + "=" * 80)
    print("Examples complete! Check the code for more details.")
    print("=" * 80)
