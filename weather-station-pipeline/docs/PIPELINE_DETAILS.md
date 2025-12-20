# Pipeline Details

This document provides detailed technical information about each step in the data processing pipeline.

## Table of Contents

1. [Data Flow](#data-flow)
2. [Step Details](#step-details)
3. [Data Transformations](#data-transformations)
4. [Quality Control](#quality-control)
5. [Performance Optimization](#performance-optimization)

## Data Flow

```
Raw NetCDF Files (station_*.nc)
    ↓
[Step 1: Download]
    ↓
Station Observations (hourly, scattered)
    ↓
[Step 2: Build Yearly]
    ↓
Yearly Grid Cubes (weather_data_YYYY.npz)
    ↓
[Step 3: Merge]
    ↓
Complete Dataset (weather_data_all.npz)
    ↓
[Step 4: Convert to Hybrid]
    ↓
Memory-Mappable Format (*.npy + *.npz)
    ↓
[Step 5-6: Create Holdout Split]
    ↓
Holdout Mask (holdout_mask.npz)
    ↓
[Step 7: Split Dataset]
    ↓
Training Dataset + Holdout Dataset
```

## Step Details

### Step 2: Build Yearly Data

**Key Operations:**

1. **Time Snapping**
   - Observations are snapped to nearest hour
   - Configurable threshold (default: ±15 minutes)
   - Records outside threshold are discarded

2. **Deduplication**
   - Multiple observations per (station, time, variable) are resolved
   - Keeps observation closest to target hour
   - Logs number of duplicates found

3. **Wind Component Calculation**
   - Derives u/v components from speed + direction
   - Uses meteorological convention (from-direction)
   - Formula:
     ```python
     u = -speed * sin(direction_rad)
     v = -speed * cos(direction_rad)
     ```

4. **Time Shift Tracking**
   - Stores actual observation time offset
   - Useful for quality assessment
   - Stored as separate variable 'ts'

5. **Grid Mapping**
   - Maps station IDs to grid positions
   - Uses precomputed mapping JSON
   - Handles missing stations gracefully

**Output Format:**
```python
{
    'data': [T_year, V, 90, 180],      # Main data cube
    'timestamps': [T_year],             # Hourly timestamps
    'variables': [V],                   # Variable names
    'mapping': station_info,            # Station metadata
    'lon_grid': [90, 180],             # Longitude grid
    'lat_grid': [90, 180],             # Latitude grid
    'locations': [V],                   # Normalization centers
    'scales': [V],                      # Normalization scales
}
```

### Step 3: Merge Yearly Data

**Validation Checks:**

1. **Shape Consistency**
   - All years must have same V, H, W
   - Temporal dimension can vary

2. **Variable Consistency**
   - Variable names must match exactly
   - Order must be identical

3. **Normalization Stats**
   - Verifies all years use same stats
   - Warns if discrepancies found

**Merge Strategy:**
- Concatenates along temporal (T) axis
- Sorts years chronologically
- Combines timestamps in order

### Step 5: Holdout Selection

**Uniform Geographic Sampling Algorithm:**

1. **K-means Clustering**
   - Uses 3D sphere coordinates (lat/lon → x,y,z)
   - Number of clusters ≈ n_holdout / 2
   - Ensures even spatial distribution

2. **Equal Sampling**
   - Samples same number from each cluster
   - Prioritizes high-coverage stations within cluster
   - Fills remainder from largest clusters

3. **Coverage Filtering**
   - Pre-filters by minimum coverage threshold
   - Auto-adjusts threshold if needed
   - Reports final coverage statistics

**Alternative Methods:**

- **Random**: Simple random sampling (may cluster geographically)
- **Stratified**: Proportional to station density (favors dense regions)

### Step 7: Dataset Splitting

**Masking Strategy:**

Instead of creating separate smaller files, both datasets keep full grid:

**Training Dataset:**
```python
training_data[t, v, row, col] = {
    original_value  if (row, col) is training station
    NaN             if (row, col) is holdout station
}
```

**Holdout Dataset:**
```python
holdout_data[t, v, row, col] = {
    original_value  if (row, col) is holdout station
    NaN             if (row, col) is training station
}
```

**Benefits:**
- Preserves spatial structure
- Easy to filter: `data[:, :, mask]`
- Compatible with convolutional models
- No index mapping required

## Data Transformations

### Normalization

Data is normalized using global statistics:

```python
normalized = (raw - location) / scale
```

Where:
- `location`: Per-variable mean or median (robust statistic)
- `scale`: Per-variable standard deviation or IQR

**To denormalize:**
```python
raw = normalized * scale + location
```

### Variable Transformations

Some variables undergo additional preprocessing:

1. **Temperature** (2t, dt)
   - Stored in Kelvin
   - Already normalized

2. **Pressure** (pa, msl)
   - Stored in Pascals
   - Already normalized

3. **Wind** (10u, 10v)
   - Derived from speed/direction
   - Stored in m/s
   - Already normalized

4. **Time Shift** (ts)
   - Stored in minutes
   - Range: [-threshold, +threshold]
   - Already normalized

## Quality Control

### Missing Data Handling

1. **Spatial Gaps**: Grid cells without stations contain NaN
2. **Temporal Gaps**: Missing observations set to NaN
3. **Invalid Values**: Out-of-range values set to NaN

### Coverage Metrics

Coverage is measured as:
```python
coverage = (valid_observations / total_timesteps) * 100
```

Where:
- `valid_observations`: Non-NaN values
- `total_timesteps`: Total hours in period

### Quality Checks

1. **Value Range Checks**
   - Temperature: [-100°C, +60°C]
   - Pressure: [50000 Pa, 110000 Pa]
   - Wind Speed: [0 m/s, 100 m/s]

2. **Temporal Consistency**
   - Checks for large jumps (>3 std devs)
   - Flags suspicious patterns

3. **Spatial Consistency**
   - Compares with neighbors
   - Flags outliers

## Performance Optimization

### Memory Management

1. **Memory Mapping**
   - Use `mmap_mode='r'` for large arrays
   - Avoids loading entire dataset into RAM
   - OS handles caching automatically

2. **Batch Processing**
   - Process files in batches (default: 200)
   - Prevents memory exhaustion
   - Configurable via `--batch_size`

3. **Garbage Collection**
   - Explicit `gc.collect()` after batches
   - Frees unused memory promptly

### Parallel Processing

1. **Multiprocessing**
   - Uses process pool for file reading
   - Default: 8 workers
   - Configurable via `--workers`
   - Optimal: number of CPU cores

2. **I/O vs CPU Balance**
   - More workers for I/O-bound tasks
   - Fewer workers for CPU-bound tasks
   - Monitor CPU/disk usage to tune

### Disk I/O

1. **Sequential Access**
   - Reads files in order
   - Better for HDDs
   - Less important for SSDs

2. **Compression**
   - NPZ files use zlib compression
   - NPY files are uncompressed (faster access)
   - Trade-off: space vs speed

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Solution: Use `--use_memmap` flag
   - Reduce `--batch_size`
   - Process fewer years at once

2. **Slow Processing**
   - Solution: Increase `--workers`
   - Use SSD instead of HDD
   - Check I/O bottlenecks

3. **Disk Space**
   - Solution: Remove intermediate files
   - Use compression
   - Process years incrementally

4. **Inconsistent Normalization**
   - Solution: Ensure same stats JSON used
   - Regenerate if needed
   - Verify variable order

## Best Practices

1. **Incremental Processing**
   - Process years one at a time if memory-limited
   - Use `--start_year` and `--end_year` flags
   - Merge afterwards

2. **Validation**
   - Always check coverage plots
   - Verify data ranges
   - Spot-check random samples

3. **Backup**
   - Keep raw data backed up
   - Save normalization stats
   - Version control config files

4. **Documentation**
   - Log processing parameters
   - Save random seeds
   - Document any modifications
