# Quick Start Guide

Get started with the weather station data processing pipeline in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- 100GB+ free disk space
- 16GB+ RAM (32GB recommended)

## Installation

```bash
# Clone the repository
# Install dependencies
pip install -r requirements.txt
```

## Run the Pipeline

### Option 1: Automated Pipeline (Recommended)

Run the complete pipeline with a single command:

```bash
./run_pipeline.sh
```

This will execute all 7 steps automatically with default settings.

### Option 2: Step-by-Step Execution

Run each step individually for more control:

```bash
# Step 1: Download data (skip if you already have data)
python scripts/01_download_data.py

# Step 2: Build yearly data cubes
python scripts/02_build_station_data_yearly.py \
    --nc_folder station_data/raw_data \
    --mapping_json final_results_mapping_90x180.json \
    --stats_json weather_data_norm_stats.json \
    --grid_npz final_results_station_grid_90x180.npz \
    --out_prefix weather_data \
    --workers 48

# Step 3: Merge yearly data
python scripts/03_merge_yearly_npz.py \
    --input_pattern "weather_data_*.npz" \
    --output_file weather_data_all.npz

# Step 4: Convert to hybrid format
python scripts/04_convert_npz_to_hybrid.py

# Step 5: Select holdout stations
python scripts/05_select_holdout_stations.py \
    --coverage_csv station_coverage_2023.csv \
    --mapping_json final_results_mapping_90x180.json \
    --n_holdout 1000 \
    --min_coverage 40.0 \
    --out_json final_holdout_mapping_90x180.json \
    --sampling_method uniform

# Step 6: Create holdout mask
python scripts/06_create_holdout_mask.py \
    --holdout_json final_holdout_mapping_90x180.json \
    --out_npz holdout_mask.npz

# Step 7: Split dataset
python scripts/07_split_dataset_holdout.py \
    --data_npy stations_11k_var9_data.npy \
    --meta_npz stations_11k_var9_meta.npz \
    --holdout_mask holdout_mask.npz \
    --out_dir ESFM_Dataset
```

## Load and Use the Data

```python
import numpy as np

# Load training data
train_data = np.load('ESFM_Dataset/stations_11k_data_training.npy', mmap_mode='r')
train_meta = np.load('ESFM_Dataset/stations_11k_meta_training.npz', allow_pickle=True)

# Load holdout data
holdout_data = np.load('ESFM_Dataset/stations_11k_data_holdout.npy', mmap_mode='r')
holdout_meta = np.load('ESFM_Dataset/stations_11k_meta_holdout.npz', allow_pickle=True)

# Check shapes
print(f"Training data: {train_data.shape}")   # [219168, 9, 90, 180]
print(f"Holdout data: {holdout_data.shape}")  # [219168, 9, 90, 180]

# Get valid training stations
training_mask = train_meta['training_mask']
valid_train = train_data[:, :, training_mask]  # [219168, 9, 15200]

# Get valid holdout stations
holdout_mask = holdout_meta['holdout_mask']
valid_holdout = holdout_data[:, :, holdout_mask]  # [219168, 9, 1000]
```

## Next Steps

1. **Explore the data**: See `examples/usage_examples.py` for more examples
2. **Customize settings**: Edit `config/default_config.json`
3. **Visualize**: Check `coverage_plots/holdout_distribution.png`
4. **Read docs**: See `docs/PIPELINE_DETAILS.md` for technical details

## Common Customizations

### Change Number of Holdout Stations

```bash
python scripts/05_select_holdout_stations.py \
    --n_holdout 500  # Instead of default 1000
```

### Change Coverage Threshold

```bash
python scripts/05_select_holdout_stations.py \
    --min_coverage 50.0  # Instead of default 40.0
```

### Process Specific Years Only

```bash
python scripts/02_build_station_data_yearly.py \
    --start_year 2020 \
    --end_year 2023  # Process only 2020-2023
```

### Use More Workers for Faster Processing

```bash
python scripts/02_build_station_data_yearly.py \
    --workers 64  # Use 64 parallel workers
```

## Troubleshooting

### Out of Memory
```bash
# Use memory mapping
python scripts/02_build_station_data_yearly.py --use_memmap
```

### Disk Space Issues
```bash
# Remove intermediate files after each step
rm weather_data_20*.npz  # After step 3
rm weather_data_all.npz  # After step 4
```

### Slow Processing
```bash
# Increase workers
./run_pipeline.sh --workers 64
```

## Support

- **Documentation**: See `README.md` and `docs/`
- **Examples**: See `examples/usage_examples.py`
- **Issues**: Create an issue on GitHub
- **Questions**: Check the FAQ in README.md

## What's Next?

After running the pipeline, you'll have:

✅ Training dataset with 15,200 stations
✅ Holdout dataset with 1,000 stations
✅ 25 years of hourly weather data
✅ 9 weather variables
✅ Ready for machine learning!

Start building your weather forecasting model! 🚀
