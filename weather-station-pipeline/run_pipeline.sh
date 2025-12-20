#!/bin/bash
################################################################################
# Weather Station Data Processing Pipeline
#
# This script runs the complete data processing pipeline from raw data
# download to train/holdout split.
#
# Usage:
#   ./run_pipeline.sh [--skip-download] [--start-year YYYY] [--end-year YYYY]
#
# Options:
#   --skip-download    Skip data download step (use existing data)
#   --start-year YYYY  Process only years >= YYYY
#   --end-year YYYY    Process only years <= YYYY
#   --help             Show this help message
################################################################################

set -e  # Exit on error

# Default values
SKIP_DOWNLOAD=false
START_YEAR=""
END_YEAR=""
WORKERS=48
BATCH_SIZE=200
N_HOLDOUT=1000
MIN_COVERAGE=40.0
SAMPLING_METHOD="uniform"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-download)
      SKIP_DOWNLOAD=true
      shift
      ;;
    --start-year)
      START_YEAR="--start_year $2"
      shift 2
      ;;
    --end-year)
      END_YEAR="--end_year $2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --help)
      head -n 20 "$0" | tail -n +2 | sed 's/^# //'
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_step() {
  echo ""
  echo -e "${BLUE}========================================${NC}"
  echo -e "${BLUE}STEP $1: $2${NC}"
  echo -e "${BLUE}========================================${NC}"
}

print_success() {
  echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
  echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
  echo -e "${RED}✗ $1${NC}"
}

# Check Python version
python3 --version >/dev/null 2>&1 || {
  print_error "Python 3 is required but not found"
  exit 1
}

# Start pipeline
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Weather Station Data Processing Pipeline                 ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Download data
if [ "$SKIP_DOWNLOAD" = false ]; then
  print_step 1 "Downloading raw data"
  python3 scripts/01_download_data.py
  print_success "Data download complete"
else
  print_warning "Skipping data download (using existing data)"
fi

# Step 2: Build yearly data
print_step 2 "Building yearly data cubes"
python3 scripts/02_build_station_data_yearly.py \
  --nc_folder station_data/raw_data \
  --mapping_json final_results_mapping_90x180.json \
  --stats_json weather_data_norm_stats.json \
  --grid_npz final_results_station_grid_90x180.npz \
  --out_prefix weather_data \
  --workers $WORKERS \
  --batch_size $BATCH_SIZE \
  $START_YEAR $END_YEAR
print_success "Yearly data cubes built"

# Step 3: Merge yearly data
print_step 3 "Merging yearly data"
python3 scripts/03_merge_yearly_npz.py \
  --input_pattern "weather_data_*.npz" \
  --output_file weather_data_all.npz
print_success "Data merged into single file"

# Step 4: Convert to hybrid format
print_step 4 "Converting to hybrid format (NPY + NPZ)"
python3 scripts/04_convert_npz_to_hybrid.py
print_success "Converted to hybrid format"

# Step 5: Select holdout stations
print_step 5 "Selecting holdout stations"
python3 scripts/05_select_holdout_stations.py \
  --coverage_csv station_coverage_2023.csv \
  --mapping_json final_results_mapping_90x180.json \
  --n_holdout $N_HOLDOUT \
  --min_coverage $MIN_COVERAGE \
  --out_json final_holdout_mapping_90x180.json \
  --sampling_method $SAMPLING_METHOD
print_success "Holdout stations selected ($N_HOLDOUT stations)"

# Step 6: Create holdout mask
print_step 6 "Creating holdout mask"
python3 scripts/06_create_holdout_mask.py \
  --holdout_json final_holdout_mapping_90x180.json \
  --out_npz holdout_mask.npz
print_success "Holdout mask created"

# Step 7: Split dataset
print_step 7 "Splitting dataset into training and holdout"
python3 scripts/07_split_dataset_holdout.py \
  --data_npy stations_11k_var9_data.npy \
  --meta_npz stations_11k_var9_meta.npz \
  --holdout_mask holdout_mask.npz \
  --out_dir ESFM_Dataset
print_success "Dataset split complete"

# Optional: Plot holdout distribution
if [ -f "scripts/plot_holdout_distribution.py" ]; then
  print_step "BONUS" "Plotting holdout distribution"
  python3 scripts/plot_holdout_distribution.py
  print_success "Visualization saved to coverage_plots/holdout_distribution.png"
fi

# Summary
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Pipeline Complete!                                        ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Output files:"
echo "  - ESFM_Dataset/stations_11k_data_training.npy"
echo "  - ESFM_Dataset/stations_11k_meta_training.npz"
echo "  - ESFM_Dataset/stations_11k_data_holdout.npy"
echo "  - ESFM_Dataset/stations_11k_meta_holdout.npz"
echo ""
echo "Next steps:"
echo "  1. Review coverage_plots/holdout_distribution.png"
echo "  2. Load data using examples in docs/usage_examples.py"
echo "  3. Start training your model!"
echo ""
