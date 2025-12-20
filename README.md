# SwissClim Data Processing Scripts

A collection of data processing pipelines for climate and weather datasets used in SwissClim projects.

## 📚 Overview

This repository contains multiple independent data processing pipelines, each designed to handle specific climate/weather datasets and prepare them for machine learning applications.

## 🗂️ Available Pipelines

### 1. Weather Station Pipeline

**Status**: ✅ Complete
**Directory**: [`weather-station-pipeline/`](./weather-station-pipeline/)
**Purpose**: Process global weather station observations into ML-ready datasets

**Key Features**:
- Processes 25+ years of hourly weather station data
- Handles ~11,800 stations globally
- Creates 90×180 spatial grid (2° resolution)
- Train/holdout split with geographic stratification
- 9 weather variables (temperature, pressure, wind, etc.)
- Memory-efficient hybrid format (120GB+ data)

**Output**:
- Training dataset: 15,200 stations
- Holdout dataset: 1,000 stations (uniform geographic distribution)
- Hourly temporal resolution
- Normalized and ML-ready

**Documentation**: See [weather-station-pipeline/README.md](./weather-station-pipeline/README.md)

**Quick Start**:
```bash
cd weather-station-pipeline
./run_pipeline.sh
```

---

### 2. [Additional Pipelines]

More dataset processing pipelines will be added here...

---

## 🎯 Pipeline Comparison

| Pipeline | Data Source | Spatial Coverage | Temporal Resolution | Grid Size | Output Size |
|----------|-------------|------------------|---------------------|-----------|-------------|
| **Weather Station** | Station observations | Global (11.8k stations) | Hourly | 90×180 | ~240 GB |
| _Future Pipeline_ | TBD | TBD | TBD | TBD | TBD |

## 📋 Repository Structure

```
SwissClim_data_processing_scripts/
├── README.md                          # This file
├── .gitignore                         # Global gitignore
│
├── weather-station-pipeline/          # Weather station processing
│   ├── README.md                      # Detailed pipeline documentation
│   ├── QUICKSTART.md                  # Quick start guide
│   ├── run_pipeline.sh                # Automated runner
│   ├── scripts/                       # Processing scripts (Steps 0-7)
│   ├── data/                          # Metadata files (~7 MB)
│   ├── config/                        # Configuration files
│   ├── docs/                          # Additional documentation
│   ├── examples/                      # Usage examples
│   └── utils/                         # Utility functions
│
└── [other-pipelines]/                 # Future additions
    └── ...
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- 100GB+ free disk space (varies by pipeline)
- 16GB+ RAM (32GB+ recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/SwissClim_data_processing_scripts.git
cd SwissClim_data_processing_scripts

# Navigate to specific pipeline
cd weather-station-pipeline

# Install dependencies
pip install -r requirements.txt

# Run pipeline
./run_pipeline.sh
```

## 📖 Documentation

Each pipeline has its own comprehensive documentation:

- **Weather Station Pipeline**: [weather-station-pipeline/README.md](./weather-station-pipeline/README.md)
  - Pipeline steps: [weather-station-pipeline/docs/PIPELINE_DETAILS.md](./weather-station-pipeline/docs/PIPELINE_DETAILS.md)
  - Quick start: [weather-station-pipeline/QUICKSTART.md](./weather-station-pipeline/QUICKSTART.md)
  - Usage examples: [weather-station-pipeline/examples/usage_examples.py](./weather-station-pipeline/examples/usage_examples.py)

## 🔧 Common Features

All pipelines in this repository share:

- ✅ **Reproducible**: Fixed random seeds, documented normalization
- ✅ **Scalable**: Handles large datasets with memory mapping
- ✅ **Well-documented**: Comprehensive README, usage examples
- ✅ **Automated**: One-command pipeline execution
- ✅ **ML-ready**: Normalized data with proper train/validation splits
- ✅ **Quality-controlled**: Data validation and coverage statistics

## 🤝 Contributing

When adding a new pipeline to this repository:

1. **Create a new directory** with a descriptive name
2. **Include comprehensive documentation**:
   - README.md with pipeline overview
   - QUICKSTART.md for quick start
   - Usage examples
3. **Follow the established structure**:
   - `scripts/` for processing scripts
   - `data/` for small metadata files
   - `config/` for configuration
   - `docs/` for additional documentation
   - `examples/` for usage examples
4. **Use relative paths** (no hardcoded absolute paths)
5. **Include requirements.txt** for dependencies
6. **Add automated runner script** when possible
7. **Update this README** with pipeline information

## 📊 Data Management

### What to Commit

✅ **Include in Git**:
- Scripts and code
- Documentation
- Configuration files
- Small metadata files (<10 MB)
- Examples and tutorials

❌ **Exclude from Git** (in .gitignore):
- Large data files (>10 MB)
- Generated datasets
- Temporary files
- Cache directories
- Model checkpoints

### Storage Guidelines

- **Small metadata** (<10 MB): Store in `data/` directory, commit to git
- **Medium files** (10 MB - 1 GB): Store in shared storage, document location
- **Large datasets** (>1 GB): Use dedicated storage, provide download instructions

## 🌐 Data Sources

- **Weather Station Data**: Global weather station observations (2000-2024)
  - Source: International station network
  - Format: NetCDF files with hourly observations
  - Variables: Temperature, pressure, wind, humidity
- _Additional sources will be documented as pipelines are added_

## 📝 Citation

If you use these pipelines in your research, please cite:

```bibtex
@software{swissclim_data_processing,
  title = {SwissClim Data Processing Scripts},
  author = {SwissClim Team},
  year = {2024},
  url = {https://github.com/YOUR_USERNAME/SwissClim_data_processing_scripts},
  note = {Data processing pipelines for climate and weather datasets}
}
```

## 📄 License

MIT License

See individual pipeline directories for specific licenses if different.

## 🙏 Acknowledgments

- Weather station data from international station networks
- SwissClim project team and contributors
- Open-source community for tools and libraries

## 📧 Contact

For questions or issues:
- Create an issue on GitHub
- Pull requests are welcome
- For collaboration inquiries, please open a discussion

## 🗺️ Roadmap

### Completed
- ✅ Weather Station Pipeline (v1.0.0)

### Planned
- ⏳ Additional climate dataset pipelines
- ⏳ Unified data loader utilities
- ⏳ Common preprocessing functions
- ⏳ Cross-dataset validation tools

---

**Last Updated**: December 2024
**Version**: 1.0.0
**Repository**: SwissClim Data Processing Scripts
