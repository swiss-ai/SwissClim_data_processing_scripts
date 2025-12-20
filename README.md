# SwissClim Data Processing Scripts

Data processing pipelines for climate and weather datasets used in SwissClim projects.

## Available Pipelines

### Weather Station Pipeline

**Directory**: [`weather-station-pipeline/`](./weather-station-pipeline/)

Processes global weather station observations into ML-ready datasets:
- 25+ years of hourly data (~11,800 stations)
- 90×180 spatial grid with train/holdout split
- 9 weather variables (temperature, pressure, wind, etc.)

See [weather-station-pipeline/README.md](./weather-station-pipeline/README.md) for details.

### Future Pipelines

Additional climate dataset processing pipelines will be added here.

## License

MIT License
