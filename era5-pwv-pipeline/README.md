# ERA5 PWV Tools

Author: Zhenyi Zhang  
Contact: zhenyzhang@ethz.ch

Copyright (c) Zhenyi Zhang.

License: PolyForm Noncommercial 1.0.0. Noncommercial use only.

This project provides a small workflow to:

- download hourly ERA5 pressure-level GRIB files
- compute station PWV directly from GRIB
- run a minimal demo with a `sit.pos` file

The downloader only requests the fields used by this workflow:

- geopotential
- specific humidity
- temperature

## Layout

- `config/`: small static configuration files
- `docs/`: short documentation
- `examples/`: runnable example project
- `scripts/`: command-line entry points

Core implementation modules stay in the repository root to keep the project small.

## Quick Start

Download one ERA5 time step:

```bash
python scripts/download_era5.py --start 2024-10-02-00 --end 2024-10-02-00 --output-root examples/demo_project/data
```

Compute PWV from the downloaded GRIB file:

```bash
python scripts/compute_pwv.py examples/demo_project/data/2024/10/02/era5_global_37_2024100200_gst.grib 2024100200 examples/demo_project/sit.pos --output examples/demo_project/pwv_demo_output.txt
```

## Files

- `scripts/download_era5.py`: ERA5 downloader entry point
- `scripts/compute_pwv.py`: PWV computation entry point
- `examples/demo_project/`: small demo folder with `sit.pos`

## Optional Email Notification

Set these environment variables and add `--notify` to the downloader command:

- `ERA5_SMTP_USER`
- `ERA5_SMTP_PASS`
- `ERA5_NOTIFY_TO`
