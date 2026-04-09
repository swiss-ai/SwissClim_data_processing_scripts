# Quickstart

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Download one ERA5 time step

```bash
python scripts/download_era5.py --start 2024-10-02-00 --end 2024-10-02-00 --output-root examples/demo_project/data
```

## 3. Compute PWV

```bash
python scripts/compute_pwv.py examples/demo_project/data/2024/10/02/era5_global_37_2024100200_gst.grib 2024100200 examples/demo_project/sit.pos --output examples/demo_project/pwv_demo_output.txt
```
