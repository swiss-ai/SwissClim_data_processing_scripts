# Demo Project

This folder shows the smallest complete workflow.

## Step 1

Download one ERA5 time step:

```bash
python ../download_era5_open.py --start 2024-10-02-00 --end 2024-10-02-00 --output-root data
```

## Step 2

Compute PWV from the downloaded GRIB file:

```bash
python ../gen_sit_pwv_from_grib.py data/2024/10/02/era5_global_37_2024100200_gst.grib 2024100200 sit.pos --output pwv_demo_output.txt
```

## Included file

- `sit.pos`: example station list
