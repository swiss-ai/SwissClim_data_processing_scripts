#!/usr/bin/env python3
"""
build_station_yearly_npz.py
---------------------------
Build yearly 90x180 grid NPZ files for station data, using precomputed
global normalization stats (locations / scales) from a JSON file.

Key ideas:
    - Year is parsed from filename: station_YYYY_MM_DD.nc
    - For each year:
        * Build a full-year hourly timeline (e.g. 2001-01-01 00:00 to 2002-01-01 00:00)
        * Only process files belonging to that year
        * Use original logic (time_threshold, nearest_hour, u/v wind, mapping) to
          map station obs → (time_idx, var_idx, row, col) and fill cube
    - locations / scales are loaded from stats JSON (computed separately).

Output:
    For each year Y:
        {out_prefix}_{Y}.npz with:
            - data:       [T_year, V, 90, 180]
            - timestamps: [T_year] (hourly grid for the year)
            - variables:  [V]
            - mapping:    station mapping table
            - lon_grid, lat_grid
            - locations, scales (global stats)

Usage:
    python build_station_data_yearly.py \
      --nc_folder station_data/raw_data \
      --mapping_json final_results_mapping_90x180.json \
      --stats_json weather_data_norm_stats.json \
      --grid_npz final_results_station_grid_90x180.npz \
      --out_prefix weather_data \
      --pattern "*.nc" \
      --workers 48
"""

import argparse
import json
import glob
from pathlib import Path
from datetime import datetime, timedelta
from multiprocessing import Pool
import warnings
import tempfile
import os
import time
import gc

import netCDF4 as nc
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

GRID_H = 90
GRID_W = 180

# Variable name mapping: long name -> short name
VARIABLE_NAME_MAPPING = {
    "air_pressure": "pa",
    "air_pressure_at_sea_level": "msl",
    "air_temperature": "2t",
    "dew_point_temperature": "dt",
    "time_shift": "ts",
    "u_wind": "10u",
    "v_wind": "10v",
    "wind_from_direction": "wd",
    "wind_speed": "ws",
}


# ----------------------------------------------------------------------
# NetCDF processing
# ----------------------------------------------------------------------
def compute_wind_components_fast(df):
    """Fast wind component computation."""
    if "wind_speed" not in df["variable"].values or "wind_from_direction" not in df[
        "variable"
    ].values:
        return df

    df_pivot = df.pivot_table(
        index=["nearest_hour", "station_id"],
        columns="variable",
        values="value",
        aggfunc="first",
    )

    if "wind_speed" not in df_pivot.columns or "wind_from_direction" not in df_pivot.columns:
        return df

    direction_rad = np.pi / 180.0 * df_pivot["wind_from_direction"].values
    u_wind = -df_pivot["wind_speed"].values * np.sin(direction_rad)
    v_wind = -df_pivot["wind_speed"].values * np.cos(direction_rad)

    wind_records = []
    for i, (idx, _) in enumerate(df_pivot.iterrows()):
        if not np.isnan(u_wind[i]):
            wind_records.append(
                {
                    "nearest_hour": idx[0],
                    "station_id": idx[1],
                    "variable": "u_wind",
                    "value": u_wind[i],
                }
            )
        if not np.isnan(v_wind[i]):
            wind_records.append(
                {
                    "nearest_hour": idx[0],
                    "station_id": idx[1],
                    "variable": "v_wind",
                    "value": v_wind[i],
                }
            )

    if wind_records:
        df = pd.concat([df, pd.DataFrame(wind_records)], ignore_index=True)

    return df


def add_time_shift_variable(df):
    """Add time_diff_minutes as a separate variable 'time_shift'."""
    if "time_diff_minutes" not in df.columns:
        return df

    time_shift_records = []
    for _, row in df.iterrows():
        time_shift_records.append({
            "nearest_hour": row["nearest_hour"],
            "station_id": row["station_id"],
            "variable": "time_shift",
            "value": row["time_diff_minutes"],  # In minutes
        })

    if time_shift_records:
        df = pd.concat([df, pd.DataFrame(time_shift_records)], ignore_index=True)

    return df


def process_nc_file_direct(args):
    """
    Read one NC file and apply full filtering logic:

        - report_timestamp, primary_station_id, observed_variable, observation_value
        - Snap to nearest hour & filter by time_threshold_minutes
        - Deduplicate per (nearest_hour, station_id, variable)
        - Compute u/v wind
        - Map station_id → station index
        - Map variable   → var index

    Returns:
        hours:         [N] datetime64[ns]
        station_idx:   [N] int
        var_idx:       [N] int
        values:        [N] float
    """
    nc_file, station_to_idx, var_to_idx, time_threshold_minutes = args

    try:
        ds = nc.Dataset(nc_file, "r")

        timestamps = ds.variables["report_timestamp"][:]
        station_ids_raw = ds.variables["primary_station_id"][:]
        observed_vars_raw = ds.variables["observed_variable"][:]
        observation_values = ds.variables["observation_value"][:]

        ds.close()

        reference_date = datetime(1900, 1, 1)
        timestamps_dt = np.array(
            [reference_date + timedelta(seconds=int(ts)) for ts in timestamps]
        )

        n_obs = len(timestamps)
        station_ids = np.empty(n_obs, dtype="U50")
        observed_vars = np.empty(n_obs, dtype="U50")

        for i in range(n_obs):
            station_ids[i] = b"".join(station_ids_raw[i]).decode("utf-8").strip()
            observed_vars[i] = (
                b"".join(observed_vars_raw[i])
                .decode("utf-8")
                .replace("\x00", "")
                .strip()
            )

        df = pd.DataFrame(
            {
                "datetime": timestamps_dt,
                "station_id": station_ids,
                "variable": observed_vars,
                "value": observation_values,
            }
        )

        # Snap to nearest hour and filter by time threshold
        dt_series = pd.to_datetime(df["datetime"])
        df["nearest_hour"] = dt_series.dt.round("h")
        df["time_diff_minutes"] = (
            dt_series - df["nearest_hour"]
        ).dt.total_seconds() / 60.0

        df = df[df["time_diff_minutes"].abs() <= time_threshold_minutes]

        if len(df) == 0:
            return (np.array([]), np.array([]), np.array([]), np.array([]))

        # Keep only the closest observation per (time, station, variable)
        df["abs_time_diff"] = df["time_diff_minutes"].abs()
        df = (
            df.sort_values("abs_time_diff")
            .groupby(["nearest_hour", "station_id", "variable"], as_index=False)
            .first()
        )

        # Compute wind components (u_wind, v_wind) if possible
        df = compute_wind_components_fast(df)

        # Add time_shift as a separate variable
        df = add_time_shift_variable(df)

        # Convert variable names to short names
        df["variable"] = df["variable"].map(lambda x: VARIABLE_NAME_MAPPING.get(x, x))

        station_indices = df["station_id"].map(station_to_idx).values
        var_indices = df["variable"].map(var_to_idx).values

        valid_mask = (~pd.isna(station_indices)) & (~pd.isna(var_indices))

        if not np.any(valid_mask):
            return (np.array([]), np.array([]), np.array([]), np.array([]))

        hours = df["nearest_hour"].values[valid_mask]
        station_indices = station_indices[valid_mask].astype(int)
        var_indices = var_indices[valid_mask].astype(int)
        values = df["value"].values[valid_mask]

        return (hours, station_indices, var_indices, values)

    except Exception as e:
        print(f"[WARN] Failed to process {nc_file}: {e}")
        return (np.array([]), np.array([]), np.array([]), np.array([]))


# ----------------------------------------------------------------------
# Utility: parse year from filename station_YYYY_MM_DD.nc
# ----------------------------------------------------------------------
def get_year_from_filename(path):
    """
    Expect filename like: station_2001_03_01.nc
    """
    name = Path(path).stem  # station_2001_03_01
    parts = name.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected filename format: {path}")
    return int(parts[1])


# ----------------------------------------------------------------------
# Build cube for one year
# ----------------------------------------------------------------------
def build_cube_for_year(
    year,
    year_files,
    station_to_idx,
    station_rows,
    station_cols,
    var_to_idx,
    variables,
    time_threshold_minutes,
    batch_size,
    n_workers,
    use_memmap,
    dtype,
    lon_grid,
    lat_grid,
    mapping_df,
    locations,
    scales,
    out_prefix,
):
    print("\n" + "=" * 80)
    print(f"[Year {year}] Building cube")
    print("=" * 80)

    if len(year_files) == 0:
        print(f"  No files for year {year}, skip.")
        return

    # Build full-year hourly timeline: [year-01-01 00:00, (year+1)-01-01 00:00)
    start = np.datetime64(f"{year}-01-01T00:00")
    end = np.datetime64(f"{year+1}-01-01T00:00")
    step = np.timedelta64(1, "h")
    timestamps_year = np.arange(start, end, step, dtype="datetime64[h]").astype(
        "datetime64[ns]"
    )

    print(f"  Year {year}: {len(timestamps_year)} hours")
    print(f"  Time range: {timestamps_year[0]} → {timestamps_year[-1]}")

    # Create cube
    if use_memmap:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{year}.dat")
        tmp_file.close()
        print("  Creating memmap cube on disk...")
        cube = np.memmap(
            tmp_file.name,
            dtype=dtype,
            mode="w+",
            shape=(len(timestamps_year), len(variables), GRID_H, GRID_W),
        )
        cube[:] = np.nan
    else:
        print("  Creating in-memory cube...")
        cube = np.full(
            (len(timestamps_year), len(variables), GRID_H, GRID_W),
            np.nan,
            dtype=dtype,
        )

    print(f"  Cube shape: {cube.shape}")
    print(f"  Approx cube size: {cube.nbytes / 1e9:.2f} GB")

    num_batches = (len(year_files) + batch_size - 1) // batch_size
    print(f"\n  Processing {num_batches} batches for year {year}...")

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(year_files))
        batch_files = year_files[start_idx:end_idx]

        print(f"\n  Year {year} - Batch {batch_idx + 1}/{num_batches}: {len(batch_files)} files")

        process_args = [
            (f, station_to_idx, var_to_idx, time_threshold_minutes)
            for f in batch_files
        ]

        if n_workers > 1:
            with Pool(n_workers) as pool:
                batch_results = list(
                    tqdm(
                        pool.imap(process_nc_file_direct, process_args),
                        total=len(batch_files),
                        desc="    Files",
                        ncols=100,
                    )
                )
        else:
            batch_results = [
                process_nc_file_direct(args)
                for args in tqdm(process_args, desc="    Files", ncols=100)
            ]

        batch_hours_list = []
        batch_station_indices_list = []
        batch_var_indices_list = []
        batch_values_list = []

        for hours, station_idx, var_idx, values in batch_results:
            if len(hours) > 0:
                batch_hours_list.append(hours)
                batch_station_indices_list.append(station_idx)
                batch_var_indices_list.append(var_idx)
                batch_values_list.append(values)

        if not batch_hours_list:
            print("    → No valid records in this batch.")
            continue

        batch_hours = np.concatenate(batch_hours_list)
        batch_station_indices = np.concatenate(batch_station_indices_list)
        batch_var_indices = np.concatenate(batch_var_indices_list)
        batch_values = np.concatenate(batch_values_list)

        # Map times to indices in the full-year hourly grid
        hours_np = batch_hours.astype("datetime64[ns]")
        time_indices = np.searchsorted(timestamps_year, hours_np)

        # Guard against potential out-of-range (should be rare)
        valid_time = (time_indices >= 0) & (time_indices < len(timestamps_year))
        if not np.all(valid_time):
            # Drop any out-of-range observations
            time_indices = time_indices[valid_time]
            batch_station_indices = batch_station_indices[valid_time]
            batch_var_indices = batch_var_indices[valid_time]
            batch_values = batch_values[valid_time]

        print(f"    → Writing {len(batch_values):,} records into cube")

        row_indices = station_rows[batch_station_indices]
        col_indices = station_cols[batch_station_indices]

        t0 = time.time()
        cube[time_indices, batch_var_indices, row_indices, col_indices] = batch_values
        if use_memmap:
            cube.flush()
        elapsed = time.time() - t0
        print(
            f"    → Updated cube in {elapsed:.2f}s "
            f"({len(batch_values) / max(elapsed, 1e-6) / 1e6:.2f}M writes/s)"
        )

        # Free memory
        del (
            batch_results,
            batch_hours_list,
            batch_station_indices_list,
            batch_var_indices_list,
            batch_values_list,
            batch_hours,
            batch_station_indices,
            batch_var_indices,
            batch_values,
            hours_np,
            time_indices,
            row_indices,
            col_indices,
        )
        gc.collect()

    # ------------------------------------------------------------------
    # Save NPZ for this year
    # ------------------------------------------------------------------
    year_out_file = f"{out_prefix}_{year}.npz"
    print(f"\n  Saving year {year} cube to {year_out_file} ...")

    if isinstance(cube, np.memmap):
        print("    Converting memmap cube to ndarray for saving...")
        cube_to_save = np.array(cube)
        tmp_path = cube.filename
        del cube
        os.unlink(tmp_path)
    else:
        cube_to_save = cube

    non_nan = np.sum(~np.isnan(cube_to_save))
    print(f"    Non-NaN count: {non_nan:,}")

    np.savez_compressed(
        year_out_file,
        data=cube_to_save,
        timestamps=timestamps_year,
        variables=np.array(variables, dtype=object),
        mapping=mapping_df.to_records(index=False),
        lon_grid=lon_grid.astype(dtype),
        lat_grid=lat_grid.astype(dtype),
        locations=locations.astype(dtype),  # GLOBAL stats from JSON
        scales=scales.astype(dtype),        # GLOBAL stats from JSON
    )

    print(f"  ✅ Saved {year_out_file}")
    print(f"     Shape: {cube_to_save.shape}")
    print(f"     Size:  {cube_to_save.nbytes / 1e9:.2f} GB")

    del cube_to_save
    gc.collect()


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------
def build_yearly_npz(
    nc_folder,
    mapping_json,
    stats_json,
    grid_npz,
    out_prefix,
    pattern="*.nc",
    time_threshold_minutes=15,
    batch_size=200,
    n_workers=8,
    use_memmap=False,
    dtype=np.float32,
    start_year=None,
    end_year=None,
):
    print("=" * 80)
    print("BUILDING YEARLY NPZ WITH PRECOMPUTED GLOBAL STATS")
    print("=" * 80)
    print(f"\nNC folder:    {nc_folder}")
    print(f"Mapping JSON: {mapping_json}")
    print(f"Stats JSON:   {stats_json}")
    print(f"Grid NPZ:     {grid_npz}")
    print(f"Out prefix:   {out_prefix}")
    print(f"Pattern:     {pattern}")
    print(f"Batch size:  {batch_size}")
    print(f"Workers:     {n_workers}")
    print(f"Use memmap:  {use_memmap}")
    print(f"Time threshold (minutes): {time_threshold_minutes}")

    # Load mapping (JSON format)
    print("\nLoading station mapping...")
    with open(mapping_json, 'r') as f:
        mapping_data = json.load(f)

    # Convert JSON to DataFrame for compatibility with NPZ saving
    mapping_records = []
    for station_id, data in mapping_data.items():
        mapping_records.append({
            'station_id': station_id,
            'index': data['index'],
            'row': data['row'],
            'col': data['col'],
            'lat_orig': data.get('lat_orig', np.nan),
            'lon_orig': data.get('lon_orig', np.nan),
        })
    mapping_df = pd.DataFrame(mapping_records).sort_values('index').reset_index(drop=True)

    # Build station lookup dictionaries
    station_to_idx = {
        row["station_id"]: row["index"] for _, row in mapping_df.iterrows()
    }
    station_rows = mapping_df["row"].values.astype(int)
    station_cols = mapping_df["col"].values.astype(int)
    print(f"  Loaded {len(station_to_idx)} stations")

    # Discover files
    nc_files = sorted(glob.glob(str(Path(nc_folder) / pattern)))
    if not nc_files:
        raise FileNotFoundError(f"No files matching {pattern} in {nc_folder}")
    print(f"  Found {len(nc_files)} NetCDF files")

    # Group files by year (from filename)
    files_by_year = {}
    for f in nc_files:
        y = get_year_from_filename(f)
        if y not in files_by_year:
            files_by_year[y] = []
        files_by_year[y].append(f)

    all_years = sorted(files_by_year.keys())
    print("\nYears found from filenames:")
    for y in all_years:
        print(f"  {y}: {len(files_by_year[y])} files")

    # Load global stats JSON
    print("\nLoading global stats JSON...")
    with open(stats_json, "r") as f:
        stats = json.load(f)

    # Convert variable names from long to short
    variables_long = stats["variables"]
    variables = [VARIABLE_NAME_MAPPING.get(v, v) for v in variables_long]

    loc_dict = stats["locations"]
    scl_dict = stats["scales"]

    # Map locations and scales using long names from JSON, but store with short names
    locations = np.array([loc_dict[v_long] for v_long in variables_long], dtype=np.float32)
    scales = np.array([scl_dict[v_long] for v_long in variables_long], dtype=np.float32)

    stats_time_threshold = stats.get("time_threshold_minutes", None)
    if stats_time_threshold is not None and stats_time_threshold != time_threshold_minutes:
        print(
            f"\n⚠️ WARNING: stats were computed with time_threshold_minutes="
            f"{stats_time_threshold}, but current script uses {time_threshold_minutes}."
        )

    var_to_idx = {v: i for i, v in enumerate(variables)}

    print("\nVariables (from stats JSON):")
    for i, v in enumerate(variables):
        print(f"  {i:2d}: {v}")

    # Determine which years to build
    years_to_build = [
        y
        for y in all_years
        if (start_year is None or y >= start_year)
        and (end_year is None or y <= end_year)
    ]

    print("\nYears to build:")
    for y in years_to_build:
        print(f"  - {y}")

    # Load coordinate grids from NPZ
    print("\nLoading coordinate grids...")
    grid_data = np.load(grid_npz)
    lon_grid = grid_data['lon']
    lat_grid = grid_data['lat']
    print(f"  Loaded grids: lat {lat_grid.shape}, lon {lon_grid.shape}")

    # Build cube for each year
    for year in years_to_build:
        # Check if output file already exists
        year_out_file = f"{out_prefix}_{year}.npz"
        if os.path.exists(year_out_file):
            print(f"\nYear {year}: output file {year_out_file} already exists, skipping.")
            continue

        year_files = files_by_year.get(year, [])
        if not year_files:
            print(f"\nYear {year}: no files, skipping.")
            continue

        build_cube_for_year(
            year=year,
            year_files=year_files,
            station_to_idx=station_to_idx,
            station_rows=station_rows,
            station_cols=station_cols,
            var_to_idx=var_to_idx,
            variables=variables,
            time_threshold_minutes=time_threshold_minutes,
            batch_size=batch_size,
            n_workers=n_workers,
            use_memmap=use_memmap,
            dtype=dtype,
            lon_grid=lon_grid,
            lat_grid=lat_grid,
            mapping_df=mapping_df,
            locations=locations,
            scales=scales,
            out_prefix=out_prefix,
        )

    print("\n" + "=" * 80)
    print("✅ DONE! All requested yearly NPZs have been built.")
    print("=" * 80)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build yearly NPZ files using precomputed global stats."
    )
    parser.add_argument("--nc_folder", required=True)
    parser.add_argument("--mapping_json", required=True, help="Station grid mapping JSON file")
    parser.add_argument(
        "--stats_json",
        required=True,
        help="Global stats JSON from build_station_stats_global.py",
    )
    parser.add_argument(
        "--grid_npz",
        required=True,
        help="Grid coordinates NPZ file (contains 'lat' and 'lon' arrays)",
    )
    parser.add_argument(
        "--out_prefix",
        required=True,
        help="Prefix for yearly NPZ files, e.g. 'weather_data' → 'weather_data_2010.npz'",
    )
    parser.add_argument("--pattern", default="*.nc")
    parser.add_argument("--time_threshold", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--use_memmap",
        action="store_true",
        help="Use memory-mapped cube for yearly data.",
    )
    parser.add_argument("--start_year", type=int, default=None)
    parser.add_argument("--end_year", type=int, default=None)

    args = parser.parse_args()

    build_yearly_npz(
        nc_folder=args.nc_folder,
        mapping_json=args.mapping_json,
        stats_json=args.stats_json,
        grid_npz=args.grid_npz,
        out_prefix=args.out_prefix,
        pattern=args.pattern,
        time_threshold_minutes=args.time_threshold,
        batch_size=args.batch_size,
        n_workers=args.workers,
        use_memmap=args.use_memmap,
        dtype=np.float32,
        start_year=args.start_year,
        end_year=args.end_year,
    )
