# Copyright (c) 2026 ETH Zurich
# Authors: see CONTRIBUTORS.md
# Licensed under the MIT License. See the LICENSE file in the repository root.

"""
Convert CMIP6 NetCDF files to sharded NumPy (.npz) files.

Pipeline
--------
For each year-group discovered under --path this script:

  1. Reads shared grid coordinates (lat, lon) and the time axis once.
  2. Loads each variable via xarray.
       - 3-D variables  (time, lat, lon)        → stored as-is.
       - 4-D variables  (time, plev, lat, lon)  → split into one array per
         pressure level, named  <var>_<level_hPa>  (e.g. "ta_500").
  3. Skips any variable/level whose data is entirely NaN.
  4. Splits each year into --num_shards_per_year shards and writes them as
     .npz files named  <start_time>_<end_time>.npz  under <save_dir>/train/.
     If a shard file already exists, new variables are merged into it
     (supports re-running after a crash).
  5. Accumulates per-year mean/std and pools them via the law of total
     variance into global normalize_mean.npz and normalize_std.npz.
  6. Saves lat.npy and lon.npy for the grid.
  7. Persists a processing_state.json so interrupted runs can resume.

Sample usage
------------
    python nc2np.py --path /cluster/data/CMIP6/MPI-ESM1-2-LR
    python nc2np.py --path /cluster/data/CMIP6/MPI-ESM1-2-LR \\
                    --save_dir /output/shards --num_shards_per_year 4
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All pressure levels (hPa) that will be kept when a variable has a plev dim.
DEFAULT_PRESSURE_LEVELS: List[int] = np.arange(1, 1001).tolist()

# Sub-directory tokens that identify relevant variable streams in MPI-ESM1-2-LR.
MPI_VAR_TYPES = {"3hr", "6hrPlevPt", "6hrPlev"}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Path discovery
# ---------------------------------------------------------------------------

def discover_nc_files(
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Walk *root* and collect all .nc files belonging to MPI variable streams.

    Returns
    -------
    year_to_paths : {year_string: [absolute_file_paths]}
    year_to_vars  : {year_string: [variable_names]}
        Variable name is the first underscore-separated token of the filename.
    """
    year_to_paths: Dict[str, List[str]] = defaultdict(list)
    year_to_vars:  Dict[str, List[str]] = defaultdict(list)

    for dirpath, _, files in os.walk(root):
        if not any(vt in dirpath for vt in MPI_VAR_TYPES):
            continue
        for fname in files:
            if not fname.endswith(".nc"):
                continue
            m = re.search(r"(\d+-\d+)\.nc$", fname)
            if not m:
                continue
            year_str = m.group(1)
            var_name = fname.split("_")[0]
            year_to_paths[year_str].append(os.path.join(dirpath, fname))
            year_to_vars[year_str].append(var_name)

    log.info("Discovered %d year-groups under %s", len(year_to_paths), root)
    return year_to_paths, year_to_vars


# ---------------------------------------------------------------------------
# Resume / progress state
# ---------------------------------------------------------------------------

def load_state(save_dir: str) -> Tuple[dict, str]:
    """Load processing state from disk, or return a fresh state dict."""
    state_file = os.path.join(save_dir, "processing_state.json")
    if os.path.exists(state_file):
        log.info("Resuming from existing state: %s", state_file)
        with open(state_file) as f:
            return json.load(f), state_file
    return {"completed_years": [], "errors": []}, state_file


def save_state(state: dict, state_file: str) -> None:
    """Persist state to disk with a timestamp."""
    state["last_update"] = datetime.now().isoformat()
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Time utilities
# ---------------------------------------------------------------------------

def to_datetime64(t) -> np.datetime64:
    """
    Convert any calendar-aware time object to numpy datetime64.

    Handles numpy datetime64, Python datetime, and cftime objects (all of
    which expose an .isoformat() method except native datetime64).
    """
    if isinstance(t, np.datetime64):
        return t
    try:
        return np.datetime64(t.isoformat())
    except AttributeError:
        return np.datetime64(str(t))


def fmt_time(t: np.datetime64) -> str:
    """Format a datetime64 as a compact string, e.g. '2000010112'."""
    return (
        np.datetime_as_string(t, unit="h")
        .replace("-", "")
        .replace(":", "")
        .replace("T", "")
    )


def year_boundaries(times: np.ndarray) -> List[Tuple[int, int]]:
    """
    Return (start_idx, end_idx) inclusive for each calendar year in *times*.

    The array-level cast to datetime64[Y] makes this fully vectorised.
    """
    years = times.astype("datetime64[Y]").astype(int) + 1970
    boundaries = []
    for yr in np.unique(years):
        idxs = np.where(years == yr)[0]
        boundaries.append((int(idxs[0]), int(idxs[-1])))
    return boundaries


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def aggregate_normalization(
    per_year_mean: Dict[str, List[float]],
    per_year_std:  Dict[str, List[float]],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Pool per-year scalar means and stds into global statistics.

    Uses the law of total variance (assuming equal group sizes):

        Var(X) = E[Var(X|year)] + Var(E[X|year])
               = mean(σ²)  +  mean(μ²)  −  mean(μ)²
    """
    global_mean: Dict[str, float] = {}
    global_std:  Dict[str, float] = {}

    for var in per_year_mean:
        means = np.array(per_year_mean[var])
        stds  = np.array(per_year_std[var])
        mu    = float(means.mean())
        var_  = float((stds ** 2).mean() + (means ** 2).mean() - mu ** 2)
        global_mean[var] = mu
        global_std[var]  = float(np.sqrt(max(var_, 0.0)))  # guard fp negatives

    return global_mean, global_std


# ---------------------------------------------------------------------------
# Core: load one year-group
# ---------------------------------------------------------------------------

def process_year(
    var_paths: List[str],
    var_names: List[str],
) -> Tuple[
    Dict[str, np.ndarray],   # data arrays
    np.ndarray,              # times  (datetime64)
    np.ndarray,              # lat
    np.ndarray,              # lon
    Dict[str, float],        # per-variable mean
    Dict[str, float],        # per-variable std
]:
    """
    Load all variables for one year-group of NC files.

    Grid coordinates and the time axis are read once from the first file —
    all files in a year-group share the same grid and temporal extent.

    Variables that are entirely NaN are silently skipped.
    Individual file errors are logged and skipped without aborting the year.
    """
    # --- shared coordinates & time axis (read once) -------------------------
    with xr.open_dataset(var_paths[0]) as ds0:
        lat   = ds0.lat.values
        lon   = ds0.lon.values
        times = np.array([to_datetime64(t) for t in ds0.time.to_numpy()])
    log.debug("Grid: %d lat × %d lon, %d timesteps", lat.size, lon.size, times.size)

    data:      Dict[str, np.ndarray] = {}
    norm_mean: Dict[str, float]      = {}
    norm_std:  Dict[str, float]      = {}

    for var_name, var_path in zip(var_names, var_paths):
        try:
            log.info("  Loading %-20s from %s", var_name, var_path)
            ds = xr.open_mfdataset(
                var_path, combine="by_coords", chunks={"time": 100}
            ).persist()

            if ds[var_name].ndim == 3:
                # ── Surface variable  (time, lat, lon) ──────────────────────
                arr = ds[var_name].to_numpy().astype(np.float32)
                if np.all(np.isnan(arr)):
                    log.warning("  Skipping %s: entirely NaN", var_name)
                    continue
                data[var_name]      = arr
                norm_mean[var_name] = float(np.nanmean(arr))
                norm_std[var_name]  = float(np.nanstd(arr))

            else:
                # ── Pressure-level variable  (time, plev, lat, lon) ─────────
                levels = np.intersect1d(
                    (ds["plev"][:].to_numpy() / 100).astype(int),
                    DEFAULT_PRESSURE_LEVELS,
                )
                for level in levels:
                    name = f"{var_name}_{level}"
                    arr  = (
                        ds.sel(plev=[level * 100.0])[var_name]
                        .to_numpy()
                        .astype(np.float32)
                    )
                    if np.all(np.isnan(arr)):
                        log.warning("  Skipping %s: entirely NaN", name)
                        continue
                    data[name]      = arr
                    norm_mean[name] = float(np.nanmean(arr))
                    norm_std[name]  = float(np.nanstd(arr))

        except Exception as exc:
            log.error("  Failed to load %s: %s", var_name, exc)
            continue

    return data, times, lat, lon, norm_mean, norm_std


# ---------------------------------------------------------------------------
# Core: shard and write
# ---------------------------------------------------------------------------

def shard_and_save(
    data:                Dict[str, np.ndarray],
    times:               np.ndarray,
    num_shards_per_year: int,
    save_dir:            str,
) -> None:
    """
    Slice *data* into *num_shards_per_year* equal windows per calendar year
    and write each window as a .npz file under <save_dir>/train/.

    File naming: <start_time>_<end_time>.npz  (hour resolution).

    If the target file already exists the new variables are merged in — this
    allows resuming a run that was interrupted mid-year.  The last shard of
    each year absorbs any remainder timesteps so nothing is dropped.
    """
    for yr_start, yr_end in year_boundaries(times):
        interval  = yr_end - yr_start + 1
        shard_len = interval // num_shards_per_year

        for i in range(num_shards_per_year):
            s = yr_start + i * shard_len
            e = yr_end if i == num_shards_per_year - 1 else s + shard_len - 1

            shard = {k: data[k][s: e + 1] for k in data}
            shard["times"] = times[s: e + 1]

            fname = f"{fmt_time(times[s])}_{fmt_time(times[e])}.npz"
            path  = os.path.join(save_dir, "train", fname)

            if os.path.exists(path):
                existing = dict(np.load(path))
                existing.update(shard)
                np.savez(path, **existing)
                log.info("  Updated  %s", path)
            else:
                np.savez(path, **shard)
                log.info("  Saved    %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CMIP6 MPI-ESM1-2-LR NC files to sharded .npz files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Root directory of the CMIP6 dataset tree.",
    )
    parser.add_argument(
        "--save_dir",
        default="sharded_npz_files",
        help="Output directory for shards and metadata.",
    )
    parser.add_argument(
        "--num_shards_per_year",
        type=int,
        default=1,
        help="How many .npz shards to produce for each year.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    _setup_logging(args.verbose)

    os.makedirs(os.path.join(args.save_dir, "train"), exist_ok=True)
    state, state_file = load_state(args.save_dir)

    year_to_paths, year_to_vars = discover_nc_files(args.path)
    if not year_to_paths:
        log.error("No .nc files found under %s — check --path.", args.path)
        sys.exit(1)

    log.info("State file: %s", state_file)
    log.info("Output dir: %s", args.save_dir)
    log.info("Shards/year: %d\n", args.num_shards_per_year)

    per_year_mean: Dict[str, List[float]] = defaultdict(list)
    per_year_std:  Dict[str, List[float]] = defaultdict(list)
    lat: Optional[np.ndarray] = None
    lon: Optional[np.ndarray] = None

    for year, var_paths in tqdm(sorted(year_to_paths.items()), desc="Years"):
        if year in state["completed_years"]:
            log.info("Skipping completed year: %s", year)
            continue

        log.info("Processing year group: %s", year)
        try:
            data, times, lat, lon, nm, ns = process_year(var_paths, year_to_vars[year])

            if not data:
                log.warning("No valid variables for %s — skipping.", year)
                continue

            shard_and_save(data, times, args.num_shards_per_year, args.save_dir)

            for var in nm:
                per_year_mean[var].append(nm[var])
                per_year_std[var].append(ns[var])

            state["completed_years"].append(year)
            save_state(state, state_file)

        except Exception as exc:
            state["errors"].append({
                "year":      year,
                "error":     str(exc),
                "timestamp": datetime.now().isoformat(),
            })
            save_state(state, state_file)
            log.error("Failed to process year %s: %s", year, exc)

    # ── Save global normalization stats and grid ────────────────────────────
    log.info("\nAggregating normalization statistics …")
    norm_mean, norm_std = aggregate_normalization(per_year_mean, per_year_std)
    np.savez(os.path.join(args.save_dir, "normalize_mean.npz"), **norm_mean)
    np.savez(os.path.join(args.save_dir, "normalize_std.npz"),  **norm_std)
    log.info("Saved normalize_mean.npz and normalize_std.npz")

    if lat is not None and lon is not None:
        np.save(os.path.join(args.save_dir, "lat.npy"), lat)
        np.save(os.path.join(args.save_dir, "lon.npy"), lon)
        log.info("Saved lat.npy and lon.npy")

    errors = state.get("errors", [])
    if errors:
        log.warning("%d year(s) finished with errors — see %s", len(errors), state_file)

    log.info("Done.")


if __name__ == "__main__":
    main()
