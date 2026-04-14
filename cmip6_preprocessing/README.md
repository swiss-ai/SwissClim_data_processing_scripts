# CMIP6 Preprocessing

Utilities for converting raw [CMIP6](https://pcmdi.llnl.gov/CMIP6/) NetCDF
files into sharded NumPy archives suitable for large-scale ML training.

---

## Overview

CMIP6 climate model output is distributed as NetCDF (`.nc`) files organised
by variable, frequency, and year.  Directly streaming these files during
training is slow.  This tooling converts them into compact `.npz` shards that
can be memory-mapped or loaded in parallel by a PyTorch `DataLoader`.

---

## Script

### `preprocess_cmip6.py`

Converts an MPI-ESM1-2-LR dataset tree into sharded `.npz` files.

**What it does**

| Step | Output |
|------|--------|
| Discovers all `.nc` files under `--path` grouped by year | — |
| Loads surface variables `(time, lat, lon)` | stored as `<var>` |
| Loads pressure-level variables `(time, plev, lat, lon)` | stored as `<var>_<hPa>` (e.g. `ta_500`) |
| Skips variables that are entirely NaN | warning logged |
| Splits each year into N equal shards | `<save_dir>/train/<t0>_<t1>.npz` |
| Pools per-year mean/std (law of total variance) | `normalize_mean.npz`, `normalize_std.npz` |
| Saves grid coordinates | `lat.npy`, `lon.npy` |
| Persists progress for crash recovery | `processing_state.json` |

**Usage**

```bash
python preprocess_cmip6.py --path /cluster/data/CMIP6/MPI-ESM1-2-LR

# Split each year into 4 shards, write to a custom output directory
python preprocess_cmip6.py \
    --path /cluster/data/CMIP6/MPI-ESM1-2-LR \
    --save_dir /output/cmip6_shards \
    --num_shards_per_year 4

# Enable verbose (DEBUG) logging
python preprocess_cmip6.py --path /cluster/data/CMIP6/MPI-ESM1-2-LR --verbose
```

**Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--path` | *(required)* | Root of the CMIP6 dataset tree |
| `--save_dir` | `sharded_npz_files` | Output directory |
| `--num_shards_per_year` | `1` | Shards per year |
| `--verbose` | off | Enable DEBUG logging |

**Output layout**

```
<save_dir>/
├── train/
│   ├── 2000010100_2000073112.npz   # shard 1
│   ├── 2000080100_2000123112.npz   # shard 2
│   └── ...
├── normalize_mean.npz              # global per-variable mean
├── normalize_std.npz               # global per-variable std
├── lat.npy                         # latitude grid
├── lon.npy                         # longitude grid
└── processing_state.json           # resume checkpoint
```

Each `.npz` shard contains:
- One array per variable (or per variable+level), shape `(T, ...)`.
- A `times` array of `datetime64` values, shape `(T,)`.

---

## Dependencies

```
numpy
xarray
tqdm
```

Install with:

```bash
pip install numpy xarray tqdm
# Optional: for cftime calendar support (NoLeap, 360-day, etc.)
pip install cftime
```

---

## License

Copyright (c) 2026 ETH Zurich.  
Authors: see [CONTRIBUTORS.md](../CONTRIBUTORS.md).  
Licensed under the MIT License — see [LICENSE](../LICENSE) for details.
