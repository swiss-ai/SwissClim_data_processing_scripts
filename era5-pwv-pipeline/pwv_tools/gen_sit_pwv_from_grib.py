from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .constants import C2K, ERA5_N360_NISO, G0
from .grib_io import load_grib_fields
from .grid import build_era5_axes, build_horizontal_lookup
from .physics import (
    compute_weighted_mean_temperature,
    integrate_wet_delay,
    translate_geopotential_profile,
    zwd_to_pwv_factor,
)
from .stations import load_station_table
from .vertical import adjust_bottom_layer

OUTPUT_HEADER = (
    f"{'NAME':<12}"
    f"{'latitude':>20}"
    f"{'longitude':>20}"
    f"{'height(m)':>20}"
    f"{'PWV(m)':>20}\n"
)


def _blend_column_profiles(
    row_index: int,
    column_index: int,
    corner_weights: np.ndarray,
    temperature_cube: np.ndarray,
    humidity_cube: np.ndarray,
    geopotential_cube: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height_profile = (
        geopotential_cube[column_index, row_index, :] * corner_weights[0]
        + geopotential_cube[column_index + 1, row_index, :] * corner_weights[1]
        + geopotential_cube[column_index, row_index + 1, :] * corner_weights[2]
        + geopotential_cube[column_index + 1, row_index + 1, :] * corner_weights[3]
    ).astype(np.float64)
    temperature_profile = (
        temperature_cube[column_index, row_index, :] * corner_weights[0]
        + temperature_cube[column_index + 1, row_index, :] * corner_weights[1]
        + temperature_cube[column_index, row_index + 1, :] * corner_weights[2]
        + temperature_cube[column_index + 1, row_index + 1, :] * corner_weights[3]
    ).astype(np.float64)
    specific_humidity = (
        humidity_cube[column_index, row_index, :] * corner_weights[0]
        + humidity_cube[column_index + 1, row_index, :] * corner_weights[1]
        + humidity_cube[column_index, row_index + 1, :] * corner_weights[2]
        + humidity_cube[column_index + 1, row_index + 1, :] * corner_weights[3]
    ).astype(np.float64)
    specific_humidity[specific_humidity <= 0.0] = 1.0e-10
    return height_profile, temperature_profile, specific_humidity


def compute_site_pwv(
    site_latitude: float,
    site_height: float,
    row_index: int,
    column_index: int,
    corner_weights: np.ndarray,
    pressure_levels_hpa: np.ndarray,
    temperature_cube: np.ndarray,
    humidity_cube: np.ndarray,
    geopotential_cube: np.ndarray,
) -> float:
    height_profile, temperature_profile, specific_humidity = _blend_column_profiles(
        row_index,
        column_index,
        corner_weights,
        temperature_cube,
        humidity_cube,
        geopotential_cube,
    )
    vapor_pressure_profile = pressure_levels_hpa * specific_humidity / (0.622 + 0.378 * specific_humidity)
    log_vapor_pressure = np.log(vapor_pressure_profile / 6.1121)
    dewpoint_profile = log_vapor_pressure * 243.5 / (17.67 - log_vapor_pressure) + C2K
    height_profile /= G0

    gravity_profile = translate_geopotential_profile(site_latitude, height_profile)
    anchor_level = int(np.argmin(np.abs(site_height - height_profile)))
    if site_height < height_profile[anchor_level] and anchor_level < ERA5_N360_NISO - 1:
        anchor_level += 1

    usable_levels = np.arange(anchor_level, -1, -1)
    pressure_slice = pressure_levels_hpa[usable_levels] * 100.0
    height_slice = height_profile[usable_levels].copy()
    temperature_slice = temperature_profile[usable_levels].copy()
    dewpoint_slice = dewpoint_profile[usable_levels].copy()
    humidity_slice = specific_humidity[usable_levels].copy()
    gravity_slice = gravity_profile[usable_levels].copy()

    adjust_bottom_layer(
        site_latitude,
        site_height,
        height_slice,
        pressure_slice,
        temperature_slice,
        dewpoint_slice,
        humidity_slice,
        gravity_slice,
    )

    zwd = integrate_wet_delay(pressure_slice, humidity_slice, gravity_slice, temperature_slice)
    weighted_mean_temperature = compute_weighted_mean_temperature(temperature_slice, dewpoint_slice, height_slice)
    return zwd * zwd_to_pwv_factor(weighted_mean_temperature)


def main() -> None:
    parser = argparse.ArgumentParser(description="Directly read ERA5 GRIB and generate PWV results.")
    parser.add_argument("gribfile", help="ERA5 GRIB file")
    parser.add_argument("sdate", help="Date string, e.g. YYYYMMDDHH")
    parser.add_argument("sitfile", help="Station position file")
    parser.add_argument("--output", default=None, help="Optional output file path.")
    args = parser.parse_args()

    grid_latitudes, grid_longitudes, pressure_levels_hpa = build_era5_axes()
    station_ids, station_latitudes, station_longitudes, station_heights = load_station_table(args.sitfile)
    temperature_cube, humidity_cube, geopotential_cube = load_grib_fields(args.gribfile)
    row_index, column_index, corner_weights = build_horizontal_lookup(
        station_latitudes,
        station_longitudes,
        grid_latitudes,
        grid_longitudes,
    )

    output = Path(args.output or f"pwv_{args.sdate}_sit_era5")
    with output.open("w", encoding="utf-8") as stream:
        stream.write(OUTPUT_HEADER)
        for site_idx, station_id in enumerate(station_ids):
            pwv = compute_site_pwv(
                float(station_latitudes[site_idx]),
                float(station_heights[site_idx]),
                int(row_index[site_idx]),
                int(column_index[site_idx]),
                corner_weights[:, site_idx],
                pressure_levels_hpa,
                temperature_cube,
                humidity_cube,
                geopotential_cube,
            )
            stream.write(
                f"{station_id:<12}"
                f"{float(station_latitudes[site_idx]):20.8f}"
                f"{float(station_longitudes[site_idx]):20.8f}"
                f"{float(station_heights[site_idx]):20.8f}"
                f"{pwv:20.8f}\n"
            )


if __name__ == "__main__":
    main()
