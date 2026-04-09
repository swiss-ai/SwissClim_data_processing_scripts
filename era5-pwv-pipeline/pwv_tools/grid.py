from __future__ import annotations

import math

import numpy as np

from .constants import D2R, ERA5_N360_LEVELS_HPA, ERA5_N360_NLAT, ERA5_N360_NLON


def build_era5_axes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    longitudes = np.arange(ERA5_N360_NLON, dtype=np.float64) * 360.0 / ERA5_N360_NLON
    latitudes = 90.0 - np.arange(ERA5_N360_NLAT, dtype=np.float64) * 0.25
    pressure_levels = np.asarray(ERA5_N360_LEVELS_HPA, dtype=np.float64)
    return latitudes, longitudes, pressure_levels


def _angular_separation(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    cosine_value = (
        math.sin(lat_a * D2R) * math.sin(lat_b * D2R)
        + math.cos(lat_a * D2R) * math.cos(lat_b * D2R) * math.cos((lon_a - lon_b) * D2R)
    )
    cosine_value = max(-1.0, min(1.0, float(cosine_value)))
    return max(math.acos(cosine_value), 1.0e-8)


def build_horizontal_lookup(
    station_latitudes: np.ndarray,
    station_longitudes: np.ndarray,
    grid_latitudes: np.ndarray,
    grid_longitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    station_count = len(station_latitudes)
    if np.any((station_latitudes > grid_latitudes[0]) | (station_latitudes < grid_latitudes[-1])):
        raise ValueError("The point is not in the range of the region.")

    wrapped_longitudes = np.where(station_longitudes < 0.0, station_longitudes + 360.0, station_longitudes)
    if np.any((wrapped_longitudes < grid_longitudes[0]) | (wrapped_longitudes > 360.0)):
        raise ValueError("The point is not in the range of the region.")

    latitude_index = np.searchsorted(-grid_latitudes, -station_latitudes, side="right") - 1
    latitude_index = np.clip(latitude_index, 0, len(grid_latitudes) - 2).astype(np.int64)

    longitude_index = np.searchsorted(grid_longitudes, wrapped_longitudes, side="right") - 1
    longitude_index = np.where(wrapped_longitudes > grid_longitudes[-1], len(grid_longitudes) - 1, longitude_index)
    longitude_index = np.clip(longitude_index, 0, len(grid_longitudes) - 2).astype(np.int64)

    corner_weights = np.zeros((4, station_count), dtype=np.float64)
    for idx, (site_latitude, site_longitude) in enumerate(zip(station_latitudes, wrapped_longitudes, strict=True)):
        row_index = int(latitude_index[idx])
        if site_longitude > grid_longitudes[-1] and site_longitude <= 360.0:
            column_index = len(grid_longitudes) - 1
            cell_corners = (
                (row_index, len(grid_longitudes) - 1),
                (row_index, 0),
                (row_index + 1, len(grid_longitudes) - 1),
                (row_index + 1, 0),
            )
        else:
            column_index = int(longitude_index[idx])
            cell_corners = (
                (row_index, column_index),
                (row_index, column_index + 1),
                (row_index + 1, column_index),
                (row_index + 1, column_index + 1),
            )

        corner_distances = np.array(
            [
                _angular_separation(grid_latitudes[i], grid_longitudes[j], site_latitude, site_longitude)
                for i, j in cell_corners
            ],
            dtype=np.float64,
        )
        inverse_distance = 1.0 / corner_distances
        corner_weights[:, idx] = inverse_distance / inverse_distance.sum()

    return latitude_index, longitude_index, corner_weights
