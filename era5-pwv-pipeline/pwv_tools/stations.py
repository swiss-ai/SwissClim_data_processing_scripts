from __future__ import annotations

from pathlib import Path

import numpy as np


def load_station_table(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    station_ids: list[str] = []
    station_latitudes: list[float] = []
    station_longitudes: list[float] = []
    station_heights: list[float] = []

    with Path(path).open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue

            fields = line.split()
            if len(fields) < 4:
                continue

            try:
                latitude = float(fields[1])
                longitude = float(fields[2])
                height = float(fields[3])
            except ValueError:
                # Allow a simple header such as: NAME latitude longitude height(m)
                continue

            station_ids.append(fields[0])
            station_latitudes.append(latitude)
            station_longitudes.append(longitude + 360.0 if longitude < 0.0 else longitude)
            station_heights.append(height)

    return (
        np.asarray(station_ids, dtype=str),
        np.asarray(station_latitudes, dtype=np.float64),
        np.asarray(station_longitudes, dtype=np.float64),
        np.asarray(station_heights, dtype=np.float64),
    )
