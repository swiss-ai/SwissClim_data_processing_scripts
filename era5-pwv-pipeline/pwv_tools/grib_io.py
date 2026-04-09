from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .constants import ERA5_N360_LEVELS_HPA, ERA5_N360_NISO, ERA5_N360_NLAT, ERA5_N360_NLON

PARAM_TO_NAME = {129: "z", 130: "t", 133: "q"}


def _read_uint24(buf: bytes) -> int:
    return (buf[0] << 16) | (buf[1] << 8) | buf[2]


def _read_signed_grib_int16(buf: bytes) -> int:
    value = struct.unpack(">H", buf)[0]
    return -(value & 0x7FFF) if (value & 0x8000) else value


def _decode_ibm_float32(buf: bytes) -> float:
    value = int.from_bytes(buf, "big", signed=False)
    if value == 0:
        return 0.0
    sign = -1.0 if ((value >> 31) & 1) else 1.0
    exponent = (value >> 24) & 0x7F
    fraction = value & 0x00FFFFFF
    return sign * (fraction / float(1 << 24)) * (16.0 ** (exponent - 64))


def _unpack_simple_grid(section_bds: bytes, value_count: int, decimal_scale: int) -> np.ndarray:
    packing_flag = section_bds[3]
    if packing_flag & 0x60:
        raise ValueError("Unsupported GRIB packing mode.")
    binary_scale = _read_signed_grib_int16(section_bds[4:6])
    reference_value = _decode_ibm_float32(section_bds[6:10])
    bits_per_value = section_bds[10]
    ignored_tail_bits = packing_flag & 0x0F
    payload_bytes = memoryview(section_bds)[11:]

    if bits_per_value == 0:
        packed_values = np.zeros(value_count, dtype=np.float64)
    elif bits_per_value == 16 and ignored_tail_bits == 0:
        packed_values = np.frombuffer(payload_bytes, dtype=np.dtype(">u2"), count=value_count).astype(np.float64)
    else:
        bitstream = np.unpackbits(np.frombuffer(payload_bytes, dtype=np.uint8))
        if ignored_tail_bits:
            bitstream = bitstream[:-ignored_tail_bits]
        packed_values = bitstream[: value_count * bits_per_value].reshape(value_count, bits_per_value).dot(
            1 << np.arange(bits_per_value - 1, -1, -1, dtype=np.uint64)
        ).astype(np.float64)

    return (reference_value + packed_values * (2.0 ** binary_scale)) * (10.0 ** (-decimal_scale))


def load_grib_fields(file_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    file_path = Path(file_path)
    pressure_levels = np.asarray(ERA5_N360_LEVELS_HPA, dtype=np.int64)
    level_lookup = {int(level): i for i, level in enumerate(pressure_levels)}
    variable_cubes = {
        name: np.full((ERA5_N360_NLON, ERA5_N360_NLAT, ERA5_N360_NISO), np.nan, dtype=np.float64)
        for name in ("t", "q", "z")
    }

    with file_path.open("rb") as stream:
        while True:
            message_header = stream.read(8)
            if not message_header:
                break
            if len(message_header) < 8 or message_header[:4] != b"GRIB":
                raise ValueError(f"Invalid GRIB message header in {file_path}.")

            total_length = _read_uint24(message_header[4:7])
            if message_header[7] != 1:
                raise ValueError("Only GRIB1 is supported.")
            message = message_header + stream.read(total_length - 8)
            if message[-4:] != b"7777":
                raise ValueError("Invalid GRIB end marker.")

            cursor = 8
            section1_len = _read_uint24(message[cursor : cursor + 3])
            section1 = message[cursor : cursor + section1_len]
            cursor += section1_len
            if not (section1[7] & 0x80) or (section1[7] & 0x40):
                raise ValueError("Unsupported GDS/BMS configuration.")

            section2_len = _read_uint24(message[cursor : cursor + 3])
            section2 = message[cursor : cursor + section2_len]
            cursor += section2_len
            section4_len = _read_uint24(message[cursor : cursor + 3])
            section4 = message[cursor : cursor + section4_len]

            if section2[27] != 0:
                raise ValueError("Unsupported scanning mode.")
            grid_width = (section2[6] << 8) | section2[7]
            grid_height = (section2[8] << 8) | section2[9]
            if grid_width != ERA5_N360_NLON or grid_height != ERA5_N360_NLAT:
                raise ValueError("Unexpected grid size.")

            if section1[9] != 100:
                continue
            pressure_level = (section1[10] << 8) | section1[11]
            variable_name = PARAM_TO_NAME.get(section1[8])
            if variable_name is None or pressure_level not in level_lookup:
                continue

            decoded_values = _unpack_simple_grid(
                section4,
                grid_width * grid_height,
                _read_signed_grib_int16(section1[26:28]),
            )
            variable_cubes[variable_name][:, :, level_lookup[pressure_level]] = decoded_values.reshape(
                (grid_height, grid_width)
            ).T

    missing_fields = [name for name, cube in variable_cubes.items() if np.isnan(cube).any()]
    if missing_fields:
        raise ValueError(f"Missing GRIB records for variables: {missing_fields}")

    return variable_cubes["t"], variable_cubes["q"], variable_cubes["z"]
