from __future__ import annotations

import math

import numpy as np

from .constants import C2D, C2K, E_RATIO, G0, RD, RE_EARTH, RHOW, RP_EARTH, RV


def translate_geopotential_profile(site_latitude: float, geopotential_profile: np.ndarray) -> np.ndarray:
    sin_latitude = math.sin(site_latitude * C2D)
    cos_latitude = math.cos(site_latitude * C2D)
    surface_gravity = 9.780327 * (
        1.0 + 5.3024e-3 * sin_latitude * sin_latitude - 5.8e-6 * math.sin(2.0 * site_latitude * C2D) ** 2
    )
    earth_radius = RE_EARTH / math.sqrt(
        RE_EARTH * RE_EARTH / RP_EARTH / RP_EARTH * sin_latitude * sin_latitude + cos_latitude * cos_latitude
    )
    orthometric_height = geopotential_profile * earth_radius / ((surface_gravity / G0) * earth_radius - geopotential_profile)
    gravity_profile = surface_gravity * earth_radius * earth_radius / (earth_radius + orthometric_height) / (
        earth_radius + orthometric_height
    )
    geopotential_profile[:] = orthometric_height
    return gravity_profile


def integrate_wet_delay(
    pressure_profile: np.ndarray,
    humidity_profile: np.ndarray,
    gravity_profile: np.ndarray,
    temperature_profile: np.ndarray,
) -> float:
    wet_coeff_2 = 0.712952
    wet_coeff_3 = 0.0375463e5
    wet_delay = 0.0
    for level_idx in range(1, len(pressure_profile)):
        wet_delay += (
            1.0e-6
            * RV
            * (humidity_profile[level_idx] + humidity_profile[level_idx - 1])
            / (gravity_profile[level_idx] + gravity_profile[level_idx - 1])
            * (
                -0.776890 * E_RATIO
                + wet_coeff_2
                + wet_coeff_3 * 2.0 / (temperature_profile[level_idx] + temperature_profile[level_idx - 1])
            )
            * (pressure_profile[level_idx - 1] - pressure_profile[level_idx])
        )
    return wet_delay


def compute_weighted_mean_temperature(
    temperature_profile: np.ndarray,
    dewpoint_profile: np.ndarray,
    height_profile: np.ndarray,
) -> float:
    numerator = 0.0
    denominator = 0.0
    for level_idx in range(len(temperature_profile) - 1):
        vapor_pressure_1 = 6.1121 * math.exp(
            17.67 * (dewpoint_profile[level_idx] - C2K) / (dewpoint_profile[level_idx] - C2K + 243.5)
        )
        vapor_pressure_2 = 6.1121 * math.exp(
            17.67 * (dewpoint_profile[level_idx + 1] - C2K) / (dewpoint_profile[level_idx + 1] - C2K + 243.5)
        )
        mean_vapor_pressure = 0.5 * (vapor_pressure_1 + vapor_pressure_2)
        mean_temperature = 0.5 * (temperature_profile[level_idx] + temperature_profile[level_idx + 1])
        layer_thickness = height_profile[level_idx + 1] - height_profile[level_idx]
        numerator += (mean_vapor_pressure / mean_temperature) * layer_thickness
        denominator += (mean_vapor_pressure / (mean_temperature * mean_temperature)) * layer_thickness
    return numerator / denominator


def zwd_to_pwv_factor(weighted_mean_temperature: float) -> float:
    return 1.0e6 / RHOW / RV / (3739.0 / weighted_mean_temperature + 0.221)
