from __future__ import annotations

import math

import numpy as np

from .constants import C2D, C2K, ETA, G0, LAPSE, RD, RE_EARTH, RP_EARTH


def extrapolate_pressure(
    target_height: float,
    anchor_height: float,
    anchor_temp_c: float,
    anchor_pressure: float,
    mean_gravity: float,
) -> float:
    height_gap = target_height - anchor_height
    step_count = max(1, math.ceil(abs(height_gap / 20.0)))
    running_temp_c = anchor_temp_c
    hydrostatic_sum = 0.0
    current_height = anchor_height
    for step_index in range(step_count):
        height_step = target_height - current_height if step_index == step_count - 1 else math.copysign(20.0, height_gap)
        current_height += height_step
        next_temp_c = running_temp_c + LAPSE * height_step
        hydrostatic_sum += height_step / (((running_temp_c + next_temp_c) / 2.0) + C2K)
        running_temp_c = next_temp_c
    return anchor_pressure * math.exp(-mean_gravity * hydrostatic_sum / RD)


def adjust_bottom_layer(
    site_latitude: float,
    site_height: float,
    height_profile: np.ndarray,
    pressure_profile: np.ndarray,
    temperature_profile: np.ndarray,
    dewpoint_profile: np.ndarray,
    humidity_profile: np.ndarray,
    gravity_profile: np.ndarray,
) -> None:
    lower_height = float(height_profile[0])
    next_height = float(height_profile[1])

    if site_height >= lower_height:
        lapse_a = (temperature_profile[0] - temperature_profile[1]) / (lower_height - next_height)
        lapse_b = (temperature_profile[1] * lower_height - temperature_profile[0] * next_height) / (lower_height - next_height)
        surface_temp = lapse_a * site_height + lapse_b
        dew_a = (dewpoint_profile[0] - dewpoint_profile[1]) / (lower_height - next_height)
        dew_b = (dewpoint_profile[1] * lower_height - dewpoint_profile[0] * next_height) / (lower_height - next_height)
        surface_dewpoint = dew_a * site_height + dew_b
    else:
        surface_temp = temperature_profile[0] - LAPSE * (lower_height - site_height)
        surface_dewpoint = dewpoint_profile[0] - LAPSE * (lower_height - site_height)

    surface_pressure = extrapolate_pressure(
        site_height,
        lower_height,
        temperature_profile[0] - C2K,
        pressure_profile[0],
        0.5 * (gravity_profile[0] + gravity_profile[1]),
    )
    vapor_pressure = 6.1121 * math.exp(17.67 * (surface_dewpoint - C2K) / (surface_dewpoint - C2K + 243.5))
    surface_specific_humidity = ETA * vapor_pressure / (surface_pressure / 100.0 - (1.0 - ETA) * vapor_pressure)

    sin_latitude = math.sin(site_latitude * C2D)
    cos_latitude = math.cos(site_latitude * C2D)
    surface_gravity = G0 * (1.0 + 5.2885e-3 * sin_latitude * sin_latitude - 5.9e-6 * math.sin(2.0 * site_latitude * C2D) ** 2)
    earth_radius = RE_EARTH / math.sqrt(
        RE_EARTH * RE_EARTH / RP_EARTH / RP_EARTH * sin_latitude * sin_latitude + cos_latitude * cos_latitude
    )
    adjusted_gravity = surface_gravity * earth_radius * earth_radius / (earth_radius + site_height) / (earth_radius + site_height)

    height_profile[0] = site_height
    pressure_profile[0] = surface_pressure
    temperature_profile[0] = surface_temp
    dewpoint_profile[0] = surface_dewpoint
    humidity_profile[0] = surface_specific_humidity
    gravity_profile[0] = adjusted_gravity
