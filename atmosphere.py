"""International Standard Atmosphere (ISA), 0-20 km.

All functions take geometric altitude above sea level in metres and plain floats,
so they are cheap enough to call inside the integrator loop.
"""
from __future__ import annotations

import math

G0 = 9.80665          # standard gravity, m/s^2
R_AIR = 287.053       # specific gas constant of dry air, J/(kg K)
GAMMA = 1.4           # heat capacity ratio of air
T_SEA = 288.15        # sea-level temperature, K
P_SEA = 101325.0      # sea-level pressure, Pa
LAPSE = 0.0065        # tropospheric lapse rate, K/m
H_TROPOPAUSE = 11_000.0
H_MAX = 20_000.0

_EXPONENT = G0 / (R_AIR * LAPSE)
_T_TROPOPAUSE = T_SEA - LAPSE * H_TROPOPAUSE
_P_TROPOPAUSE = P_SEA * (_T_TROPOPAUSE / T_SEA) ** _EXPONENT


def _clamp(h: float) -> float:
    return min(max(h, 0.0), H_MAX)


def temperature(h: float, delta_t: float = 0.0) -> float:
    """Air temperature in K. ``delta_t`` shifts the whole profile (hot / cold day)."""
    h = _clamp(h)
    base = T_SEA - LAPSE * h if h <= H_TROPOPAUSE else _T_TROPOPAUSE
    return base + delta_t


def pressure(h: float) -> float:
    """Static pressure in Pa."""
    h = _clamp(h)
    if h <= H_TROPOPAUSE:
        return P_SEA * (1.0 - LAPSE * h / T_SEA) ** _EXPONENT
    return _P_TROPOPAUSE * math.exp(-G0 * (h - H_TROPOPAUSE) / (R_AIR * _T_TROPOPAUSE))


def density(h: float, delta_t: float = 0.0) -> float:
    """Air density in kg/m^3 from the ideal gas law."""
    return pressure(h) / (R_AIR * temperature(h, delta_t))


def speed_of_sound(h: float, delta_t: float = 0.0) -> float:
    """Speed of sound in m/s."""
    return math.sqrt(GAMMA * R_AIR * temperature(h, delta_t))
