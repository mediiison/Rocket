"""Airframe and recovery parameters."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Rocket:
    """A model rocket as seen by the simulator (SI units).

    ``dry_mass`` excludes the motor. Body drag uses the frontal area of ``diameter``;
    the parachute is a flat disc of ``chute_diameter`` (0 means no parachute).
    """

    dry_mass: float                    # kg
    diameter: float                    # m, body tube
    cd: float = 0.65                   # body drag coefficient (subsonic)
    chute_diameter: float = 0.30       # m, 0 = no parachute
    chute_cd: float = 0.75
    chute_inflation_time: float = 0.5  # s, linear ramp of drag area after ejection
    name: str = "Model rocket"

    def __post_init__(self) -> None:
        if self.dry_mass <= 0.0 or self.diameter <= 0.0:
            raise ValueError("dry_mass and diameter must be positive")
        if self.cd <= 0.0 or self.chute_cd <= 0.0:
            raise ValueError("drag coefficients must be positive")
        if self.chute_diameter < 0.0 or self.chute_inflation_time < 0.0:
            raise ValueError("chute_diameter and chute_inflation_time cannot be negative")

    @property
    def ref_area(self) -> float:
        return math.pi * self.diameter ** 2 / 4.0

    @property
    def chute_area(self) -> float:
        return math.pi * self.chute_diameter ** 2 / 4.0

    @property
    def has_parachute(self) -> bool:
        return self.chute_diameter > 0.0


DEFAULT_ROCKET = Rocket(dry_mass=0.055, diameter=0.0247, name="Generic 24 mm rocket")
