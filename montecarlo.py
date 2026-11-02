"""Monte Carlo dispersion: where does the rocket land when nothing is exactly nominal?"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from .motor import Motor
from .rocket import Rocket
from .sim import Conditions, simulate


@dataclass(frozen=True)
class Uncertainty:
    """One-sigma spreads of the inputs (all normally distributed)."""

    wind_speed_sd: float = 1.5     # m/s
    wind_dir_sd: float = 20.0      # deg
    tilt_sd: float = 1.0           # deg, rail tilt error in each horizontal axis
    thrust_sd: float = 0.05        # relative, motor-to-motor thrust variation
    cd_sd: float = 0.08            # relative, body drag coefficient
    chute_cd_sd: float = 0.05      # relative, parachute drag coefficient
    delay_sd: float = 0.3          # s, ejection delay tolerance


@dataclass
class MCResult:
    runs: int
    seed: Optional[int]
    motor_name: str
    conditions: Conditions
    uncertainty: Uncertainty
    landing: np.ndarray            # (n, 2) east / north of the pad, m
    apogee: np.ndarray             # (n,) m
    flight_time: np.ndarray        # (n,) s
    landing_speed: np.ndarray      # (n,) m/s
    deploy_vz: np.ndarray          # (n,) m/s vertical speed when the chute fires (nan: no chute)
    failures: int                  # flights that never cleared the rail / never landed

    @property
    def drift(self) -> np.ndarray:
        return np.hypot(self.landing[:, 0], self.landing[:, 1])

    def radius(self, percent: float = 95.0) -> float:
        """Radius around the pad containing ``percent`` % of the landings."""
        return float(np.percentile(self.drift, percent)) if len(self.drift) else math.nan

    @property
    def early_deploy_fraction(self) -> float:
        """Share of flights where the chute fired while the rocket was still climbing."""
        vz = self.deploy_vz[~np.isnan(self.deploy_vz)]
        return float(np.mean(vz > 0.0)) if len(vz) else math.nan

    def summary(self) -> dict[str, float]:
        if not len(self.drift):
            return {"flights": 0, "failures": self.failures}
        return {
            "flights": len(self.drift),
            "failures": self.failures,
            "apogee_mean_m": float(self.apogee.mean()),
            "apogee_sd_m": float(self.apogee.std(ddof=1)) if len(self.apogee) > 1 else 0.0,
            "landing_mean_x_m": float(self.landing[:, 0].mean()),
            "landing_mean_y_m": float(self.landing[:, 1].mean()),
            "drift_mean_m": float(self.drift.mean()),
            "radius_50_m": self.radius(50.0),
            "radius_95_m": self.radius(95.0),
            "drift_max_m": float(self.drift.max()),
            "flight_time_mean_s": float(self.flight_time.mean()),
            "landing_speed_mean_ms": float(self.landing_speed.mean()),
            "early_deploy_fraction": self.early_deploy_fraction,
        }


def run_monte_carlo(rocket: Rocket, motor: Motor, conditions: Optional[Conditions] = None,
                    uncertainty: Optional[Uncertainty] = None, *, runs: int = 300,
                    seed: Optional[int] = None, delay: Optional[float] = None,
                    dt: float = 0.005) -> MCResult:
    """Simulate ``runs`` flights with randomly perturbed inputs around the nominal ones."""
    if runs < 1:
        raise ValueError("runs must be at least 1")
    cond = conditions or Conditions()
    unc = uncertainty or Uncertainty()
    nominal_delay = motor.delay if delay is None else float(delay)
    rng = np.random.default_rng(seed)

    tilt_e = cond.tilt * math.sin(math.radians(cond.tilt_azimuth))
    tilt_n = cond.tilt * math.cos(math.radians(cond.tilt_azimuth))

    landing, apogee, ftime, lspeed, dvz = [], [], [], [], []
    failures = 0
    for _ in range(runs):
        wind = max(0.0, rng.normal(cond.wind_speed, unc.wind_speed_sd))
        wind_from = rng.normal(cond.wind_from, unc.wind_dir_sd) % 360.0
        te = tilt_e + rng.normal(0.0, unc.tilt_sd)
        tn = tilt_n + rng.normal(0.0, unc.tilt_sd)
        thrust_scale = max(0.5, rng.normal(1.0, unc.thrust_sd))
        cd = max(0.05, rng.normal(rocket.cd, rocket.cd * unc.cd_sd))
        chute_cd = max(0.05, rng.normal(rocket.chute_cd, rocket.chute_cd * unc.chute_cd_sd))
        delay_i = max(0.0, rng.normal(nominal_delay, unc.delay_sd))

        flight = simulate(
            replace(rocket, cd=cd, chute_cd=chute_cd),
            motor.scaled(thrust_scale),
            replace(cond, wind_speed=wind, wind_from=wind_from,
                    tilt=min(math.hypot(te, tn), 60.0),
                    tilt_azimuth=math.degrees(math.atan2(te, tn)) % 360.0),
            delay=delay_i, dt=dt,
        )
        if not flight.landed:
            failures += 1
            continue
        landing.append(flight.landing_xy)
        apogee.append(flight.apogee)
        ftime.append(float(flight.t[-1]))
        lspeed.append(flight.landing_speed)
        dep = flight.events.get("deploy")
        dvz.append(dep.vz if dep else math.nan)

    return MCResult(
        runs=runs, seed=seed, motor_name=motor.name, conditions=cond, uncertainty=unc,
        landing=np.array(landing).reshape(-1, 2), apogee=np.array(apogee),
        flight_time=np.array(ftime), landing_speed=np.array(lspeed),
        deploy_vz=np.array(dvz), failures=failures,
    )
