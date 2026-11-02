"""3-DOF flight simulation: thrust, quadratic drag, wind, parachute, RK4 integration.

Frame: x = East, y = North, z = up, origin at the launch pad.
The rocket is a point mass whose axis follows the air-relative velocity once it has left
the rail ("perfect weathercocking"), so thrust always pushes along the airflow.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from .atmosphere import G0, density
from .motor import Motor
from .rocket import Rocket

State = tuple[float, float, float, float, float, float]  # x, y, z, vx, vy, vz


@dataclass(frozen=True)
class Conditions:
    """Launch site and weather."""

    wind_speed: float = 0.0        # m/s, measured at 10 m above ground
    wind_from: float = 270.0       # deg, direction the wind blows FROM (0 = N, 90 = E)
    wind_exponent: float = 1.0 / 7.0   # power-law wind shear exponent
    tilt: float = 0.0              # deg, launch rail tilt from vertical
    tilt_azimuth: float = 0.0      # deg, compass bearing the rail leans towards
    rail_length: float = 1.0       # m
    site_elevation: float = 0.0    # m above sea level
    delta_t: float = 0.0           # K, offset from the standard-day temperature

    def __post_init__(self) -> None:
        if self.wind_speed < 0.0:
            raise ValueError("wind_speed cannot be negative")
        if not 0.0 <= self.tilt < 90.0:
            raise ValueError("tilt must be in [0, 90) degrees")
        if self.rail_length <= 0.0:
            raise ValueError("rail_length must be positive")

    def launch_direction(self) -> tuple[float, float, float]:
        tilt, az = math.radians(self.tilt), math.radians(self.tilt_azimuth)
        return (math.sin(tilt) * math.sin(az), math.sin(tilt) * math.cos(az), math.cos(tilt))

    def wind_toward(self) -> tuple[float, float]:
        """Unit vector (east, north) the air moves towards."""
        bearing = math.radians(self.wind_from + 180.0)
        return (math.sin(bearing), math.cos(bearing))


@dataclass(frozen=True)
class Event:
    name: str
    t: float
    x: float
    y: float
    z: float
    speed: float
    vz: float


@dataclass
class Flight:
    """Result of one simulated flight."""

    rocket: Rocket
    motor: Motor
    conditions: Conditions
    delay: float
    t: np.ndarray                  # (N,) s
    r: np.ndarray                  # (N, 3) m
    v: np.ndarray                  # (N, 3) m/s
    events: dict[str, Event]
    status: str                    # "landed", "rail_failure" or "timeout"

    @property
    def landed(self) -> bool:
        return self.status == "landed"

    @property
    def altitude(self) -> np.ndarray:
        return self.r[:, 2]

    @property
    def speed(self) -> np.ndarray:
        return np.linalg.norm(self.v, axis=1)

    @property
    def ground_distance(self) -> np.ndarray:
        """Horizontal distance from the pad at every sample."""
        return np.hypot(self.r[:, 0], self.r[:, 1])

    @property
    def downrange(self) -> np.ndarray:
        """Signed horizontal distance from the pad along the pad-to-landing direction.

        Negative values mean the rocket is upwind of the pad (it weathercocks into the wind
        on the way up and is carried back over the pad by the parachute).
        """
        x, y = self.landing_xy
        norm = math.hypot(x, y)
        ux, uy = (x / norm, y / norm) if norm > 1.0 else self.conditions.wind_toward()
        return self.r[:, 0] * ux + self.r[:, 1] * uy

    def _proper_acceleration(self) -> np.ndarray:
        """What an accelerometer on board would read, as vectors in m/s^2 (dv/dt + g)."""
        t, v = self.t, self.v
        keep = np.concatenate(([True], np.diff(t) > 0.0))
        tk, vk = t[keep], v[keep]
        if len(tk) < 2:
            a = np.zeros((len(t), 3))
        else:
            ak = np.gradient(vk, tk, axis=0)
            a = np.column_stack([np.interp(t, tk, ak[:, i]) for i in range(3)])
        a[:, 2] += G0
        return a

    @property
    def g_load(self) -> np.ndarray:
        """Magnitude of the accelerometer reading, in g."""
        return np.linalg.norm(self._proper_acceleration(), axis=1) / G0

    @property
    def g_axial(self) -> np.ndarray:
        """Accelerometer reading along the flight path, in g: positive under thrust,
        negative when drag is braking the rocket."""
        speed = self.speed
        launch = np.array(self.conditions.launch_direction())
        direction = np.where(speed[:, None] > 0.5, self.v / np.maximum(speed, 1e-9)[:, None], launch)
        return np.einsum("ij,ij->i", self._proper_acceleration(), direction) / G0

    @property
    def apogee(self) -> float:
        ev = self.events.get("apogee")
        return ev.z if ev else float(self.altitude.max())

    @property
    def landing_xy(self) -> tuple[float, float]:
        return float(self.r[-1, 0]), float(self.r[-1, 1])

    @property
    def drift(self) -> float:
        return math.hypot(*self.landing_xy)

    @property
    def landing_speed(self) -> float:
        return float(self.speed[-1])

    def summary(self) -> dict[str, Optional[float]]:
        ev = self.events
        out: dict[str, Optional[float]] = {
            "apogee_m": self.apogee,
            "time_to_apogee_s": ev["apogee"].t if "apogee" in ev else None,
            "max_speed_ms": float(self.speed.max()),
            "max_g": float(self.g_load.max()),
            "rail_exit_speed_ms": ev["rail_exit"].speed if "rail_exit" in ev else None,
            "deploy_time_s": ev["deploy"].t if "deploy" in ev else None,
            "deploy_altitude_m": ev["deploy"].z if "deploy" in ev else None,
            "deploy_vz_ms": ev["deploy"].vz if "deploy" in ev else None,
            "landing_speed_ms": self.landing_speed,
            "flight_time_s": float(self.t[-1]),
            "drift_m": self.drift,
            "landing_x_m": self.landing_xy[0],
            "landing_y_m": self.landing_xy[1],
        }
        return out


def rk4_step(f: Callable[[float, Sequence[float]], Sequence[float]],
             t: float, y: Sequence[float], h: float) -> tuple[float, ...]:
    """One classical Runge-Kutta step for y' = f(t, y)."""
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, [a + 0.5 * h * b for a, b in zip(y, k1)])
    k3 = f(t + 0.5 * h, [a + 0.5 * h * b for a, b in zip(y, k2)])
    k4 = f(t + h, [a + h * b for a, b in zip(y, k3)])
    return tuple(a + h / 6.0 * (b + 2.0 * c + 2.0 * d + e)
                 for a, b, c, d, e in zip(y, k1, k2, k3, k4))


def _lerp(a: Sequence[float], b: Sequence[float], frac: float) -> tuple[float, ...]:
    return tuple(p + frac * (q - p) for p, q in zip(a, b))


def simulate(rocket: Rocket, motor: Motor, conditions: Optional[Conditions] = None, *,
             delay: Optional[float] = None, dt: float = 0.005, t_max: float = 900.0) -> Flight:
    """Fly ``rocket`` on ``motor`` and return the full trajectory.

    ``delay`` overrides the motor's ejection delay (seconds after burnout). ``dt`` is the
    fine time step used on the rail, during the burn and while the parachute opens; coast
    and descent use larger steps automatically.
    """
    cond = conditions or Conditions()
    delay = motor.delay if delay is None else float(delay)
    if delay < 0.0:
        raise ValueError("delay cannot be negative")

    t_burn = motor.burn_time
    has_chute = rocket.has_parachute
    t_deploy = t_burn + delay if has_chute else math.inf
    t_inflated = t_deploy + rocket.chute_inflation_time
    marks = sorted({t_burn, t_deploy, t_inflated})

    dx, dy, dz = cond.launch_direction()
    ex, ey = cond.wind_toward()
    w_ref, alpha = cond.wind_speed, cond.wind_exponent
    elev, d_temp = cond.site_elevation, cond.delta_t
    dry = rocket.dry_mass
    cda_body = rocket.cd * rocket.ref_area
    cda_chute = rocket.chute_cd * rocket.chute_area if has_chute else cda_body
    inflation = rocket.chute_inflation_time
    on_rail = True

    def cda_at(t: float) -> float:
        if t <= t_deploy:
            return cda_body
        if t >= t_inflated:
            return cda_chute
        return cda_body + (cda_chute - cda_body) * (t - t_deploy) / inflation

    def deriv(t: float, s: Sequence[float]) -> tuple[float, ...]:
        _, _, pz, vx, vy, vz = s
        mass = dry + motor.mass(t)
        height = pz if pz > 1.0 else 1.0
        wind = w_ref * (height * 0.1) ** alpha
        rvx, rvy, rvz = vx - wind * ex, vy - wind * ey, vz      # velocity relative to the air
        airspeed = math.sqrt(rvx * rvx + rvy * rvy + rvz * rvz)
        k = 0.5 * density(elev + pz, d_temp) * cda_at(t) * airspeed
        thrust = motor.thrust(t)
        if on_rail or airspeed < 1.0:
            ux, uy, uz = dx, dy, dz
        else:
            ux, uy, uz = rvx / airspeed, rvy / airspeed, rvz / airspeed
        ax = (thrust * ux - k * rvx) / mass
        ay = (thrust * uy - k * rvy) / mass
        az = (thrust * uz - k * rvz) / mass - G0
        if on_rail:                                  # the rail only allows motion along itself
            along = ax * dx + ay * dy + az * dz
            if along < 0.0 and vx * dx + vy * dy + vz * dz <= 0.0:
                along = 0.0                          # still sitting on the pad
            ax, ay, az = along * dx, along * dy, along * dz
        return (vx, vy, vz, ax, ay, az)

    def event(name: str, t: float, s: Sequence[float]) -> Event:
        return Event(name, t, s[0], s[1], s[2], math.sqrt(s[3] ** 2 + s[4] ** 2 + s[5] ** 2), s[5])

    fine, medium, coarse = dt, 4.0 * dt, 10.0 * dt
    t = 0.0
    s: tuple[float, ...] = (0.0,) * 6
    ts, states = [t], [s]
    events: dict[str, Event] = {}
    status = "timeout"

    while t < t_max:
        if on_rail or t < t_burn or t_deploy - 1e-9 <= t < t_inflated + 0.5:
            h = fine
        elif t < t_deploy:
            h = medium
        else:
            h = coarse
        t_new = t + h
        for mark in marks:                           # land exactly on burnout / deployment
            if mark > t + 1e-9:
                if mark < t_new:
                    h, t_new = mark - t, mark
                break

        s_new = rk4_step(deriv, t, s, h)

        if on_rail:
            along = max(s_new[0] * dx + s_new[1] * dy + s_new[2] * dz, 0.0)
            v_along = max(s_new[3] * dx + s_new[4] * dy + s_new[5] * dz, 0.0)
            s_new = (along * dx, along * dy, along * dz, v_along * dx, v_along * dy, v_along * dz)
            if along >= cond.rail_length:
                on_rail = False
                events["rail_exit"] = event("rail_exit", t_new, s_new)
            elif t_new > t_burn:
                ts.append(t_new)
                states.append(s_new)
                status = "rail_failure"              # burnt out without clearing the rail
                break

        if "burnout" not in events and t_new >= t_burn - 1e-9:
            events["burnout"] = event("burnout", t_new, s_new)
        if has_chute and "deploy" not in events and t_new >= t_deploy - 1e-9:
            events["deploy"] = event("deploy", t_new, s_new)
        if "apogee" not in events and not on_rail and s[5] > 0.0 >= s_new[5]:
            frac = s[5] / (s[5] - s_new[5])
            events["apogee"] = event("apogee", t + frac * h, _lerp(s, s_new, frac))

        if not on_rail and s_new[2] < 0.0:           # touched the ground inside this step
            frac = s[2] / (s[2] - s_new[2])
            s_land = _lerp(s, s_new, frac)
            t_land = t + frac * h
            ts.append(t_land)
            states.append(s_land)
            events["landing"] = event("landing", t_land, s_land)
            status = "landed"
            break

        ts.append(t_new)
        states.append(s_new)
        t, s = t_new, s_new

    data = np.array(states)
    flight = Flight(rocket, motor, cond, delay, np.array(ts), data[:, :3], data[:, 3:],
                    dict(sorted(events.items(), key=lambda kv: kv[1].t)), status)
    return flight
