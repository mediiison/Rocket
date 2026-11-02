"""Solid-propellant motor model: thrust curve, burnout mass and ejection delay."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field, replace

# Normalised thrust curve shared by the presets: (fraction of burn time, thrust / peak).
# Typical black-powder shape: sharp spike, fast decay, long plateau, short tail-off.
_SHAPE = (
    (0.00, 0.00),
    (0.03, 0.30),
    (0.12, 0.85),
    (0.18, 1.00),
    (0.25, 0.80),
    (0.35, 0.40),
    (0.50, 0.30),
    (0.70, 0.27),
    (0.90, 0.26),
    (0.97, 0.24),
    (1.00, 0.00),
)


@dataclass(frozen=True)
class Motor:
    """A motor described by a piecewise-linear thrust curve.

    ``loaded_mass`` is the mass at ignition (case, propellant, delay grain, charge);
    only ``propellant_mass`` is burnt, in proportion to the impulse delivered so far.
    ``delay`` is the time between burnout and the ejection charge firing.
    """

    name: str
    times: tuple[float, ...]       # s, starts at 0, strictly increasing
    thrusts: tuple[float, ...]     # N, same length as times
    propellant_mass: float         # kg
    loaded_mass: float             # kg
    delay: float = 0.0             # s
    _cum: tuple[float, ...] = field(init=False, repr=False, compare=False, default=())

    def __post_init__(self) -> None:
        times = tuple(float(t) for t in self.times)
        thrusts = tuple(float(f) for f in self.thrusts)
        if len(times) != len(thrusts) or len(times) < 2:
            raise ValueError("times and thrusts must have the same length (at least 2)")
        if times[0] != 0.0 or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("times must start at 0 and be strictly increasing")
        if any(f < 0.0 for f in thrusts):
            raise ValueError("thrust cannot be negative")
        if not 0.0 < self.propellant_mass < self.loaded_mass:
            raise ValueError("need 0 < propellant_mass < loaded_mass")
        if self.delay < 0.0:
            raise ValueError("delay cannot be negative")
        cum = [0.0]
        for (t0, f0), (t1, f1) in zip(zip(times, thrusts), zip(times[1:], thrusts[1:])):
            cum.append(cum[-1] + 0.5 * (f0 + f1) * (t1 - t0))
        if cum[-1] <= 0.0:
            raise ValueError("thrust curve has zero total impulse")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "thrusts", thrusts)
        object.__setattr__(self, "_cum", tuple(cum))

    @property
    def burn_time(self) -> float:
        return self.times[-1]

    @property
    def total_impulse(self) -> float:
        return self._cum[-1]

    @property
    def average_thrust(self) -> float:
        return self.total_impulse / self.burn_time

    @property
    def peak_thrust(self) -> float:
        return max(self.thrusts)

    def thrust(self, t: float) -> float:
        """Thrust in N at time ``t`` after ignition."""
        if t <= 0.0 or t >= self.times[-1]:
            return 0.0
        i = bisect_right(self.times, t) - 1
        t0, t1 = self.times[i], self.times[i + 1]
        f0, f1 = self.thrusts[i], self.thrusts[i + 1]
        return f0 + (f1 - f0) * (t - t0) / (t1 - t0)

    def impulse_at(self, t: float) -> float:
        """Impulse delivered between ignition and ``t``, in N*s (exact for the linear curve)."""
        if t <= 0.0:
            return 0.0
        if t >= self.times[-1]:
            return self._cum[-1]
        i = bisect_right(self.times, t) - 1
        t0, t1 = self.times[i], self.times[i + 1]
        f0, f1 = self.thrusts[i], self.thrusts[i + 1]
        dt = t - t0
        slope = (f1 - f0) / (t1 - t0)
        return self._cum[i] + f0 * dt + 0.5 * slope * dt * dt

    def mass(self, t: float) -> float:
        """Motor mass in kg at time ``t``: propellant leaves in step with the impulse."""
        burnt = self.propellant_mass * self.impulse_at(t) / self.total_impulse
        return self.loaded_mass - burnt

    def scaled(self, factor: float) -> "Motor":
        """Same motor with thrust multiplied by ``factor`` (batch-to-batch variation)."""
        return replace(self, thrusts=tuple(f * factor for f in self.thrusts))

    def with_delay(self, delay: float) -> "Motor":
        return replace(self, delay=delay)


def _make(name: str, total_impulse: float, burn_time: float,
          propellant_g: float, loaded_g: float, delay: float) -> Motor:
    area = sum(0.5 * (f0 + f1) * (t1 - t0)
               for (t0, f0), (t1, f1) in zip(_SHAPE, _SHAPE[1:]))
    peak = total_impulse / (burn_time * area)
    return Motor(
        name=name,
        times=tuple(tau * burn_time for tau, _ in _SHAPE),
        thrusts=tuple(f * peak for _, f in _SHAPE),
        propellant_mass=propellant_g / 1000.0,
        loaded_mass=loaded_g / 1000.0,
        delay=delay,
    )


# Class-level numbers (impulse, burn time, masses) follow common 18-24 mm hobby motors,
# but the curve shape is generic. Treat them as realistic stand-ins, not certified data.
MOTORS: dict[str, Motor] = {
    "B6": _make("B6", total_impulse=5.0, burn_time=0.85, propellant_g=6.2, loaded_g=16.7, delay=4.0),
    "C6": _make("C6", total_impulse=8.8, burn_time=1.85, propellant_g=12.3, loaded_g=24.0, delay=5.0),
    "D12": _make("D12", total_impulse=16.8, burn_time=1.60, propellant_g=21.0, loaded_g=44.0, delay=5.0),
}


def get_motor(spec: str) -> Motor:
    """Look a preset up by name, optionally with a delay: ``"C6"``, ``"C6-7"``, ``"d12-3"``."""
    base, sep, delay_text = spec.strip().upper().partition("-")
    if base not in MOTORS:
        raise ValueError(f"unknown motor {spec!r}; available: {', '.join(sorted(MOTORS))}")
    motor = MOTORS[base]
    if not sep:
        return motor
    try:
        delay = float(delay_text)
    except ValueError:
        raise ValueError(f"bad delay in motor spec {spec!r}") from None
    return replace(motor, name=f"{base}-{delay:g}", delay=delay)
