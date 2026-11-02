"""rocketsim - a small 3-DOF model rocket flight simulator with Monte Carlo landing dispersion."""

from .montecarlo import MCResult, Uncertainty, run_monte_carlo
from .motor import MOTORS, Motor, get_motor
from .rocket import DEFAULT_ROCKET, Rocket
from .sim import Conditions, Event, Flight, simulate

__version__ = "0.1.0"

__all__ = [
    "Conditions", "DEFAULT_ROCKET", "Event", "Flight", "MCResult", "MOTORS", "Motor",
    "Rocket", "Uncertainty", "get_motor", "run_monte_carlo", "simulate",
]
