"""How the ejection delay changes the recovery: sweep the delay for one motor.

Run after `pip install -e .`:  python examples/delay_sweep.py
"""
from rocketsim import DEFAULT_ROCKET, Conditions, get_motor, simulate


def main() -> None:
    conditions = Conditions(wind_speed=3.0, wind_from=270.0)
    motor = get_motor("C6")
    print(f"{'delay, s':>9}{'chute vz, m/s':>15}{'apogee, m':>11}{'drift, m':>10}")
    for delay in (3, 4, 5, 5.5, 6, 7, 8, 9):
        flight = simulate(DEFAULT_ROCKET, motor, conditions, delay=delay)
        deploy = flight.events["deploy"]
        print(f"{delay:>9g}{deploy.vz:>+15.1f}{flight.apogee:>11.0f}{flight.drift:>10.0f}")


if __name__ == "__main__":
    main()
