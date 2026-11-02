"""Command line interface: ``rocketsim fly``, ``rocketsim mc``, ``rocketsim motors``."""
from __future__ import annotations

import argparse
import csv
import math
import sys
from typing import Optional, Sequence

from .montecarlo import Uncertainty, run_monte_carlo
from .motor import MOTORS, get_motor
from .rocket import Rocket
from .sim import Conditions, Flight, simulate


def _common() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    r = p.add_argument_group("rocket and motor")
    r.add_argument("--motor", default="C6-5", help="preset with optional delay, e.g. C6-5, B6-4, D12-3 (default C6-5)")
    r.add_argument("--delay", type=float, default=None, help="ejection delay in s, overrides the one in --motor")
    r.add_argument("--mass", type=float, default=55.0, help="dry mass without the motor, g (default 55)")
    r.add_argument("--diameter", type=float, default=24.7, help="body diameter, mm (default 24.7)")
    r.add_argument("--cd", type=float, default=0.65, help="body drag coefficient (default 0.65)")
    r.add_argument("--chute", type=float, default=30.0, help="parachute diameter, cm, 0 = none (default 30)")
    r.add_argument("--chute-cd", type=float, default=0.75, help="parachute drag coefficient (default 0.75)")
    w = p.add_argument_group("launch conditions")
    w.add_argument("--wind", type=float, default=3.0, help="wind speed at 10 m, m/s (default 3)")
    w.add_argument("--wind-from", type=float, default=270.0, help="direction the wind blows FROM, deg, 0 = N, 90 = E (default 270)")
    w.add_argument("--tilt", type=float, default=0.0, help="rail tilt from vertical, deg (default 0)")
    w.add_argument("--tilt-az", type=float, default=None, help="compass bearing the rail leans to, deg (default: into the wind)")
    w.add_argument("--rail", type=float, default=1.0, help="rail length, m (default 1)")
    w.add_argument("--elevation", type=float, default=0.0, help="launch site elevation, m (default 0)")
    w.add_argument("--temp-offset", type=float, default=0.0, help="temperature offset from a standard day, K (default 0)")
    return p


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rocketsim", description="Model rocket flight simulator.")
    sub = parser.add_subparsers(dest="command", required=True)
    common = _common()

    fly = sub.add_parser("fly", parents=[common], help="simulate one flight and plot it")
    fly.add_argument("--out", default="flight.png", help="figure file (default flight.png)")
    fly.add_argument("--no-plot", action="store_true", help="print the report only")
    fly.add_argument("--csv", default=None, help="also write the trajectory to this CSV file")

    mc = sub.add_parser("mc", parents=[common], help="Monte Carlo landing dispersion")
    mc.add_argument("--runs", type=int, default=300, help="number of flights (default 300)")
    mc.add_argument("--seed", type=int, default=None, help="random seed for reproducible runs")
    mc.add_argument("--out", default="landing.png", help="figure file (default landing.png)")
    mc.add_argument("--no-plot", action="store_true", help="print the report only")
    d = Uncertainty()
    mc.add_argument("--wind-sd", type=float, default=d.wind_speed_sd, help="wind speed sd, m/s")
    mc.add_argument("--wind-dir-sd", type=float, default=d.wind_dir_sd, help="wind direction sd, deg")
    mc.add_argument("--tilt-sd", type=float, default=d.tilt_sd, help="rail tilt error sd, deg")
    mc.add_argument("--thrust-sd", type=float, default=d.thrust_sd, help="relative thrust sd")
    mc.add_argument("--delay-sd", type=float, default=d.delay_sd, help="ejection delay sd, s")

    sub.add_parser("motors", help="list the motor presets")
    return parser


def _build(args):
    rocket = Rocket(dry_mass=args.mass / 1000.0, diameter=args.diameter / 1000.0, cd=args.cd,
                    chute_diameter=args.chute / 100.0, chute_cd=args.chute_cd)
    motor = get_motor(args.motor)
    cond = Conditions(
        wind_speed=args.wind, wind_from=args.wind_from, tilt=args.tilt,
        tilt_azimuth=args.wind_from if args.tilt_az is None else args.tilt_az,
        rail_length=args.rail, site_elevation=args.elevation, delta_t=args.temp_offset,
    )
    return rocket, motor, cond


def _header_lines(rocket: Rocket, motor, cond: Conditions, delay: float) -> list[str]:
    chute = f"{rocket.chute_diameter * 100:.0f} cm chute" if rocket.has_parachute else "no chute"
    return [
        f"Rocket  {rocket.dry_mass * 1000:.0f} g dry, {rocket.diameter * 1000:.1f} mm, Cd {rocket.cd:.2f}, {chute}",
        f"Motor   {motor.name}: {motor.total_impulse:.1f} Ns, peak {motor.peak_thrust:.1f} N, "
        f"burn {motor.burn_time:.2f} s, delay {delay:g} s",
        f"Site    wind {cond.wind_speed:.1f} m/s from {cond.wind_from:.0f} deg, rail {cond.rail_length:g} m, "
        f"tilt {cond.tilt:.1f} deg, elevation {cond.site_elevation:g} m",
    ]


def _bearing(x: float, y: float) -> float:
    return math.degrees(math.atan2(x, y)) % 360.0


def format_flight_report(flight: Flight) -> str:
    ev, s = flight.events, flight.summary()
    lines = _header_lines(flight.rocket, flight.motor, flight.conditions, flight.delay)
    lines.append("")
    if flight.status == "rail_failure":
        lines.append("The rocket burnt out without clearing the launch rail: too heavy for this motor.")
        return "\n".join(lines)
    if "rail_exit" in ev:
        e = ev["rail_exit"]
        lines.append(f"Rail exit     t = {e.t:5.2f} s   speed {e.speed:5.1f} m/s")
    if "burnout" in ev:
        e = ev["burnout"]
        lines.append(f"Burnout       t = {e.t:5.2f} s   alt {e.z:6.1f} m   speed {e.speed:5.1f} m/s")
    if "apogee" in ev:
        e = ev["apogee"]
        lines.append(f"Apogee        t = {e.t:5.2f} s   alt {e.z:6.1f} m")
    if "deploy" in ev:
        e = ev["deploy"]
        lines.append(f"Chute out     t = {e.t:5.2f} s   alt {e.z:6.1f} m   vertical speed {e.vz:+.1f} m/s")
    if "landing" in ev:
        e = ev["landing"]
        x, y = flight.landing_xy
        lines.append(f"Landing       t = {e.t:5.2f} s   {flight.drift:.0f} m from the pad, "
                     f"bearing {_bearing(x, y):.0f} deg, speed {flight.landing_speed:.1f} m/s")
    lines.append("")
    lines.append(f"Max speed {s['max_speed_ms']:.1f} m/s, max load {s['max_g']:.1f} g, "
                 f"flight time {s['flight_time_s']:.1f} s")
    dep, apo = ev.get("deploy"), ev.get("apogee")
    if dep and apo:
        if dep.vz > 3.0:
            lines.append(f"Note: the chute fires {apo.t - dep.t:.1f} s BEFORE apogee, "
                         f"while still climbing at {dep.vz:.0f} m/s. A longer delay would fix that.")
        elif dep.vz < -10.0:
            lines.append(f"Note: the chute fires {dep.t - apo.t:.1f} s AFTER apogee, "
                         f"at {-dep.vz:.0f} m/s descent. A shorter delay would fix that.")
    if flight.status == "timeout":
        lines.append("Warning: the flight did not end within the time limit.")
    return "\n".join(lines)


def write_csv(flight: Flight, path: str) -> None:
    g = flight.g_load
    speed = flight.speed
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "x", "y", "z", "vx", "vy", "vz", "speed", "g_load"])
        for i in range(len(flight.t)):
            w.writerow([f"{flight.t[i]:.4f}", *(f"{c:.4f}" for c in flight.r[i]),
                        *(f"{c:.4f}" for c in flight.v[i]), f"{speed[i]:.4f}", f"{g[i]:.4f}"])


def _cmd_fly(args) -> int:
    rocket, motor, cond = _build(args)
    flight = simulate(rocket, motor, cond, delay=args.delay)
    print(format_flight_report(flight))
    if args.csv:
        write_csv(flight, args.csv)
        print(f"\nTrajectory written to {args.csv}")
    if not args.no_plot:
        from .plotting import plot_flight
        print(f"Figure written to {plot_flight(flight, args.out)}")
    return 0


def _cmd_mc(args) -> int:
    rocket, motor, cond = _build(args)
    unc = Uncertainty(wind_speed_sd=args.wind_sd, wind_dir_sd=args.wind_dir_sd, tilt_sd=args.tilt_sd,
                      thrust_sd=args.thrust_sd, delay_sd=args.delay_sd)
    delay = motor.delay if args.delay is None else args.delay
    result = run_monte_carlo(rocket, motor, cond, unc, runs=args.runs, seed=args.seed, delay=args.delay)
    sm = result.summary()
    print("\n".join(_header_lines(rocket, motor, cond, delay)))
    print(f"MC      {args.runs} flights, seed {args.seed}, wind sd {unc.wind_speed_sd:.1f} m/s / "
          f"{unc.wind_dir_sd:.0f} deg, thrust sd {unc.thrust_sd * 100:.0f}%, delay sd {unc.delay_sd:.1f} s")
    print()
    if sm["flights"] == 0:
        print("No flight landed: nothing to report.")
        return 1
    print(f"Apogee          mean {sm['apogee_mean_m']:.0f} m, sd {sm['apogee_sd_m']:.0f} m")
    print(f"Landing centre  {sm['landing_mean_x_m']:+.0f} m east, {sm['landing_mean_y_m']:+.0f} m north")
    print(f"Distance        median {sm['radius_50_m']:.0f} m, 95% within {sm['radius_95_m']:.0f} m, "
          f"worst {sm['drift_max_m']:.0f} m")
    print(f"Landing speed   mean {sm['landing_speed_mean_ms']:.1f} m/s, flight time {sm['flight_time_mean_s']:.0f} s")
    if not math.isnan(sm["early_deploy_fraction"]):
        print(f"Chute fires before apogee in {sm['early_deploy_fraction'] * 100:.0f}% of flights")
    if sm["failures"]:
        print(f"{sm['failures']} flight(s) never cleared the rail and are excluded")
    if not args.no_plot:
        from .plotting import plot_monte_carlo
        print(f"\nFigure written to {plot_monte_carlo(result, args.out)}")
    return 0


def _cmd_motors(_args) -> int:
    print(f"{'name':<6}{'impulse':>10}{'avg thrust':>12}{'peak':>8}{'burn':>8}{'loaded':>9}{'default delay':>15}")
    for name, m in MOTORS.items():
        print(f"{name:<6}{m.total_impulse:>8.1f} Ns{m.average_thrust:>10.1f} N{m.peak_thrust:>6.1f} N"
              f"{m.burn_time:>6.2f} s{m.loaded_mass * 1000:>7.1f} g{m.delay:>13g} s")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "fly":
            return _cmd_fly(args)
        if args.command == "mc":
            return _cmd_mc(args)
        return _cmd_motors(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
