import math
from dataclasses import replace

import numpy as np
import pytest

from rocketsim import DEFAULT_ROCKET, Conditions, Rocket, get_motor, simulate
from rocketsim.atmosphere import G0, density
from rocketsim.sim import rk4_step

MOTOR = get_motor("C6-5")


def test_rk4_is_fourth_order():
    def f(t, y):
        return (-y[0],)

    def error(h):
        y, t = (1.0,), 0.0
        for _ in range(round(1.0 / h)):
            y = rk4_step(f, t, y, h)
            t += h
        return abs(y[0] - math.exp(-1.0))

    e1, e2 = error(0.1), error(0.05)
    assert e1 < 1e-6
    assert e1 / e2 == pytest.approx(16.0, rel=0.1)


def test_vertical_flight_without_wind_stays_over_the_pad():
    f = simulate(DEFAULT_ROCKET, MOTOR, Conditions())
    assert f.landed
    assert np.abs(f.r[:, :2]).max() < 1e-9
    assert f.apogee > 100.0


def test_events_come_in_the_expected_order():
    f = simulate(DEFAULT_ROCKET, MOTOR, Conditions(wind_speed=2.0))
    names = list(f.events)
    assert names == ["rail_exit", "burnout", "apogee", "deploy", "landing"] or \
        names == ["rail_exit", "burnout", "deploy", "apogee", "landing"]
    assert f.events["burnout"].t == pytest.approx(MOTOR.burn_time)
    assert f.events["deploy"].t == pytest.approx(MOTOR.burn_time + MOTOR.delay)
    assert f.events["landing"].z == pytest.approx(0.0)


def test_landing_speed_matches_terminal_velocity_under_the_chute():
    f = simulate(DEFAULT_ROCKET, get_motor("C6-7"), Conditions())
    mass = DEFAULT_ROCKET.dry_mass + MOTOR.loaded_mass - MOTOR.propellant_mass
    cda = DEFAULT_ROCKET.chute_cd * DEFAULT_ROCKET.chute_area
    terminal = math.sqrt(2.0 * mass * G0 / (density(0.0) * cda))
    assert f.landing_speed == pytest.approx(terminal, rel=0.05)


def test_wind_pushes_the_rocket_downwind():
    west = simulate(DEFAULT_ROCKET, MOTOR, Conditions(wind_speed=4.0, wind_from=270.0))
    assert west.landing_xy[0] > 20.0
    assert abs(west.landing_xy[1]) < 1.0
    north = simulate(DEFAULT_ROCKET, MOTOR, Conditions(wind_speed=4.0, wind_from=0.0))
    assert north.landing_xy[1] < -20.0
    assert abs(north.landing_xy[0]) < 1.0


def test_stronger_wind_means_more_drift():
    weak = simulate(DEFAULT_ROCKET, MOTOR, Conditions(wind_speed=2.0))
    strong = simulate(DEFAULT_ROCKET, MOTOR, Conditions(wind_speed=6.0))
    assert strong.drift > weak.drift


def test_more_drag_lowers_apogee():
    slick = simulate(replace(DEFAULT_ROCKET, cd=0.4), MOTOR, Conditions())
    rough = simulate(replace(DEFAULT_ROCKET, cd=1.0), MOTOR, Conditions())
    assert slick.apogee > rough.apogee


def test_thin_air_at_altitude_raises_apogee():
    sea = simulate(DEFAULT_ROCKET, MOTOR, Conditions())
    high = simulate(DEFAULT_ROCKET, MOTOR, Conditions(site_elevation=2000.0))
    assert high.apogee > sea.apogee


def test_early_ejection_is_reported_and_costs_altitude():
    early = simulate(DEFAULT_ROCKET, MOTOR, Conditions(), delay=1.0)
    late = simulate(DEFAULT_ROCKET, MOTOR, Conditions(), delay=7.0)
    assert early.events["deploy"].vz > 0.0
    assert early.apogee < late.apogee


def test_rocket_without_parachute_lands_hard():
    f = simulate(replace(DEFAULT_ROCKET, chute_diameter=0.0), MOTOR, Conditions())
    assert f.landed
    assert "deploy" not in f.events
    assert f.landing_speed > 20.0


def test_overweight_rocket_stays_on_the_rail():
    f = simulate(replace(DEFAULT_ROCKET, dry_mass=3.0), MOTOR, Conditions())
    assert f.status == "rail_failure"
    assert not f.landed


def test_tilt_leans_the_flight_the_right_way():
    f = simulate(DEFAULT_ROCKET, MOTOR, Conditions(tilt=10.0, tilt_azimuth=90.0))
    assert f.landing_xy[0] > 10.0


def test_g_load_reads_one_g_in_free_fall_at_apogee_and_peaks_on_the_burn():
    f = simulate(DEFAULT_ROCKET, MOTOR, Conditions())
    g = f.g_load
    assert len(g) == len(f.t)
    apogee_idx = int(np.argmin(np.abs(f.t - f.events["apogee"].t)))
    assert g[apogee_idx] < 0.3          # coasting over the top: almost weightless
    assert f.t[int(np.argmax(g))] < MOTOR.burn_time + 0.05


def test_axial_acceleration_is_positive_under_thrust_and_negative_when_coasting():
    f = simulate(DEFAULT_ROCKET, MOTOR, Conditions())
    g = f.g_axial
    assert g[int(np.argmin(np.abs(f.t - 0.3)))] > 5.0
    assert g[int(np.argmin(np.abs(f.t - (MOTOR.burn_time + 0.3))))] < -1.0
    assert g[0] == pytest.approx(1.0, abs=0.5)      # on the pad the sensor reads about +1 g


def test_rocket_weathercocks_upwind_then_drifts_back_over_the_pad():
    f = simulate(DEFAULT_ROCKET, MOTOR, Conditions(wind_speed=3.0, wind_from=270.0))
    d = f.downrange
    assert d.min() < -20.0                           # climbed upwind of the pad
    assert d[-1] == pytest.approx(f.drift, rel=1e-6)  # ends at the landing point


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        Conditions(tilt=95.0)
    with pytest.raises(ValueError):
        Rocket(dry_mass=-1.0, diameter=0.02)
    with pytest.raises(ValueError):
        simulate(DEFAULT_ROCKET, MOTOR, Conditions(), delay=-1.0)
