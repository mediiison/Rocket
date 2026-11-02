import numpy as np
import pytest

from rocketsim import DEFAULT_ROCKET, Conditions, Uncertainty, get_motor, run_monte_carlo, simulate

MOTOR = get_motor("C6-5")
COND = Conditions(wind_speed=3.0, wind_from=270.0)


def test_same_seed_gives_identical_results():
    a = run_monte_carlo(DEFAULT_ROCKET, MOTOR, COND, runs=20, seed=7)
    b = run_monte_carlo(DEFAULT_ROCKET, MOTOR, COND, runs=20, seed=7)
    assert np.array_equal(a.landing, b.landing)
    c = run_monte_carlo(DEFAULT_ROCKET, MOTOR, COND, runs=20, seed=8)
    assert not np.array_equal(a.landing, c.landing)


def test_zero_uncertainty_reproduces_the_nominal_flight():
    zero = Uncertainty(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    res = run_monte_carlo(DEFAULT_ROCKET, MOTOR, COND, zero, runs=5, seed=1)
    nominal = simulate(DEFAULT_ROCKET, MOTOR, COND)
    assert res.landing.std(axis=0).max() < 1e-6
    assert res.landing[0, 0] == pytest.approx(nominal.landing_xy[0], abs=1e-6)
    assert res.apogee[0] == pytest.approx(nominal.apogee, abs=1e-6)


def test_more_wind_uncertainty_widens_the_scatter():
    calm = run_monte_carlo(DEFAULT_ROCKET, MOTOR, COND, Uncertainty(wind_speed_sd=0.3, wind_dir_sd=5.0),
                           runs=40, seed=3)
    gusty = run_monte_carlo(DEFAULT_ROCKET, MOTOR, COND, Uncertainty(wind_speed_sd=2.5, wind_dir_sd=40.0),
                            runs=40, seed=3)
    assert gusty.radius(95) > calm.radius(95)


def test_summary_and_radius_are_consistent():
    res = run_monte_carlo(DEFAULT_ROCKET, MOTOR, COND, runs=40, seed=5)
    sm = res.summary()
    assert sm["flights"] == 40 and sm["failures"] == 0
    assert sm["radius_50_m"] <= sm["radius_95_m"] <= sm["drift_max_m"]
    assert sm["landing_mean_x_m"] > 0.0          # wind from the west pushes landings east
    assert 0.0 <= sm["early_deploy_fraction"] <= 1.0


def test_requires_at_least_one_run():
    with pytest.raises(ValueError):
        run_monte_carlo(DEFAULT_ROCKET, MOTOR, COND, runs=0)
