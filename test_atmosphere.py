import pytest

from rocketsim import atmosphere as atm


def test_sea_level_values():
    assert atm.temperature(0) == pytest.approx(288.15)
    assert atm.pressure(0) == pytest.approx(101325.0)
    assert atm.density(0) == pytest.approx(1.225, abs=1e-3)
    assert atm.speed_of_sound(0) == pytest.approx(340.3, abs=0.1)


def test_tropopause_values():
    assert atm.temperature(11_000) == pytest.approx(216.65)
    assert atm.pressure(11_000) == pytest.approx(22632.0, rel=2e-3)
    assert atm.density(11_000) == pytest.approx(0.3639, rel=2e-3)


def test_density_decreases_with_altitude():
    values = [atm.density(h) for h in range(0, 20_001, 1000)]
    assert all(a > b for a, b in zip(values, values[1:]))


def test_temperature_offset_makes_air_thinner():
    assert atm.density(0, delta_t=15.0) < atm.density(0)
    assert atm.density(0, delta_t=-15.0) > atm.density(0)


def test_altitude_is_clamped():
    assert atm.density(-50.0) == atm.density(0.0)
    assert atm.density(50_000.0) == atm.density(20_000.0)
