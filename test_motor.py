import pytest

from rocketsim import MOTORS, Motor, get_motor


@pytest.mark.parametrize("name,impulse,burn", [("B6", 5.0, 0.85), ("C6", 8.8, 1.85), ("D12", 16.8, 1.60)])
def test_presets_match_their_nominal_numbers(name, impulse, burn):
    m = MOTORS[name]
    assert m.total_impulse == pytest.approx(impulse)
    assert m.burn_time == pytest.approx(burn)
    assert m.average_thrust == pytest.approx(impulse / burn)


def test_thrust_is_zero_outside_the_burn():
    m = MOTORS["C6"]
    assert m.thrust(-1.0) == 0.0
    assert m.thrust(m.burn_time) == 0.0
    assert m.thrust(m.burn_time + 5.0) == 0.0
    assert m.thrust(0.3) > 0.0


def test_impulse_matches_numerical_integral_of_thrust():
    m = MOTORS["C6"]
    n = 20_000
    h = m.burn_time / n
    total = sum(0.5 * (m.thrust(i * h) + m.thrust((i + 1) * h)) * h for i in range(n))
    assert total == pytest.approx(m.total_impulse, rel=1e-4)
    assert m.impulse_at(0.9) == pytest.approx(
        sum(0.5 * (m.thrust(i * h) + m.thrust(min((i + 1) * h, 0.9))) * min(h, max(0.9 - i * h, 0))
            for i in range(n)), rel=1e-3)


def test_mass_falls_monotonically_to_the_burnout_mass():
    m = MOTORS["C6"]
    samples = [m.mass(i * 0.05) for i in range(60)]
    assert samples[0] == pytest.approx(m.loaded_mass)
    assert all(a >= b for a, b in zip(samples, samples[1:]))
    assert m.mass(10.0) == pytest.approx(m.loaded_mass - m.propellant_mass)


def test_scaled_changes_thrust_but_not_mass_flow_shape():
    m = MOTORS["C6"]
    s = m.scaled(1.1)
    assert s.total_impulse == pytest.approx(1.1 * m.total_impulse)
    assert s.mass(0.9) == pytest.approx(m.mass(0.9))


def test_get_motor_parses_delay():
    m = get_motor("c6-7")
    assert m.name == "C6-7" and m.delay == 7.0
    assert get_motor("D12").delay == 5.0


def test_get_motor_rejects_unknown_specs():
    with pytest.raises(ValueError):
        get_motor("Z99")
    with pytest.raises(ValueError):
        get_motor("C6-abc")


def test_invalid_curves_are_rejected():
    with pytest.raises(ValueError):
        Motor("bad", (0.0, 1.0, 0.5), (0.0, 5.0, 0.0), 0.01, 0.02)
    with pytest.raises(ValueError):
        Motor("bad", (0.0, 1.0), (0.0, -5.0), 0.01, 0.02)
    with pytest.raises(ValueError):
        Motor("bad", (0.0, 1.0), (0.0, 5.0), 0.03, 0.02)
