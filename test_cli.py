from rocketsim.cli import main


def test_fly_prints_a_report_and_writes_the_figure(tmp_path, capsys):
    out = tmp_path / "flight.png"
    csv_path = tmp_path / "flight.csv"
    assert main(["fly", "--out", str(out), "--csv", str(csv_path)]) == 0
    text = capsys.readouterr().out
    assert "Apogee" in text and "Landing" in text
    assert out.stat().st_size > 10_000
    lines = csv_path.read_text().splitlines()
    assert lines[0] == "t,x,y,z,vx,vy,vz,speed,g_load"
    assert len(lines) > 100


def test_mc_runs_with_a_seed(tmp_path, capsys):
    out = tmp_path / "landing.png"
    assert main(["mc", "--runs", "20", "--seed", "1", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "95% within" in text
    assert out.stat().st_size > 10_000


def test_motors_lists_the_presets(capsys):
    assert main(["motors"]) == 0
    text = capsys.readouterr().out
    assert "B6" in text and "C6" in text and "D12" in text


def test_bad_input_returns_an_error_code(capsys):
    assert main(["fly", "--motor", "Z99", "--no-plot"]) == 2
    assert "unknown motor" in capsys.readouterr().err


def test_no_chute_run_does_not_crash(capsys):
    assert main(["fly", "--chute", "0", "--no-plot"]) == 0
    assert "no chute" in capsys.readouterr().out
