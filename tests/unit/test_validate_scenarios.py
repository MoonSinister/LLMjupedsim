from pathlib import Path

from jupedsim_mall.experiments.validate_scenarios import validate_all, validate_scenario


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_valid_scenario_fixture_passes():
    assert validate_scenario(FIXTURES / "scenario_valid.json") == []


def test_invalid_scenario_reports_structure_and_semantics():
    errors = validate_scenario(FIXTURES / "scenario_invalid.json")

    assert any("does not match filename" in error for error in errors)
    assert any("must be >= 1" in error for error in errors)
    assert any("movement-model" in error for error in errors)
    assert any("LOCAL_LLM_BASE_URL" in error for error in errors)
    assert any(".sqlite" in error for error in errors)


def test_repository_scenarios_pass(capsys):
    scenario_dir = Path(__file__).resolve().parents[2] / "configs" / "scenarios"
    assert validate_all(sorted(scenario_dir.glob("*.json"))) == 0
    assert "Validated 7 scenario file(s)" in capsys.readouterr().out

