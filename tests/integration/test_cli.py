import json

from jupedsim_mall.cli import main


def test_doctor_json_reports_required_environment(capsys):
    assert main(["doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    checks = {item["name"]: item for item in payload["checks"]}
    assert checks["project_root"]["status"] == "ok"
    assert checks["data/map/geometry.wkt"]["status"] == "ok"


def test_scenario_list_uses_package_entrypoint(capsys):
    assert main(["scenarios", "list"]) == 0
    output = capsys.readouterr().out
    assert "smoke_baseline" in output
    assert "baseline_random" in output


def test_run_dry_run_builds_legacy_compatible_command(tmp_path, capsys):
    assert main([
        "run",
        "--dry-run",
        "--manifest-dir",
        str(tmp_path),
        "smoke_baseline",
    ]) == 0
    output = capsys.readouterr().out
    assert "src\\demo_map_simulation.py" in output or "src/demo_map_simulation.py" in output
    assert list(tmp_path.glob("experiment_manifest_*.json"))
    run_manifests = list(tmp_path.glob("*/manifest.json"))
    assert len(run_manifests) == 1
    run_manifest = json.loads(run_manifests[0].read_text(encoding="utf-8"))
    assert run_manifest["run_id"].startswith("smoke_baseline_seed2026_")
    assert (run_manifests[0].parent / "resolved_scenario.json").is_file()
    assert "--event-output" in run_manifest["command"]


def test_profile_prepare_reports_fallback_for_missing_source(tmp_path, capsys):
    assert main([
        "profiles",
        "prepare",
        "--source",
        "llmob",
        "--data-path",
        str(tmp_path / "missing"),
        "--report-output",
        str(tmp_path / "report.json"),
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["requested_source"] == "llmob"
    assert payload["active_source"] == "mall"
    assert (tmp_path / "report.json").is_file()
