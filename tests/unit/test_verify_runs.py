import json
import sqlite3

from jupedsim_mall.experiments.verify_runs import verify_manifest


def test_verify_completed_isolated_run(tmp_path):
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    trajectory = run_dir / "trajectory.sqlite"
    with sqlite3.connect(trajectory) as connection:
        connection.execute("CREATE TABLE trajectory_data (id INTEGER)")
    (run_dir / "plans.json").write_text("{}", encoding="utf-8")
    (run_dir / "resolved_scenario.json").write_text("{}", encoding="utf-8")
    (run_dir / "events.jsonl").write_text('{"event_type":"run_finished"}\n', encoding="utf-8")
    (run_dir / "run.log").write_text("complete", encoding="utf-8")
    artifacts = {name: str(run_dir / filename) for name, filename in {
        "trajectory": "trajectory.sqlite", "plans": "plans.json", "events": "events.jsonl",
        "log": "run.log", "resolved_scenario": "resolved_scenario.json",
    }.items()}
    manifest = {
        "schema_version": "1.0", "run_id": "run-1", "scenario_name": "smoke",
        "seed": 2026, "status": "completed", "returncode": 0, "artifacts": artifacts,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_manifest(manifest_path)["valid"] is True


def test_verify_rejects_interrupted_or_missing_completed_artifacts(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": "1.0",
        "run_id": "broken",
        "scenario_name": "smoke",
        "seed": 1,
        "status": "completed",
        "returncode": 0,
        "artifacts": {},
    }), encoding="utf-8")
    report = verify_manifest(manifest_path)
    assert report["valid"] is False
    assert any("missing artifact declaration" in error for error in report["errors"])
