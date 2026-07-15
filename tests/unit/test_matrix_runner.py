import json

from jupedsim_mall.experiments import matrix_runner


def matrix_config(seeds=None):
    return {
        "schema_version": "1.0",
        "name": "test_matrix",
        "description": "Deterministic test matrix.",
        "frozen": True,
        "seeds": seeds or [101, 202],
        "cells": [
            {
                "id": "B1",
                "scenario": "smoke_baseline",
                "factors": {"planner": "random"},
                "requires_llm": False,
            },
            {
                "id": "B2",
                "scenario": "smoke_baseline",
                "factors": {"planner": "nearest"},
                "args": ["--baseline-routing", "nearest"],
                "requires_llm": False,
            },
        ],
        "resources": {
            "max_workers": 4,
            "max_llm_workers": 1,
            "max_memory_mb": 1024,
            "memory_mb_per_run": 512,
        },
        "estimates": {
            "seconds_per_run": 10,
            "storage_mb_per_run": 2,
            "llm_calls_per_run": 0,
        },
    }


def test_matrix_expansion_is_deterministic_and_uses_paired_seeds(tmp_path):
    config_path = tmp_path / "matrix.json"
    config_path.write_text(json.dumps(matrix_config()), encoding="utf-8")
    config = matrix_runner.validate_matrix(config_path)
    first = matrix_runner.expand_matrix(config, config_path)
    second = matrix_runner.expand_matrix(config, config_path)
    assert first["matrix_fingerprint"] == second["matrix_fingerprint"]
    assert [item["fingerprint"] for item in first["runs"]] == [
        item["fingerprint"] for item in second["runs"]
    ]
    assert first["effective_workers"] == 2
    assert {item["seed"] for item in first["runs"] if item["cell_id"] == "B1"} == {101, 202}
    assert {item["seed"] for item in first["runs"] if item["cell_id"] == "B2"} == {101, 202}


def test_prepare_is_idempotent_and_writes_status_tables(tmp_path):
    config_path = tmp_path / "matrix.json"
    plan_path = tmp_path / "plan.json"
    config_path.write_text(json.dumps(matrix_config()), encoding="utf-8")
    assert matrix_runner.prepare(config_path, plan_path) == plan_path
    first = plan_path.read_text(encoding="utf-8")
    assert matrix_runner.prepare(config_path, plan_path) == plan_path
    assert plan_path.read_text(encoding="utf-8") == first
    assert (tmp_path / "status.json").is_file()
    assert (tmp_path / "status.csv").is_file()


def test_retry_preserves_failed_attempt(monkeypatch, tmp_path):
    config_path = tmp_path / "matrix.json"
    config = matrix_config([101])
    config["cells"] = config["cells"][:1]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    plan = matrix_runner.expand_matrix(matrix_runner.validate_matrix(config_path), config_path)
    plan_path = tmp_path / "plan.json"
    matrix_runner._atomic_write_json(plan_path, plan)
    outcomes = iter(["failed", "completed"])

    def fake_run(*_args, **_kwargs):
        status = next(outcomes)
        return {
            "run_id": f"run-{status}",
            "status": status,
            "returncode": 0 if status == "completed" else 1,
            "started_at": "start",
            "finished_at": "finish",
            "artifacts": {},
        }

    monkeypatch.setattr(matrix_runner.run_experiment_suite, "run_scenario", fake_run)
    assert matrix_runner.execute_plan(
        plan_path, python=None, dry_run=False, retry_failed=False, resume=False
    ) == 1
    assert matrix_runner.execute_plan(
        plan_path, python=None, dry_run=False, retry_failed=True, resume=False
    ) == 0
    updated = json.loads(plan_path.read_text(encoding="utf-8"))
    assert [attempt["status"] for attempt in updated["runs"][0]["attempts"]] == [
        "failed",
        "completed",
    ]


def test_cancel_local_creates_cooperative_marker(tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps({"matrix_name": "m", "matrix_fingerprint": "abc", "runs": []}),
        encoding="utf-8",
    )
    assert matrix_runner.request_cancel(plan_path) == 0
    assert (tmp_path / "cancel.requested").is_file()


def test_resume_skips_completed_run_with_matching_fingerprint(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"matrix": {"plan_item_fingerprint": "same"}}),
        encoding="utf-8",
    )
    item = {
        "fingerprint": "same",
        "status": "completed",
        "attempts": [{"manifest": str(manifest_path)}],
    }
    monkeypatch.setattr(matrix_runner, "verify_manifest", lambda _path: {"valid": True})
    matrix_runner.reconcile_for_resume({"runs": [item]}, retry_failed=True)
    assert item["status"] == "completed"
