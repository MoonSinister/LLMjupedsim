import json
import os
from pathlib import Path
import subprocess
import sys

from jsonschema import Draft202012Validator

from jupedsim_mall.analysis.experiment_summary import summarize_plan, summarize_sqlite
from jupedsim_mall.project import PROJECT_ROOT, SCHEMA_DIR


GOLDEN_PATH = PROJECT_ROOT / "tests" / "golden" / "smoke_baseline_metrics.json"


def test_fixed_seed_smoke_matches_golden_baseline(tmp_path):
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    trajectory = tmp_path / "smoke.sqlite"
    plan = tmp_path / "smoke_plan.json"
    settings = golden["simulation"]
    command = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "demo_map_simulation.py"),
        "-n",
        str(settings["num_agents"]),
        "--seed",
        str(golden["seed"]),
        "--max-iters",
        str(settings["max_iters"]),
        "--movement-model",
        settings["movement_model"],
        "--output",
        str(trajectory),
        "--llm-plan-output",
        str(plan),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    env["PYTHONUTF8"] = "1"
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout

    plan_payload = json.loads(plan.read_text(encoding="utf-8"))
    plan_schema = json.loads((SCHEMA_DIR / "plan.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(plan_schema).validate(plan_payload)

    plan_metrics = summarize_plan(plan)
    trajectory_metrics = summarize_sqlite(trajectory)
    expected = golden["expected"]
    tolerances = golden["tolerances"]

    assert plan_metrics["agents"] == expected["plan_agents"]
    assert plan_metrics["validation_valid"] is expected["plan_validation_valid"]
    assert plan_metrics["roles"] == expected["roles"]
    assert plan_metrics["exits"] == expected["exits"]
    assert plan_metrics["rerouted_due_to_stuck"] == expected["rerouted_due_to_stuck"]
    assert trajectory_metrics["trajectory_agents_observed"] == expected["trajectory_agents_observed"]

    for field in (
        "trajectory_frames",
        "trajectory_duration_seconds",
        "avg_observed_travel_seconds",
        "avg_path_length_m",
        "avg_observed_speed_mps",
    ):
        assert abs(trajectory_metrics[field] - expected[field]) <= tolerances[field]
