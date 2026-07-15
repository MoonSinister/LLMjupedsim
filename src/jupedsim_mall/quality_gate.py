"""Pre-experiment quality gate; no command contacts a remote LLM."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import subprocess
import sys
from collections.abc import Sequence

from jupedsim_mall.project import OUTPUT_DIR, PROJECT_ROOT


def run_step(name: str, command: list[str], env: dict[str, str]) -> dict:
    print(f"\n== quality gate: {name} ==")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True, check=False)
    return {"name": name, "command": command, "returncode": completed.returncode, "passed": completed.returncode == 0}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Skip the real local smoke simulation.")
    parser.add_argument("--output", type=pathlib.Path, default=OUTPUT_DIR / "quality" / "quality_gate.json")
    args = parser.parse_args(argv)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    env["LOCAL_LLM_BASE_URL"] = "http://127.0.0.1:1/v1"
    python = sys.executable
    steps = [
        ("compile", [python, "-m", "compileall", "-q", "src", "tests"]),
        ("doctor", [python, "-m", "jupedsim_mall", "doctor"]),
        ("scenarios", [python, "-m", "jupedsim_mall", "scenarios", "validate"]),
        ("matrix-prepare", [
            python, "-m", "jupedsim_mall", "matrix", "prepare",
            "configs/matrices/smoke_matrix.json",
            "--output", str(OUTPUT_DIR / "quality" / "smoke_matrix_plan.json"),
        ]),
        ("ruff", [python, "-m", "ruff", "check", "src/jupedsim_mall", "tests"]),
        ("mypy", [
            python, "-m", "mypy",
            "src/jupedsim_mall/analysis/run_metrics.py",
            "src/jupedsim_mall/analysis/statistics.py",
            "src/jupedsim_mall/analysis/behavior_states.py",
            "src/jupedsim_mall/experiments/matrix_runner.py",
            "src/jupedsim_mall/experiments/experiment_control.py",
            "src/jupedsim_mall/release_builder.py",
            "src/jupedsim_mall/pipeline.py",
        ]),
        ("pytest", [python, "-m", "pytest", "-q"]),
    ]
    if not args.quick:
        steps.extend([
            ("smoke", [
                python, "-m", "jupedsim_mall", "run",
                "--python", python,
                "--manifest-dir", str(OUTPUT_DIR / "runs"),
                "smoke_baseline",
            ]),
            ("verify-smoke", [python, "-m", "jupedsim_mall", "runs", "verify", "--latest", str(OUTPUT_DIR / "runs")]),
        ])
    results = []
    for name, command in steps:
        result = run_step(name, command, env)
        results.append(result)
        if not result["passed"]:
            break
    payload = {
        "schema_version": "1.0",
        "generated_at": dt.datetime.now().astimezone().isoformat(),
        "mode": "quick" if args.quick else "full",
        "remote_llm_allowed": False,
        "passed": len(results) == len(steps) and all(result["passed"] for result in results),
        "steps": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nQuality gate: {'PASSED' if payload['passed'] else 'FAILED'}")
    print(f"Report: {args.output}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
