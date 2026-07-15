"""One-command experiment validation, execution, analysis, and reporting."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
from collections.abc import Sequence

from jupedsim_mall.analysis import behavior_states, realism_evaluation, report_builder, run_metrics, statistics
from jupedsim_mall.experiments import matrix_runner, validate_scenarios
from jupedsim_mall.project import PROJECT_ROOT


def _write_status(path: pathlib.Path, steps: list[dict], passed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": "1.0",
        "generated_at": dt.datetime.now().astimezone().isoformat(),
        "passed": passed,
        "steps": steps,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def execute(args: argparse.Namespace) -> int:
    output = args.output_dir
    status_path = output / "pipeline_status.json"
    steps: list[dict] = []

    def step(name: str, function, arguments: list[str]) -> None:
        code = function(arguments)
        steps.append({"name": name, "returncode": code, "passed": code == 0})
        if code:
            _write_status(status_path, steps, False)
            raise RuntimeError(f"pipeline step failed: {name}")

    try:
        step("validate_scenarios", validate_scenarios.main, [])
        if args.run:
            command = "resume" if args.resume else "run"
            matrix_args = [command, str(args.plan)]
            if args.retry_failed:
                matrix_args.append("--retry-failed")
            step("matrix_execution", matrix_runner.main, matrix_args)
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        incomplete = [item["plan_item_id"] for item in plan["runs"] if item["status"] != "completed"]
        if incomplete and not args.allow_incomplete:
            raise RuntimeError(f"matrix has {len(incomplete)} incomplete run(s)")
        manifests = [
            pathlib.Path(item["attempts"][-1]["manifest"])
            for item in plan["runs"]
            if item["status"] == "completed" and item["attempts"] and item["attempts"][-1].get("manifest")
        ]
        if not manifests:
            raise RuntimeError("matrix has no completed run manifests")
        step("metrics", run_metrics.main, [str(path) for path in manifests])
        metric_paths = [path.with_name("metrics.json") for path in manifests]
        realism_dir = output / "realism"
        step("realism", realism_evaluation.main, [
            "--reference", str(args.reference), "--sim", *[str(path) for path in metric_paths],
            "--output-json", str(realism_dir / "realism.json"),
            "--output-csv", str(realism_dir / "realism.csv"),
        ])
        statistics_dir = output / "statistics"
        step("statistics", statistics.main, [
            *[str(path) for path in metric_paths], "--plan", str(args.plan),
            "--output-dir", str(statistics_dir),
        ])
        states_dir = output / "states"
        step("states", behavior_states.main, [
            *[str(path) for path in manifests], "--output-dir", str(states_dir),
        ])
        status = args.plan.with_name("status.json")
        matrix_runner.write_status_files(args.plan, plan)
        report_args = [
            *[str(path) for path in metric_paths], "--statistics-dir", str(statistics_dir),
            "--realism", str(realism_dir / "realism.json"), "--states", str(states_dir),
            "--matrix-status", str(status), "--output-dir", str(output / "report"),
        ]
        step("report", report_builder.main, report_args)
    except (OSError, ValueError, RuntimeError, SystemExit) as exc:
        steps.append({"name": "pipeline", "passed": False, "error": str(exc)})
        _write_status(status_path, steps, False)
        print(f"Pipeline failed: {exc}")
        return 1
    _write_status(status_path, steps, True)
    print("Pipeline report: " + str(output / "report" / "experiment_report.md"))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=pathlib.Path)
    parser.add_argument("--reference", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, default=PROJECT_ROOT / "outputs" / "pipeline")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    return execute(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
