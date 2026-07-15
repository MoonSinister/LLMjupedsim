"""Persistent, fingerprinted experiment-matrix orchestration."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import pathlib
import sys
import threading
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO

from jsonschema import Draft202012Validator

from jupedsim_mall.experiments import run_experiment_suite
from jupedsim_mall.experiments.verify_runs import verify_manifest
from jupedsim_mall.project import MATRIX_DIR, OUTPUT_DIR, PROJECT_ROOT, SCENARIO_DIR, SCHEMA_DIR

PLAN_VERSION = "1.0"
TERMINAL_STATUSES = {"completed", "failed", "cancelled", "skipped"}


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_text(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_json(path: pathlib.Path, payload: dict) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def validate_matrix(config_path: pathlib.Path) -> dict:
    config = _load_json(config_path)
    schema = _load_json(SCHEMA_DIR / "matrix.schema.json")
    errors = sorted(Draft202012Validator(schema).iter_errors(config), key=lambda error: list(error.path))
    if errors:
        details = "; ".join(f"{'/'.join(map(str, error.path)) or '<root>'}: {error.message}" for error in errors)
        raise ValueError(f"invalid matrix config: {details}")
    cell_ids = [cell["id"] for cell in config["cells"]]
    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError("matrix cell ids must be unique")
    for cell in config["cells"]:
        scenario_path = SCENARIO_DIR / f"{cell['scenario']}.json"
        if not scenario_path.is_file():
            raise ValueError(f"matrix cell {cell['id']} references missing scenario {cell['scenario']}")
    return config


def _scenario_snapshot(cell: dict) -> tuple[pathlib.Path, dict]:
    path = SCENARIO_DIR / f"{cell['scenario']}.json"
    scenario = run_experiment_suite.load_scenario(path)
    scenario["args"] = [*scenario.get("args", []), *[str(item) for item in cell.get("args", [])]]
    scenario["env"] = {**scenario.get("env", {}), **{key: str(value) for key, value in cell.get("env", {}).items()}}
    return path, scenario


def expand_matrix(config: dict, config_path: pathlib.Path) -> dict:
    matrix_snapshot = {
        "schema_version": config["schema_version"],
        "name": config["name"],
        "description": config["description"],
        "seeds": config["seeds"],
        "cells": config["cells"],
        "resources": config["resources"],
        "estimates": config["estimates"],
    }
    matrix_fingerprint = _canonical_hash(matrix_snapshot)
    runs = []
    enabled_cells = [cell for cell in config["cells"] if cell.get("enabled", True)]
    for cell in enabled_cells:
        scenario_path, scenario = _scenario_snapshot(cell)
        scenario_hash = _canonical_hash(scenario)
        for seed_index, seed in enumerate(config["seeds"], 1):
            identity = {
                "matrix_fingerprint": matrix_fingerprint,
                "cell_id": cell["id"],
                "scenario_hash": scenario_hash,
                "seed": seed,
            }
            fingerprint = _canonical_hash(identity)
            runs.append({
                "plan_item_id": f"{cell['id']}_seed{seed}",
                "fingerprint": fingerprint,
                "cell_id": cell["id"],
                "scenario": cell["scenario"],
                "scenario_path": str(scenario_path.relative_to(PROJECT_ROOT)),
                "scenario_hash": scenario_hash,
                "seed": seed,
                "seed_index": seed_index,
                "factors": cell["factors"],
                "override": {
                    "args": [str(item) for item in cell.get("args", [])],
                    "env": {key: str(value) for key, value in cell.get("env", {}).items()},
                },
                "requires_llm": bool(cell.get("requires_llm", "--llm-routing" in scenario.get("args", []))),
                "status": "pending",
                "attempts": [],
            })
    estimates = config["estimates"]
    llm_runs = sum(run["requires_llm"] for run in runs)
    resources = config["resources"]
    memory_workers = max(1, resources["max_memory_mb"] // resources["memory_mb_per_run"])
    effective_workers = max(1, min(resources["max_workers"], memory_workers, os.cpu_count() or 1))
    cpu_wall_seconds = len(runs) * estimates["seconds_per_run"] / effective_workers
    llm_wall_seconds = llm_runs * estimates["seconds_per_run"] / resources["max_llm_workers"]
    return {
        "schema_version": PLAN_VERSION,
        "matrix_name": config["name"],
        "matrix_fingerprint": matrix_fingerprint,
        "config_path": str(config_path.resolve()),
        "prepared_at": _now(),
        "frozen_seed_list": list(config["seeds"]),
        "resources": resources,
        "effective_workers": effective_workers,
        "estimates": {
            **estimates,
            "total_runs": len(runs),
            "llm_runs": llm_runs,
            "estimated_wall_seconds": round(max(cpu_wall_seconds, llm_wall_seconds), 2),
            "estimated_storage_mb": round(len(runs) * estimates["storage_mb_per_run"], 2),
            "estimated_llm_calls": round(llm_runs * estimates.get("llm_calls_per_run", 0), 2),
        },
        "status": "prepared",
        "runs": runs,
    }


def default_plan_path(plan: dict) -> pathlib.Path:
    return OUTPUT_DIR / "matrices" / plan["matrix_name"] / plan["matrix_fingerprint"][:12] / "plan.json"


def prepare(config_path: pathlib.Path, output: pathlib.Path | None = None) -> pathlib.Path:
    config = validate_matrix(config_path)
    plan = expand_matrix(config, config_path)
    path = output or default_plan_path(plan)
    if path.exists():
        existing = _load_json(path)
        if existing.get("matrix_fingerprint") != plan["matrix_fingerprint"]:
            raise ValueError(f"existing plan fingerprint differs: {path}")
        return path
    _atomic_write_json(path, plan)
    write_status_files(path, plan)
    return path


def _status_counts(plan: dict) -> Counter:
    return Counter(item["status"] for item in plan["runs"])


def write_status_files(plan_path: pathlib.Path, plan: dict) -> None:
    counts = _status_counts(plan)
    payload = {
        "schema_version": PLAN_VERSION,
        "matrix_name": plan["matrix_name"],
        "matrix_fingerprint": plan["matrix_fingerprint"],
        "generated_at": _now(),
        "planned_sample_size": len(plan["runs"]),
        "counts": {status: counts.get(status, 0) for status in (
            "pending", "running", "completed", "failed", "interrupted", "cancelled", "skipped"
        )},
        "sample_size_satisfied": counts.get("completed", 0) == len(plan["runs"]),
        "runs": [{
            "plan_item_id": item["plan_item_id"],
            "cell_id": item["cell_id"],
            "scenario": item["scenario"],
            "seed": item["seed"],
            "status": item["status"],
            "attempt_count": len(item["attempts"]),
            "latest_run_id": item["attempts"][-1].get("run_id") if item["attempts"] else None,
        } for item in plan["runs"]],
    }
    _atomic_write_json(plan_path.with_name("status.json"), payload)
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=[
        "plan_item_id", "cell_id", "scenario", "seed", "status", "attempt_count", "latest_run_id"
    ])
    writer.writeheader()
    writer.writerows(payload["runs"])
    _atomic_write_text(plan_path.with_name("status.csv"), stream.getvalue())


def _print_plan_summary(plan: dict) -> None:
    estimates = plan["estimates"]
    print(f"Matrix: {plan['matrix_name']} ({plan['matrix_fingerprint'][:12]})")
    print(f"Runs: {estimates['total_runs']} | paired seeds: {plan['frozen_seed_list']}")
    print(
        f"Workers: {plan['effective_workers']} CPU/memory, "
        f"{plan['resources']['max_llm_workers']} LLM | "
        f"estimated wall: {estimates['estimated_wall_seconds'] / 3600:.2f} h | "
        f"storage: {estimates['estimated_storage_mb']:.0f} MB | "
        f"LLM calls: {estimates['estimated_llm_calls']:.0f}"
    )


def _suite_args(python: str | None, dry_run: bool) -> argparse.Namespace:
    return argparse.Namespace(
        python=python,
        manifest_dir=OUTPUT_DIR / "runs",
        repeat=1,
        dry_run=dry_run,
        legacy_output_layout=False,
        stop_on_failure=False,
    )


def _preview_run(item: dict, python: str | None) -> str:
    scenario_path = PROJECT_ROOT / item["scenario_path"]
    scenario = run_experiment_suite.load_scenario(scenario_path)
    scenario["args"] = [*scenario.get("args", []), *item["override"]["args"]]
    args = run_experiment_suite.scenario_args_for_run(scenario, item["seed"], item["seed_index"], 1)
    preview_id = f"{item['scenario']}_seed{item['seed']}_<timestamp>_{item['fingerprint'][:8]}"
    args, _ = run_experiment_suite.isolated_scenario_args(
        args,
        OUTPUT_DIR / "runs" / preview_id,
        preview_id,
    )
    executable = python or os.environ.get("JUPEDSIM_PYTHON") or sys.executable
    return " ".join(run_experiment_suite.build_command(scenario, executable, args))


def _completed_attempt_valid(item: dict) -> bool:
    if not item["attempts"]:
        return False
    attempt = item["attempts"][-1]
    manifest_value = attempt.get("manifest")
    if not manifest_value:
        return False
    manifest_path = pathlib.Path(manifest_value)
    if not manifest_path.is_file():
        return False
    report = verify_manifest(manifest_path)
    manifest = _load_json(manifest_path)
    return (
        report["valid"]
        and manifest.get("matrix", {}).get("plan_item_fingerprint") == item["fingerprint"]
    )


def reconcile_for_resume(plan: dict, retry_failed: bool) -> None:
    for item in plan["runs"]:
        if item["status"] == "completed" and not _completed_attempt_valid(item):
            item["status"] = "interrupted"
        elif item["status"] == "running":
            item["status"] = "interrupted"
        elif item["status"] == "cancelled":
            item["status"] = "interrupted"
        elif item["status"] == "failed" and retry_failed:
            item["status"] = "pending"


def execute_plan(
    plan_path: pathlib.Path,
    *,
    python: str | None,
    dry_run: bool,
    retry_failed: bool,
    resume: bool,
) -> int:
    plan = _load_json(plan_path)
    if resume:
        reconcile_for_resume(plan, retry_failed=True)
    elif retry_failed:
        reconcile_for_resume(plan, retry_failed=True)
    _print_plan_summary(plan)
    targets = [item for item in plan["runs"] if item["status"] in {"pending", "interrupted"}]
    if dry_run:
        for item in targets:
            print(f"{item['plan_item_id']:24} {_preview_run(item, python)}")
        print(f"Dry run: {len(targets)} command(s), no simulation started.")
        return 0
    cancel_path = plan_path.with_name("cancel.requested")
    if resume and cancel_path.exists():
        cancel_path.unlink()
    if cancel_path.exists():
        print(f"Cancellation is requested: {cancel_path}")
        return 2

    lock = threading.Lock()
    llm_semaphore = threading.Semaphore(plan["resources"]["max_llm_workers"])
    suite_args = _suite_args(python, False)

    def save() -> None:
        plan["updated_at"] = _now()
        counts = _status_counts(plan)
        if counts.get("completed", 0) == len(plan["runs"]):
            plan["status"] = "completed"
        elif counts.get("running", 0) or counts.get("pending", 0) or counts.get("interrupted", 0):
            plan["status"] = "running"
        elif counts.get("failed", 0):
            plan["status"] = "failed"
        else:
            plan["status"] = "incomplete"
        _atomic_write_json(plan_path, plan)
        write_status_files(plan_path, plan)

    def worker(item: dict) -> None:
        if cancel_path.exists():
            with lock:
                item["status"] = "cancelled"
                save()
            return
        with lock:
            item["status"] = "running"
            item["started_at"] = _now()
            save()
        semaphore = llm_semaphore if item["requires_llm"] else threading.Semaphore(1)
        try:
            with semaphore:
                record = run_experiment_suite.run_scenario(
                    PROJECT_ROOT / item["scenario_path"],
                    suite_args,
                    item["seed_index"],
                    item["seed"],
                    scenario_override=item["override"],
                    matrix_context={
                        "matrix_name": plan["matrix_name"],
                        "matrix_fingerprint": plan["matrix_fingerprint"],
                        "plan_item_id": item["plan_item_id"],
                        "plan_item_fingerprint": item["fingerprint"],
                        "cell_id": item["cell_id"],
                        "factors": item["factors"],
                    },
                )
            attempt = {
                "attempt": len(item["attempts"]) + 1,
                "run_id": record["run_id"],
                "status": record["status"],
                "returncode": record.get("returncode"),
                "started_at": record.get("started_at"),
                "finished_at": record.get("finished_at"),
                "manifest": record.get("artifacts", {}).get("manifest"),
            }
            status = "completed" if record["status"] == "completed" else "failed"
        except Exception as exc:
            attempt = {
                "attempt": len(item["attempts"]) + 1,
                "run_id": None,
                "status": "failed",
                "finished_at": _now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
            status = "failed"
        with lock:
            item["attempts"].append(attempt)
            item["status"] = status
            item["finished_at"] = _now()
            save()

    with lock:
        plan["status"] = "running"
        plan["started_at"] = plan.get("started_at") or _now()
        save()
    with ThreadPoolExecutor(max_workers=plan["effective_workers"], thread_name_prefix="matrix-run") as executor:
        futures = [executor.submit(worker, item) for item in targets]
        for future in as_completed(futures):
            future.result()
    with lock:
        counts = _status_counts(plan)
        plan["finished_at"] = _now()
        plan["status"] = "completed" if counts.get("completed", 0) == len(plan["runs"]) else "incomplete"
        save()
    print_status(plan_path, json_output=False)
    return 0 if plan["status"] == "completed" else 1


def print_status(plan_path: pathlib.Path, json_output: bool) -> int:
    plan = _load_json(plan_path)
    counts = _status_counts(plan)
    payload = {
        "matrix_name": plan["matrix_name"],
        "matrix_fingerprint": plan["matrix_fingerprint"],
        "status": plan["status"],
        "planned": len(plan["runs"]),
        "counts": dict(sorted(counts.items())),
        "sample_size_satisfied": counts.get("completed", 0) == len(plan["runs"]),
    }
    if json_output:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"Matrix {payload['matrix_name']} [{payload['status']}] fingerprint={payload['matrix_fingerprint'][:12]}")
        print(" ".join(f"{name}={value}" for name, value in payload["counts"].items()))
        print(f"sample_size_satisfied={payload['sample_size_satisfied']}")
    return 0


def request_cancel(plan_path: pathlib.Path) -> int:
    plan = _load_json(plan_path)
    cancel_path = plan_path.with_name("cancel.requested")
    _atomic_write_text(cancel_path, _now())
    plan["cancellation_requested_at"] = _now()
    _atomic_write_json(plan_path, plan)
    print(f"Cooperative cancellation requested: {cancel_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jupedsim-mall matrix")
    commands = parser.add_subparsers(dest="matrix_command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("config", type=pathlib.Path)
    prepare_parser.add_argument("--output", type=pathlib.Path)
    for name in ("run", "resume"):
        command = commands.add_parser(name)
        command.add_argument("plan", type=pathlib.Path)
        command.add_argument("--python")
        command.add_argument("--dry-run", action="store_true")
        command.add_argument("--retry-failed", action="store_true")
    status_parser = commands.add_parser("status")
    status_parser.add_argument("plan", type=pathlib.Path)
    status_parser.add_argument("--json", action="store_true")
    cancel_parser = commands.add_parser("cancel-local")
    cancel_parser.add_argument("plan", type=pathlib.Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.matrix_command == "prepare":
        path = prepare(args.config, args.output)
        plan = _load_json(path)
        _print_plan_summary(plan)
        print(f"Plan saved: {path}")
        return 0
    if args.matrix_command == "status":
        return print_status(args.plan, args.json)
    if args.matrix_command == "cancel-local":
        return request_cancel(args.plan)
    return execute_plan(
        args.plan,
        python=args.python,
        dry_run=args.dry_run,
        retry_failed=args.retry_failed,
        resume=args.matrix_command == "resume",
    )


if __name__ == "__main__":
    raise SystemExit(main())
