"""Formal experiment preflight, immutable snapshot, and result finalization."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import urllib.request
from collections.abc import Sequence

from jupedsim_mall.experiments.matrix_runner import validate_matrix
from jupedsim_mall.project import CONFIG_DIR, OUTPUT_DIR, PROJECT_ROOT, SCENARIO_DIR


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_context() -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    return {"commit": commit.stdout.strip() if commit.returncode == 0 else None, "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None}


def preflight(matrix_path: pathlib.Path, quality_path: pathlib.Path, check_llm: bool) -> dict:
    checks = []
    try:
        matrix = validate_matrix(matrix_path)
        checks.append({"name": "matrix_schema", "passed": True, "detail": matrix["name"]})
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        matrix = {}
        checks.append({"name": "matrix_schema", "passed": False, "detail": str(exc)})
    checks.append({"name": "matrix_frozen", "passed": matrix.get("frozen") is True, "detail": str(matrix.get("frozen"))})
    quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.is_file() else {}
    checks.append({
        "name": "full_quality_gate",
        "passed": quality.get("passed") is True and quality.get("mode") == "full",
        "detail": f"passed={quality.get('passed')} mode={quality.get('mode')}",
    })
    scenarios = {cell.get("scenario") for cell in matrix.get("cells", [])}
    if "atc_llm_remote" in scenarios:
        path = pathlib.Path(os.environ.get("ATC_RAW_PATH", "data/atc-20121114/atc-20121114.csv"))
        checks.append({"name": "atc_data", "passed": path.exists(), "detail": str(path)})
    if "llmob_llm_remote" in scenarios:
        value = os.environ.get("LLMOB_DATA_ROOT", "")
        checks.append({"name": "llmob_data", "passed": bool(value) and pathlib.Path(value).exists(), "detail": value or "not configured"})
    if any(cell.get("requires_llm") for cell in matrix.get("cells", [])):
        base_url = os.environ.get("LOCAL_LLM_BASE_URL", "").rstrip("/")
        passed, detail = bool(base_url), base_url or "not configured"
        if passed and check_llm:
            try:
                with urllib.request.urlopen(base_url + "/models", timeout=3) as response:
                    passed, detail = response.status == 200, f"HTTP {response.status}"
            except OSError as exc:
                passed, detail = False, str(exc)
        checks.append({"name": "llm_endpoint", "passed": passed, "detail": detail})
    context = git_context()
    checks.append({"name": "git_commit", "passed": bool(context["commit"]), "detail": str(context)})
    return {"schema_version": "1.0", "matrix": str(matrix_path), "quality_gate": str(quality_path), "checks": checks, "passed": all(check["passed"] for check in checks)}


def freeze(matrix_path: pathlib.Path, quality_path: pathlib.Path, output_dir: pathlib.Path, check_llm: bool) -> pathlib.Path:
    report = preflight(matrix_path, quality_path, check_llm)
    if not report["passed"]:
        failed = [check["name"] for check in report["checks"] if not check["passed"]]
        raise ValueError("formal preflight failed: " + ", ".join(failed))
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    destination = output_dir / (matrix_path.stem + "_" + stamp)
    frozen = destination / "frozen"
    shutil.copytree(CONFIG_DIR, frozen / "configs")
    shutil.copy2(PROJECT_ROOT / "docs" / "experiment_protocol.md", frozen / "experiment_protocol.md")
    shutil.copy2(PROJECT_ROOT / "requirements-lock.txt", frozen / "requirements-lock.txt")
    files = [path for path in frozen.rglob("*") if path.is_file()]
    snapshot = {
        "schema_version": "1.0",
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "preflight": report,
        "git": git_context(),
        "files": {str(path.relative_to(destination)): sha256(path) for path in files},
        "status": "frozen_for_execution",
    }
    snapshot_path = destination / "FORMAL_SNAPSHOT.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return snapshot_path


def finalize(snapshot_path: pathlib.Path, plan_path: pathlib.Path, report_path: pathlib.Path) -> pathlib.Path:
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for item in plan["runs"]:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    all_completed = counts.get("completed", 0) == len(plan["runs"])
    metrics_complete = all(
        item["attempts"] and item["attempts"][-1].get("manifest")
        and pathlib.Path(item["attempts"][-1]["manifest"]).with_name("metrics.json").is_file()
        for item in plan["runs"]
    )
    if not all_completed or not metrics_complete or not report_path.is_file():
        raise ValueError(f"cannot finalize: all_completed={all_completed}, metrics_complete={metrics_complete}, report={report_path.is_file()}")
    payload = {
        "schema_version": "1.0",
        "frozen_at": dt.datetime.now().astimezone().isoformat(),
        "snapshot_sha256": sha256(snapshot_path),
        "matrix_fingerprint": plan["matrix_fingerprint"],
        "run_counts": counts,
        "report": str(report_path),
        "report_sha256": sha256(report_path),
        "status": "formal_results_frozen",
        "exploratory_results_must_use_separate_directory": True,
        "source_snapshot_status": snapshot["status"],
    }
    output = snapshot_path.parent / "RESULTS_FROZEN.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("preflight")
    check.add_argument("matrix", type=pathlib.Path)
    check.add_argument("--quality", type=pathlib.Path, default=OUTPUT_DIR / "quality" / "quality_gate.json")
    check.add_argument("--check-llm", action="store_true")
    snapshot = commands.add_parser("freeze")
    snapshot.add_argument("matrix", type=pathlib.Path)
    snapshot.add_argument("--quality", type=pathlib.Path, default=OUTPUT_DIR / "quality" / "quality_gate.json")
    snapshot.add_argument("--output-dir", type=pathlib.Path, default=OUTPUT_DIR / "formal")
    snapshot.add_argument("--check-llm", action="store_true")
    final = commands.add_parser("finalize")
    final.add_argument("snapshot", type=pathlib.Path)
    final.add_argument("plan", type=pathlib.Path)
    final.add_argument("report", type=pathlib.Path)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = preflight(args.matrix, args.quality, args.check_llm)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["passed"] else 1
    if args.command == "freeze":
        print("Snapshot: " + str(freeze(args.matrix, args.quality, args.output_dir, args.check_llm)))
        return 0
    print("Frozen results: " + str(finalize(args.snapshot, args.plan, args.report)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
