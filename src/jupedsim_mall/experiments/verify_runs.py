"""Verify isolated experiment run directories before analysis."""

from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3
from collections.abc import Sequence

from jsonschema import Draft202012Validator

from jupedsim_mall.project import OUTPUT_DIR, PROJECT_ROOT

REQUIRED_COMPLETED_ARTIFACTS = ("trajectory", "plans", "events", "log", "resolved_scenario")


def _resolve_artifact(value: str, manifest_path: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(value)
    if path.is_absolute():
        return path
    project_path = PROJECT_ROOT / path
    return project_path if project_path.exists() else manifest_path.parent / path


def _validate_json(path: pathlib.Path) -> str | None:
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return f"invalid JSON: {exc}"
    return None


def _validate_jsonl(path: pathlib.Path) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return "JSONL file is empty"
        for line_number, line in enumerate(lines, 1):
            json.loads(line)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return f"invalid JSONL near line {locals().get('line_number', 0)}: {exc}"
    return None


def _validate_sqlite(path: pathlib.Path) -> str | None:
    try:
        with path.open("rb") as handle:
            if handle.read(16) != b"SQLite format 3\x00":
                return "invalid SQLite header"
        with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as connection:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not tables:
            return "SQLite database has no tables"
    except (OSError, sqlite3.Error) as exc:
        return f"invalid SQLite database: {exc}"
    return None


def verify_manifest(manifest_path: pathlib.Path) -> dict:
    errors, warnings = [], []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {"manifest": str(manifest_path), "valid": False, "errors": [f"manifest unreadable: {exc}"], "warnings": []}

    schema_path = PROJECT_ROOT / "configs" / "schemas" / "manifest.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors.extend(error.message for error in Draft202012Validator(schema).iter_errors(manifest))
    artifacts = manifest.get("artifacts", {})
    status = manifest.get("status")
    required = REQUIRED_COMPLETED_ARTIFACTS if status == "completed" else ()
    for name in required:
        value = artifacts.get(name)
        if not value:
            errors.append(f"missing artifact declaration: {name}")
            continue
        path = _resolve_artifact(value, manifest_path)
        if not path.is_file():
            errors.append(f"missing artifact file: {name} ({path})")
            continue
        issue = None
        if name in {"plans", "resolved_scenario"}:
            issue = _validate_json(path)
        elif name == "events":
            issue = _validate_jsonl(path)
        elif name == "trajectory":
            issue = _validate_sqlite(path)
        if issue:
            errors.append(f"{name}: {issue}")
    if status == "completed" and manifest.get("returncode") != 0:
        errors.append("completed run must have returncode 0")
    if status in {"running", "planned"}:
        warnings.append(f"run is not terminal: {status}")
    return {
        "manifest": str(manifest_path),
        "run_id": manifest.get("run_id"),
        "status": status,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def discover_manifests(paths: Sequence[str]) -> list[pathlib.Path]:
    roots = [pathlib.Path(item) for item in paths] if paths else [OUTPUT_DIR / "runs"]
    manifests = []
    for root in roots:
        if root.is_file():
            manifests.append(root)
        elif (root / "manifest.json").is_file():
            manifests.append(root / "manifest.json")
        elif root.is_dir():
            manifests.extend(root.glob("*/manifest.json"))
    return sorted(set(path.resolve() for path in manifests))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", help="Run directories or per-run manifest files.")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable report.")
    parser.add_argument("--latest", action="store_true", help="Verify only the newest discovered run.")
    args = parser.parse_args(argv)
    manifests = discover_manifests(args.paths)
    if args.latest and manifests:
        manifests = [max(manifests, key=lambda path: path.stat().st_mtime_ns)]
    reports = [verify_manifest(path) for path in manifests]
    payload = {"schema_version": "1.0", "valid": bool(reports) and all(item["valid"] for item in reports), "runs": reports}
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        for report in reports:
            print(f"{'OK' if report['valid'] else 'FAIL':4} {report.get('run_id') or report['manifest']} [{report.get('status')}]")
            for error in report["errors"]:
                print(f"     error: {error}")
            for warning in report["warnings"]:
                print(f"     warning: {warning}")
        print(f"Verified {len(reports)} run(s): {'valid' if payload['valid'] else 'invalid'}")
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
