"""Build and verify a portable thesis reproduction package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import platform
import shutil
import sys
from collections.abc import Sequence

from jupedsim_mall.project import PROJECT_ROOT


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(output: pathlib.Path, plan_path: pathlib.Path, report_dir: pathlib.Path) -> pathlib.Path:
    output.mkdir(parents=True, exist_ok=True)
    for directory in ("environment", "configs", "summaries", "figures", "tables", "manifests"):
        (output / directory).mkdir(exist_ok=True)
    shutil.copytree(PROJECT_ROOT / "configs", output / "configs", dirs_exist_ok=True)
    for name in ("pyproject.toml", "requirements.txt", "requirements-lock.txt"):
        shutil.copy2(PROJECT_ROOT / name, output / "environment" / name)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    (output / "environment" / "runtime.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
    if report_dir.is_dir():
        for name in ("figures", "tables"):
            if (report_dir / name).is_dir():
                shutil.copytree(report_dir / name, output / name, dirs_exist_ok=True)
        for name in ("experiment_report.md", "report_provenance.json"):
            if (report_dir / name).is_file():
                shutil.copy2(report_dir / name, output / name)
        provenance_path = report_dir / "report_provenance.json"
        if provenance_path.is_file():
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            statistics_dir = pathlib.Path(provenance.get("statistics_dir", ""))
            if statistics_dir.is_dir():
                for source in statistics_dir.iterdir():
                    if source.is_file():
                        shutil.copy2(source, output / "summaries" / source.name)
            realism = provenance.get("realism")
            if realism and pathlib.Path(realism).is_file():
                shutil.copy2(pathlib.Path(realism), output / "summaries" / pathlib.Path(realism).name)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    rows = []
    for item in plan["runs"]:
        attempt = item["attempts"][-1] if item["attempts"] else {}
        manifest = pathlib.Path(attempt["manifest"]) if attempt.get("manifest") else None
        archived = ""
        if manifest and manifest.is_file():
            destination = output / "manifests" / (item["plan_item_id"] + ".json")
            shutil.copy2(manifest, destination)
            archived = str(destination.relative_to(output))
        rows.append({
            "plan_item_id": item["plan_item_id"],
            "cell_id": item["cell_id"],
            "seed": item["seed"],
            "status": item["status"],
            "run_id": attempt.get("run_id"),
            "manifest": archived,
        })
    with (output / "run_manifest_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["plan_item_id"])
        writer.writeheader()
        writer.writerows(rows)
    allowed_data = [
        PROJECT_ROOT / "data" / "map" / "geometry.wkt",
        PROJECT_ROOT / "data" / "map" / "stages.json",
        PROJECT_ROOT / "data" / "map" / "localization_grid_regions.json",
    ]
    data_manifest = {
        "schema_version": "1.0",
        "included": [{"path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256(path)} for path in allowed_data if path.is_file()],
        "restricted": [{
            "name": "ATC/LLMob raw mobility data",
            "included": False,
            "instructions": "Set ATC_RAW_PATH and LLMOB_DATA_ROOT to licensed local copies; verify hashes recorded in run manifests.",
        }],
    }
    (output / "data_manifest.json").write_text(json.dumps(data_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    provenance_path = output / "report_provenance.json"
    figure_map = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.is_file() else {}
    (output / "paper_artifact_map.json").write_text(json.dumps({
        "figures": figure_map.get("figures", []),
        "tables": figure_map.get("tables", []),
        "mapping_rule": "Assign paper figure/table numbers only to these generated artifact identifiers.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    guide = """# Reproduction Guide

1. Create Python 3.12 environment and install environment/requirements.txt.
2. Set PYTHONPATH=src in the source checkout.
3. Run python -m jupedsim_mall quality-gate.
4. Prepare the frozen matrix under configs/matrices.
5. Execute or resume the matrix; never delete failed attempts.
6. Run analyze metrics, analyze realism, analyze statistics, analyze states.
7. Run report build and compare hashes in this release.

Restricted ATC/LLMob raw data are intentionally excluded. The included smoke matrix requires no remote LLM or restricted data.
"""
    (output / "reproduction_guide.md").write_text(guide, encoding="utf-8")
    limitations = """# Limitations

- ATC and simulated tracks are compared distributionally, not pedestrian by pedestrian.
- Remote LLM behavior can change with model/service versions despite fixed prompts.
- Conclusions are bounded by the map, spawn process, profile construction, and tested parameter ranges.
- LLMob and some ATC source data may be license-restricted and are not redistributed.
"""
    (output / "limitations.md").write_text(limitations, encoding="utf-8")
    files = [path for path in output.rglob("*") if path.is_file() and path.name != "release_manifest.json"]
    release_manifest = {"schema_version": "1.0", "files": {str(path.relative_to(output)): sha256(path) for path in files}}
    manifest_path = output / "release_manifest.json"
    manifest_path.write_text(json.dumps(release_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def verify(path: pathlib.Path) -> bool:
    manifest_path = path / "release_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"data_manifest.json", "run_manifest_index.csv", "reproduction_guide.md", "limitations.md"}
    if not required.issubset(manifest["files"]):
        return False
    return all((path / relative).is_file() and sha256(path / relative) == digest for relative, digest in manifest["files"].items())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("build")
    create.add_argument("--output", type=pathlib.Path, required=True)
    create.add_argument("--plan", type=pathlib.Path, required=True)
    create.add_argument("--report-dir", type=pathlib.Path, required=True)
    check = commands.add_parser("verify")
    check.add_argument("path", type=pathlib.Path)
    args = parser.parse_args(argv)
    if args.command == "build":
        print("Release manifest: " + str(build(args.output, args.plan, args.report_dir)))
        return 0
    passed = verify(args.path)
    print("Release verification: " + ("PASSED" if passed else "FAILED"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
