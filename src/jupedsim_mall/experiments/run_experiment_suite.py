#!/usr/bin/env python3
"""Run configured JuPedSim experiment scenarios.

Scenario files live in configs/scenarios/*.json and contain the command-line
arguments passed through to demo_map_simulation.py. This runner makes the
project workflow reproducible without hiding the original simulation entry.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
from collections.abc import Sequence

from jupedsim_mall.project import OUTPUT_DIR, PROJECT_ROOT, SCENARIO_DIR

DEFAULT_MANIFEST_DIR = OUTPUT_DIR / "runs"
REPRODUCIBILITY_FILES = [
    PROJECT_ROOT / "data" / "map" / "geometry.wkt",
    PROJECT_ROOT / "data" / "map" / "stages.json",
    PROJECT_ROOT / "data" / "map" / "localization_grid_regions.json",
    PROJECT_ROOT / "src" / "llm_prompts.py",
]


def load_scenario(path: pathlib.Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("name", path.stem)
    data.setdefault("enabled", True)
    data.setdefault("args", [])
    data.setdefault("env", {})
    return data


def list_scenarios() -> list[pathlib.Path]:
    return sorted(SCENARIO_DIR.glob("*.json"))


def select_scenarios(names: list[str], include_disabled: bool) -> list[pathlib.Path]:
    paths = list_scenarios()
    if names:
        wanted = set(names)
        paths = [path for path in paths if path.stem in wanted]
        missing = wanted - {path.stem for path in paths}
        if missing:
            raise SystemExit(f"Unknown scenario(s): {', '.join(sorted(missing))}")
    if include_disabled:
        return paths
    return [path for path in paths if load_scenario(path).get("enabled", True)]


def file_sha256(path: pathlib.Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def git_dirty() -> bool | None:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return bool(completed.stdout.strip())


def package_versions() -> dict[str, str | None]:
    versions = {}
    for name in ("jupedsim", "shapely", "numpy", "PyYAML", "jsonschema"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def reproducibility_context(python_exe: str, scenario_args: list[str] | None = None) -> dict:
    scenario_args = scenario_args or []
    input_options = (
        "--geometry", "--regions", "--atc-raw-path", "--atc-regions",
        "--replay-plan", "--llmob-data-root",
    )
    input_hashes = {}
    for option in input_options:
        value = get_option(scenario_args, option)
        if not value:
            continue
        path = pathlib.Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        input_hashes[option] = {
            "path": value,
            "sha256": file_sha256(path) if path.is_file() else None,
            "kind": "file" if path.is_file() else "directory" if path.is_dir() else "missing",
        }
    return {
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "python_executable": python_exe,
        "runner_python": sys.version,
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "file_hashes": {
            str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
            for path in REPRODUCIBILITY_FILES
        },
        "scenario_input_hashes": input_hashes,
    }


def option_present(items: list[str], option: str) -> bool:
    return option in items


def set_option(items: list[str], option: str, value: str) -> list[str]:
    result = list(items)
    if option in result:
        index = result.index(option)
        if index + 1 >= len(result):
            result.append(value)
        else:
            result[index + 1] = value
    else:
        result.extend([option, value])
    return result


def get_option(items: list[str], option: str, default: str = "") -> str:
    if option not in items:
        return default
    index = items.index(option)
    return items[index + 1] if index + 1 < len(items) else default


def atomic_write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def make_run_id(scenario: dict, seed: int, repeat_index: int) -> str:
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S_%f")
    identity = json.dumps(
        {"scenario": scenario, "seed": seed, "repeat_index": repeat_index},
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    short_hash = hashlib.sha256(identity).hexdigest()[:8]
    safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in scenario["name"])
    return f"{safe_name}_seed{seed}_{stamp}_{short_hash}"


def isolated_scenario_args(items: list[str], run_dir: pathlib.Path, run_id: str) -> tuple[list[str], dict[str, str]]:
    artifacts = {
        "trajectory": str(run_dir / "trajectory.sqlite"),
        "plans": str(run_dir / "plans.json"),
        "events": str(run_dir / "events.jsonl"),
        "llm_calls": str(run_dir / "llm_calls.jsonl"),
        "log": str(run_dir / "run.log"),
        "resolved_scenario": str(run_dir / "resolved_scenario.json"),
        "manifest": str(run_dir / "manifest.json"),
    }
    result = set_option(items, "--output", artifacts["trajectory"])
    result = set_option(result, "--llm-plan-output", artifacts["plans"])
    result = set_option(result, "--llm-calls-output", artifacts["llm_calls"])
    result = set_option(result, "--event-output", artifacts["events"])
    result = set_option(result, "--run-id", run_id)
    for option in ("--llmob-profile-cache", "--atc-profile-cache"):
        if option in result:
            profile_path = run_dir / f"{option[2:].replace('-', '_')}.json"
            result = set_option(result, option, str(profile_path))
            artifacts[option[2:].replace("-", "_")] = str(profile_path)
    return result, artifacts


def suffix_output_path(value: str, suffix: str) -> str:
    path = pathlib.Path(value)
    return str(path.with_name(f"{path.stem}_{suffix}{path.suffix}"))


def rewrite_outputs_for_repeat(items: list[str], suffix: str) -> list[str]:
    result = list(items)
    output_options = {
        "--output",
        "--llm-plan-output",
        "--llmob-profile-cache",
        "--atc-profile-cache",
    }
    index = 0
    while index < len(result):
        if result[index] in output_options and index + 1 < len(result):
            result[index + 1] = suffix_output_path(result[index + 1], suffix)
            index += 2
            continue
        index += 1
    return result


def scenario_args_for_run(scenario: dict, seed: int, repeat_index: int, repeat_total: int) -> list[str]:
    items = [str(item) for item in scenario.get("args", [])]
    items = set_option(items, "--seed", str(seed))
    if repeat_total > 1:
        items = rewrite_outputs_for_repeat(items, f"seed{seed}_run{repeat_index:02d}")
    return items


def build_command(scenario: dict, python_exe: str, scenario_args: list[str]) -> list[str]:
    return [
        python_exe,
        str(PROJECT_ROOT / "src" / "demo_map_simulation.py"),
        *scenario_args,
    ]


def run_command_with_log(command: list[str], env: dict[str, str], log_path: pathlib.Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
        return process.wait()


def run_scenario(
    path: pathlib.Path,
    args: argparse.Namespace,
    repeat_index: int,
    seed: int,
    *,
    scenario_override: dict | None = None,
    matrix_context: dict | None = None,
) -> dict:
    scenario = load_scenario(path)
    if scenario_override:
        scenario["args"] = [*scenario.get("args", []), *scenario_override.get("args", [])]
        scenario["env"] = {**scenario.get("env", {}), **scenario_override.get("env", {})}
    python_exe = args.python or os.environ.get("JUPEDSIM_PYTHON") or sys.executable
    scenario_args = scenario_args_for_run(scenario, seed, repeat_index, args.repeat)
    run_id = make_run_id(scenario, seed, repeat_index)
    run_dir = args.manifest_dir / run_id
    artifacts = {}
    if not args.legacy_output_layout:
        run_dir.mkdir(parents=True, exist_ok=False)
        scenario_args, artifacts = isolated_scenario_args(scenario_args, run_dir, run_id)
    command = build_command(scenario, python_exe, scenario_args)
    env = os.environ.copy()
    env.update({key: str(value) for key, value in scenario.get("env", {}).items()})

    started_at = dt.datetime.now(dt.UTC).isoformat()
    started_clock = time.monotonic()
    log_stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = pathlib.Path(artifacts.get("log", args.manifest_dir / "logs" / f"{scenario['name']}_seed{seed}_run{repeat_index:02d}_{log_stamp}.log"))
    resolved_scenario = {
        **scenario,
        "config_path": str(path.relative_to(PROJECT_ROOT)),
        "resolved_args": scenario_args,
        "resolved_env": {key: str(value) for key, value in scenario.get("env", {}).items()},
        "seed": seed,
        "run_id": run_id,
    }
    record = {
        "schema_version": "1.0",
        "run_id": run_id,
        "scenario_name": scenario["name"],
        "name": scenario["name"],
        "description": scenario.get("description", ""),
        "config": str(path.relative_to(PROJECT_ROOT)),
        "config_path": str(path.relative_to(PROJECT_ROOT)),
        "repeat_index": repeat_index,
        "repeat_total": args.repeat,
        "seed": seed,
        "command": command,
        "scenario_snapshot": scenario,
        "started_at": started_at,
        "dry_run": args.dry_run,
        "status": "planned" if args.dry_run else "running",
        "log": str(log_path),
        "artifacts": artifacts,
        "model": {
            "movement_model": get_option(scenario_args, "--movement-model", "cfsv3"),
            "llm_enabled": "--llm-routing" in scenario_args,
            "llm_model": get_option(scenario_args, "--llm-model"),
            "llm_base_url": get_option(scenario_args, "--llm-base-url"),
            "llm_cache_dir": get_option(scenario_args, "--llm-cache-dir"),
            "profile_source": get_option(scenario_args, "--profile-source", "mall"),
            "prompt_file": "src/llm_prompts.py",
            "prompt_sha256": file_sha256(PROJECT_ROOT / "src" / "llm_prompts.py"),
        },
        "reproducibility": reproducibility_context(python_exe, scenario_args),
    }
    if matrix_context:
        record["matrix"] = matrix_context
        resolved_scenario["matrix"] = matrix_context

    if not args.legacy_output_layout:
        atomic_write_json(pathlib.Path(artifacts["resolved_scenario"]), resolved_scenario)
        atomic_write_json(pathlib.Path(artifacts["manifest"]), record)

    print(f"\n== {scenario['name']} run {repeat_index}/{args.repeat} seed={seed} ==")
    print(" ".join(command))
    if args.dry_run:
        record["returncode"] = None
        record["finished_at"] = dt.datetime.now(dt.UTC).isoformat()
        record["duration_seconds"] = round(time.monotonic() - started_clock, 6)
        if not args.legacy_output_layout:
            atomic_write_json(pathlib.Path(artifacts["manifest"]), record)
        return record

    returncode = run_command_with_log(command, env, log_path)
    record["returncode"] = returncode
    record["finished_at"] = dt.datetime.now(dt.UTC).isoformat()
    record["duration_seconds"] = round(time.monotonic() - started_clock, 6)
    record["status"] = "completed" if returncode == 0 else "failed"
    if returncode != 0:
        record["failure"] = {"stage": "simulation", "returncode": returncode}
    if not args.legacy_output_layout:
        atomic_write_json(pathlib.Path(artifacts["manifest"]), record)
    if returncode != 0 and args.stop_on_failure:
        save_manifest([record], args.manifest_dir)
        raise SystemExit(returncode)
    return record


def save_manifest(records: list[dict], manifest_dir: pathlib.Path) -> pathlib.Path:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = manifest_dir / f"experiment_manifest_{timestamp}.json"
    atomic_write_json(path, {"schema_version": "1.0", "runs": records})
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenarios", nargs="*", help="Scenario names from configs/scenarios.")
    parser.add_argument("--list", action="store_true", help="List available scenarios and exit.")
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", help="Python executable used for demo_map_simulation.py.")
    parser.add_argument("--manifest-dir", type=pathlib.Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat each selected scenario N times.")
    parser.add_argument("--seed-start", type=int, default=2026, help="First seed used for repeated runs.")
    parser.add_argument(
        "--legacy-output-layout",
        action="store_true",
        help="Keep scenario-configured output paths instead of creating one directory per run.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        for path in list_scenarios():
            scenario = load_scenario(path)
            status = "enabled" if scenario.get("enabled", True) else "disabled"
            print(f"{path.stem:20} {status:8} {scenario.get('description', '')}")
        return 0

    selected = select_scenarios(args.scenarios, args.include_disabled)
    if not selected:
        raise SystemExit("No scenarios selected.")

    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")

    records = []
    for repeat_index in range(1, args.repeat + 1):
        seed = args.seed_start + repeat_index - 1
        for path in selected:
            records.append(run_scenario(path, args, repeat_index, seed))
    manifest = save_manifest(records, args.manifest_dir)
    print(f"\nManifest saved: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
