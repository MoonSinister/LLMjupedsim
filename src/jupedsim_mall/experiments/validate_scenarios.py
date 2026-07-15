#!/usr/bin/env python3
"""Validate experiment scenario JSON structure and command semantics."""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter
from collections.abc import Sequence

from jsonschema import Draft202012Validator

from jupedsim_mall.project import SCENARIO_DIR, SCHEMA_DIR


SCENARIO_SCHEMA_PATH = SCHEMA_DIR / "scenario.schema.json"


VALUE_OPTIONS = {
    "-g",
    "--geometry",
    "-n",
    "--num-agents",
    "--seed",
    "--spawn-interval",
    "--spawn-jitter",
    "--max-iters",
    "--simulation-dt",
    "--movement-model",
    "--agent-radius",
    "--agent-time-gap",
    "--neighbor-repulsion-strength",
    "--neighbor-repulsion-range",
    "--geometry-repulsion-strength",
    "--geometry-repulsion-range",
    "--cfsv3-range-x-scale",
    "--cfsv3-range-y-scale",
    "--cfsv3-theta-max",
    "--cfsv3-agent-buffer",
    "--avm-wall-buffer-distance",
    "--avm-anticipation-time",
    "--avm-reaction-time",
    "--max-active-agents",
    "--agent-max-lifetime-seconds",
    "--congestion-check-interval",
    "--congestion-grace-iterations",
    "--stuck-distance-threshold",
    "--stuck-checks-before-reroute",
    "--congestion-release-radius",
    "--congestion-pair-radius",
    "--congestion-detour-max-attempts",
    "--congestion-detour-distance",
    "--congestion-detour-wall-margin",
    "--congestion-detour-wait-seconds",
    "--output",
    "--regions",
    "--profile-source",
    "--llmob-data-root",
    "--llmob-dataset",
    "--llmob-max-persons",
    "--llmob-profile-cache",
    "--atc-raw-path",
    "--atc-regions",
    "--atc-max-persons",
    "--atc-max-rows",
    "--atc-min-points",
    "--atc-profile-cache",
    "--atc-partition",
    "--baseline-routing",
    "--llm-base-url",
    "--llm-model",
    "--replay-plan",
    "--llm-timeout",
    "--llm-max-tokens",
    "--llm-temperature",
    "--llm-max-retries",
    "--llm-batch-size",
    "--llm-max-regions-per-agent",
    "--llm-plan-output",
    "--llm-cache-dir",
    "--llm-calls-output",
    "--region-target-margin",
    "--waypoint-distance",
    "--routing-waypoint-distance",
    "--routing-waypoint-max-per-leg",
}

FLAG_OPTIONS = {
    "--disable-stuck-reroute",
    "--enable-congestion-detour",
    "--llm-routing",
    "--disable-routing-waypoints",
    "--disable-activity-waiting",
    "--homogeneous-profiles",
}

OUTPUT_OPTIONS = {
    "--output",
    "--llm-plan-output",
    "--llmob-profile-cache",
    "--atc-profile-cache",
}

INTEGER_OPTIONS = {
    "-n": (1, None),
    "--num-agents": (1, None),
    "--seed": (0, None),
    "--spawn-interval": (0, None),
    "--spawn-jitter": (0, None),
    "--max-iters": (1, None),
    "--max-active-agents": (1, None),
    "--congestion-check-interval": (1, None),
    "--congestion-grace-iterations": (0, None),
    "--stuck-checks-before-reroute": (1, None),
    "--congestion-detour-max-attempts": (0, None),
    "--llmob-max-persons": (0, None),
    "--atc-max-persons": (0, None),
    "--atc-max-rows": (0, None),
    "--atc-min-points": (1, None),
    "--llm-max-tokens": (1, None),
    "--llm-max-retries": (0, None),
    "--llm-batch-size": (1, None),
    "--llm-max-regions-per-agent": (0, None),
    "--routing-waypoint-max-per-leg": (0, None),
}

FLOAT_OPTIONS = {
    "--simulation-dt": (0.000001, None),
    "--agent-radius": (0.01, 1.0),
    "--agent-time-gap": (0.0, None),
    "--agent-max-lifetime-seconds": (0.0, None),
    "--stuck-distance-threshold": (0.0, None),
    "--congestion-release-radius": (0.0, None),
    "--llm-timeout": (0.0, None),
    "--llm-temperature": (0.0, 2.0),
    "--region-target-margin": (0.0, None),
    "--waypoint-distance": (0.0, None),
    "--routing-waypoint-distance": (0.0, None),
}

ENUM_OPTIONS = {
    "--movement-model": {"avm", "cfs", "cfsv2", "cfsv3"},
    "--profile-source": {"mall", "llmob", "atc"},
    "--baseline-routing": {"random", "nearest"},
    "--atc-partition": {"train", "tuning", "evaluation", "all"},
}


def load_scenario(path: pathlib.Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("scenario root must be a JSON object")
    return data


def schema_errors(scenario: dict) -> list[str]:
    schema = json.loads(SCENARIO_SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors = []
    for error in sorted(validator.iter_errors(scenario), key=lambda item: list(item.absolute_path)):
        field = ".".join(str(part) for part in error.absolute_path) or "<root>"
        errors.append(f"schema {field}: {error.message}")
    return errors


def option_values(items: list[str]) -> dict[str, str]:
    values = {}
    index = 0
    while index < len(items):
        option = items[index]
        if option in VALUE_OPTIONS and index + 1 < len(items):
            values[option] = items[index + 1]
            index += 2
        else:
            index += 1
    return values


def _validate_number(option: str, value: str, integer: bool, bounds: tuple[float | None, float | None]) -> list[str]:
    try:
        parsed = int(value) if integer else float(value)
    except ValueError:
        return [f"{option} must be {'an integer' if integer else 'numeric'}, got {value!r}"]
    minimum, maximum = bounds
    if minimum is not None and parsed < minimum:
        return [f"{option} must be >= {minimum}, got {parsed}"]
    if maximum is not None and parsed > maximum:
        return [f"{option} must be <= {maximum}, got {parsed}"]
    return []


def validate_semantics(scenario: dict, items: list[str]) -> list[str]:
    errors = []
    values = option_values(items)
    for option, bounds in INTEGER_OPTIONS.items():
        if option in values:
            errors.extend(_validate_number(option, values[option], True, bounds))
    for option, bounds in FLOAT_OPTIONS.items():
        if option in values:
            errors.extend(_validate_number(option, values[option], False, bounds))
    for option, choices in ENUM_OPTIONS.items():
        if option in values and values[option] not in choices:
            errors.append(f"{option} must be one of {sorted(choices)}, got {values[option]!r}")

    profile_source = values.get("--profile-source", "mall")
    if profile_source == "llmob" and "--llmob-profile-cache" not in values:
        errors.append("--profile-source llmob requires --llmob-profile-cache")
    if profile_source == "atc" and "--atc-profile-cache" not in values:
        errors.append("--profile-source atc requires --atc-profile-cache")

    if "--llm-routing" in items:
        has_url_arg = bool(values.get("--llm-base-url"))
        has_url_env = bool(str(scenario.get("env", {}).get("LOCAL_LLM_BASE_URL", "")).strip())
        if not has_url_arg and not has_url_env:
            errors.append("--llm-routing requires --llm-base-url or env.LOCAL_LLM_BASE_URL")
    if "--llm-routing" in items and "--replay-plan" in values:
        errors.append("--llm-routing and --replay-plan are mutually exclusive")
    replay_plan = values.get("--replay-plan")
    if replay_plan and pathlib.Path(replay_plan).suffix.lower() != ".json":
        errors.append(f"--replay-plan must use a .json file, got {replay_plan!r}")

    for option in OUTPUT_OPTIONS:
        target = values.get(option)
        if not target:
            continue
        expected_suffix = ".sqlite" if option == "--output" else ".json"
        if pathlib.Path(target).suffix.lower() != expected_suffix:
            errors.append(f"{option} must use a {expected_suffix} target, got {target!r}")
    return errors


def validate_args(items: list[str]) -> list[str]:
    errors = []
    index = 0
    while index < len(items):
        option = items[index]
        if option in VALUE_OPTIONS:
            if index + 1 >= len(items) or str(items[index + 1]).startswith("-"):
                errors.append(f"{option} is missing a value")
                index += 1
            else:
                index += 2
            continue
        if option in FLAG_OPTIONS:
            index += 1
            continue
        if str(option).startswith("-"):
            errors.append(f"unknown option {option}")
            index += 1
            continue
        errors.append(f"unexpected bare value {option}")
        index += 1
    return errors


def output_targets(items: list[str]) -> list[str]:
    targets = []
    index = 0
    while index < len(items):
        if items[index] in OUTPUT_OPTIONS and index + 1 < len(items):
            targets.append(str(items[index + 1]))
            index += 2
            continue
        index += 1
    return targets


def validate_scenario(path: pathlib.Path) -> list[str]:
    errors = []
    scenario = load_scenario(path)
    errors.extend(schema_errors(scenario))
    name = scenario.get("name", path.stem)
    if name != path.stem:
        errors.append(f"name '{name}' does not match filename '{path.stem}'")
    if "description" not in scenario or not str(scenario.get("description", "")).strip():
        errors.append("description is required")
    if not isinstance(scenario.get("enabled", True), bool):
        errors.append("enabled must be boolean")
    args = scenario.get("args", [])
    if not isinstance(args, list):
        errors.append("args must be a list")
        args = []
    args = [str(item) for item in args]
    errors.extend(validate_args(args))
    errors.extend(validate_semantics(scenario, args))
    if not scenario.get("enabled", True) and not external_hint_present(scenario):
        errors.append("disabled scenario should describe external prerequisites")
    return errors


def external_hint_present(scenario: dict) -> bool:
    text = " ".join([
        str(scenario.get("description", "")),
        json.dumps(scenario.get("env", {}), ensure_ascii=False),
        " ".join(str(item) for item in scenario.get("args", [])),
    ]).lower()
    return any(token in text for token in ["set ", "path", "root", "atc", "llmob", "remote", "endpoint"])


def validate_all(paths: list[pathlib.Path]) -> int:
    all_errors = {}
    output_counts = Counter()
    output_sources: dict[str, list[str]] = {}

    for path in paths:
        try:
            scenario = load_scenario(path)
            args = [str(item) for item in scenario.get("args", []) if item is not None]
            for target in output_targets(args):
                output_counts[target] += 1
                output_sources.setdefault(target, []).append(path.stem)
            errors = validate_scenario(path)
        except Exception as exc:
            errors = [str(exc)]
        if errors:
            all_errors[path.name] = errors

    for target, count in output_counts.items():
        if count > 1:
            names = ", ".join(output_sources[target])
            all_errors.setdefault("<outputs>", []).append(f"duplicate output target {target}: {names}")

    if all_errors:
        for name, errors in all_errors.items():
            print(f"{name}:")
            for error in errors:
                print(f"  - {error}")
        return 1

    print(f"Validated {len(paths)} scenario file(s)")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenarios", nargs="*", type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = args.scenarios or sorted(SCENARIO_DIR.glob("*.json"))
    return validate_all(paths)


if __name__ == "__main__":
    raise SystemExit(main())
