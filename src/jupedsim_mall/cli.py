"""Unified command line entry point for the experiment project."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jupedsim-mall")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check the local experiment environment.")
    doctor.add_argument("--check-network", action="store_true", help="Probe the configured LLM endpoint.")
    doctor.add_argument("--json", action="store_true", help="Write machine-readable diagnostics.")

    scenarios = subparsers.add_parser("scenarios", help="Inspect or validate scenario files.")
    scenario_commands = scenarios.add_subparsers(dest="scenario_command", required=True)
    scenario_commands.add_parser("list", help="List configured scenarios.")
    validate = scenario_commands.add_parser("validate", help="Validate scenario JSON and semantics.")
    validate.add_argument("paths", nargs="*", help="Optional scenario JSON paths.")

    map_parser = subparsers.add_parser("map", help="Inspect and validate map inputs.")
    map_commands = map_parser.add_subparsers(dest="map_command", required=True)
    map_validate = map_commands.add_parser("validate", help="Run geometry, stage, and region checks.")
    map_validate.add_argument("--map", default="data/map", help="Map directory.")
    map_validate.add_argument("--output", help="Optional JSON quality report path.")
    map_validate.add_argument("--preview", help="Optional PNG preview path.")

    profiles = subparsers.add_parser("profiles", help="Prepare and inspect agent profile sources.")
    profile_commands = profiles.add_subparsers(dest="profile_command", required=True)
    prepare = profile_commands.add_parser("prepare", help="Build a versioned profile cache.")
    prepare.add_argument("--source", choices=["mall", "atc", "llmob"], required=True)
    prepare.add_argument("--data-path", default="")
    prepare.add_argument("--regions", default="data/map/localization_grid_regions.json")
    prepare.add_argument("--dataset", choices=["2019", "2021", "20192021"], default="2019")
    prepare.add_argument("--max-persons", type=int, default=0)
    prepare.add_argument("--max-rows", type=int, default=0)
    prepare.add_argument("--min-points", type=int, default=300)
    prepare.add_argument("--seed", type=int, default=2026)
    prepare.add_argument("--cache-output", default="")
    prepare.add_argument("--report-output", default="")
    prepare.add_argument("--strict", action="store_true", help="Fail instead of falling back to mall profiles.")
    prepare.add_argument("--atc-partition", choices=["train", "tuning", "evaluation", "all"], default="train")

    runs = subparsers.add_parser("runs", help="Inspect isolated experiment run artifacts.")
    run_commands = runs.add_subparsers(dest="runs_command", required=True)
    verify = run_commands.add_parser("verify", help="Verify manifests and required run artifacts.")
    verify.add_argument("paths", nargs="*", help="Run directories or manifest files.")
    verify.add_argument("--json", action="store_true")
    verify.add_argument("--latest", action="store_true")

    subparsers.add_parser("matrix", help="Prepare and execute a persistent experiment matrix.")
    subparsers.add_parser("analyze", help="Extract metrics and run statistical analyses.")
    subparsers.add_parser("report", help="Build traceable figures, tables, and report.")
    subparsers.add_parser("quality-gate", help="Run all pre-experiment quality checks.")
    subparsers.add_parser("experiment", help="Control formal experiment freezing and finalization.")
    subparsers.add_parser("release", help="Build or verify a reproduction package.")
    subparsers.add_parser("pipeline", help="Run validation, matrix, analysis, and reporting end to end.")

    subparsers.add_parser("run", help="Run one or more configured scenarios.")
    subparsers.add_parser("summarize", help="Summarize experiment artifacts.")
    subparsers.add_parser("realism", help="Compare simulation trajectories with an ATC reference.")
    subparsers.add_parser("atc-reference", help="Build ATC reference distributions.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and raw_args[0] == "run":
        from jupedsim_mall.experiments import run_experiment_suite

        return run_experiment_suite.main(raw_args[1:])
    if raw_args and raw_args[0] == "matrix":
        from jupedsim_mall.experiments import matrix_runner

        return matrix_runner.main(raw_args[1:])
    if raw_args[:2] == ["analyze", "metrics"]:
        from jupedsim_mall.analysis import run_metrics

        return run_metrics.main(raw_args[2:])
    if raw_args[:2] == ["analyze", "realism"]:
        from jupedsim_mall.analysis import realism_evaluation

        return realism_evaluation.main(raw_args[2:])
    if raw_args[:2] == ["analyze", "statistics"]:
        from jupedsim_mall.analysis import statistics

        return statistics.main(raw_args[2:])
    if raw_args[:2] == ["analyze", "states"]:
        from jupedsim_mall.analysis import behavior_states

        return behavior_states.main(raw_args[2:])
    if raw_args[:2] == ["report", "build"]:
        from jupedsim_mall.analysis import report_builder

        return report_builder.main(raw_args[2:])
    if raw_args and raw_args[0] == "quality-gate":
        from jupedsim_mall import quality_gate

        return quality_gate.main(raw_args[1:])
    if raw_args and raw_args[0] == "experiment":
        from jupedsim_mall.experiments import experiment_control

        return experiment_control.main(raw_args[1:])
    if raw_args and raw_args[0] == "release":
        from jupedsim_mall import release_builder

        return release_builder.main(raw_args[1:])
    if raw_args and raw_args[0] == "pipeline":
        from jupedsim_mall import pipeline

        return pipeline.main(raw_args[1:])
    if raw_args and raw_args[0] == "summarize":
        from jupedsim_mall.analysis import experiment_summary

        return experiment_summary.main(raw_args[1:])
    if raw_args and raw_args[0] == "realism":
        from jupedsim_mall.analysis import realism_evaluation

        return realism_evaluation.main(raw_args[1:])
    if raw_args and raw_args[0] == "atc-reference":
        from jupedsim_mall.analysis import atc_reference

        return atc_reference.main(raw_args[1:])
    if raw_args[:2] == ["runs", "verify"]:
        from jupedsim_mall.experiments import verify_runs

        return verify_runs.main(raw_args[2:])

    args = build_parser().parse_args(raw_args)
    if args.command == "doctor":
        from jupedsim_mall.doctor import run_doctor

        return run_doctor(check_network=args.check_network, json_output=args.json)
    if args.command == "scenarios":
        from jupedsim_mall.experiments import run_experiment_suite, validate_scenarios

        if args.scenario_command == "list":
            return run_experiment_suite.main(["--list"])
        return validate_scenarios.main(args.paths)
    if args.command == "map":
        from jupedsim_mall.geometry.map_io import validate_map, write_map_preview

        report = validate_map(args.map)
        payload = report.to_json()
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.preview:
            write_map_preview(args.map, args.preview)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if report.valid else 1
    if args.command == "profiles":
        from jupedsim_mall.profiles.providers import build_profile_provider

        result = build_profile_provider(
            args.source,
            seed=args.seed,
            data_path=args.data_path,
            regions_path=args.regions,
            dataset=args.dataset,
            max_persons=args.max_persons,
            max_rows=args.max_rows,
            min_points=args.min_points,
            cache_output=args.cache_output,
            allow_fallback=not args.strict,
            atc_partition=args.atc_partition,
        )
        payload = result.report.to_json()
        if args.report_output:
            output = Path(args.report_output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if not args.strict or payload["active_source"] == args.source else 1
    raise AssertionError(f"Unhandled command: {args.command}")
