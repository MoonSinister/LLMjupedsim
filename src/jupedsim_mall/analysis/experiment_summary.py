#!/usr/bin/env python3
"""Summarize experiment artifacts into JSON and CSV.

The summary is intentionally conservative: it reads LLM plan/profile caches and
generic SQLite table counts without assuming one fixed JuPedSim schema.
"""

from __future__ import annotations

from collections import Counter
import argparse
import csv
import json
import math
import pathlib
import re
import sqlite3
import statistics
from collections.abc import Sequence

from jupedsim_mall.analysis.metrics import (
    entropy_from_counts,
    histogram_counts,
    js_divergence_from_counts,
    plan_distribution_metrics,
)
from jupedsim_mall.project import PROJECT_ROOT


def summarize_plan(path: pathlib.Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    agents = data.get("agents", [])
    validation = data.get("validation", {})
    metadata = data.get("metadata", {})
    roles = Counter(agent.get("role", "unknown") for agent in agents)
    exits = Counter(agent.get("final_exit", "unknown") for agent in agents)
    ttl_removed = sum(1 for agent in agents if agent.get("removed_due_to_ttl"))
    rerouted = sum(1 for agent in agents if agent.get("rerouted_due_to_stuck"))
    activity_counts = [len(agent.get("activities", [])) for agent in agents]
    candidate_violations = validation.get("invalid_candidate_exit", 0)
    distributions = plan_distribution_metrics(agents)
    return {
        "artifact_type": "plan",
        "scenario": scenario_name_from_path(path),
        "path": str(path),
        "route_policy": metadata.get("route_policy", ""),
        "seed": metadata.get("seed", ""),
        "agents": len(agents),
        "roles": dict(roles),
        "exits": dict(exits),
        "exit_entropy": entropy_from_counts(exits),
        "exit_entropy_normalized": entropy_from_counts(exits, normalized=True),
        "region_visit_distribution": distributions["region_visit_distribution"],
        "activity_action_distribution": distributions["activity_action_distribution"],
        "activity_count_distribution": distributions["activity_count_distribution"],
        "wait_seconds_distribution": distributions["wait_seconds_distribution"],
        "ttl_removed": ttl_removed,
        "ttl_removed_rate": round(ttl_removed / len(agents), 4) if agents else 0,
        "rerouted_due_to_stuck": rerouted,
        "rerouted_rate": round(rerouted / len(agents), 4) if agents else 0,
        "avg_activities": round(sum(activity_counts) / len(activity_counts), 3) if activity_counts else 0,
        "validation_valid": validation.get("valid", ""),
        "invalid_candidate_exit": candidate_violations,
        "invalid_exit_indices": validation.get("invalid_exit_indices", 0),
        "invalid_regions": validation.get("invalid_regions", 0),
        "invalid_wait_seconds": validation.get("invalid_wait_seconds", 0),
        "invalid_desired_speed": validation.get("invalid_desired_speed", 0),
    }


def summarize_profile(path: pathlib.Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    profiles = data.get("profiles", data if isinstance(data, list) else [])
    roles = Counter(profile.get("role", "unknown") for profile in profiles if isinstance(profile, dict))
    return {
        "artifact_type": "profile",
        "scenario": scenario_name_from_path(path),
        "path": str(path),
        "profiles": len(profiles),
        "roles": dict(roles),
    }


def summarize_sqlite(path: pathlib.Path) -> dict:
    result = {
        "artifact_type": "trajectory",
        "scenario": scenario_name_from_path(path),
        "path": str(path),
        "tables": {},
    }
    with sqlite3.connect(path) as conn:
        table_rows = conn.execute(
            "select name from sqlite_master where type='table' order by name"
        ).fetchall()
        for (table,) in table_rows:
            try:
                count = conn.execute(f'select count(*) from "{table}"').fetchone()[0]
            except sqlite3.DatabaseError:
                count = None
            result["tables"][table] = count
        if "trajectory_data" in result["tables"]:
            frames = conn.execute("select count(distinct frame) from trajectory_data").fetchone()[0]
            agents = conn.execute("select count(distinct id) from trajectory_data").fetchone()[0]
            first_last = conn.execute("select min(frame), max(frame) from trajectory_data").fetchone()
            fps = metadata_float(conn, "fps", 0.0)
            result["trajectory_frames"] = frames
            result["trajectory_agents_observed"] = agents
            result["trajectory_first_frame"] = first_last[0]
            result["trajectory_last_frame"] = first_last[1]
            if fps > 0 and first_last[0] is not None and first_last[1] is not None:
                result["trajectory_duration_seconds"] = round((first_last[1] - first_last[0]) / fps, 3)
                durations = [
                    (last_frame - first_frame) / fps
                    for first_frame, last_frame in conn.execute(
                        "select min(frame), max(frame) from trajectory_data group by id"
                    )
                ]
                if durations:
                    result["avg_observed_travel_seconds"] = round(sum(durations) / len(durations), 3)
                    result["max_observed_travel_seconds"] = round(max(durations), 3)
                    result["travel_seconds_distribution"] = histogram_counts(durations, bin_size=30, max_value=300)
                path_stats = trajectory_path_stats(conn, fps)
                result.update(path_stats)
    return result


def trajectory_path_stats(conn: sqlite3.Connection, fps: float) -> dict:
    try:
        rows = conn.execute(
            "select id, frame, pos_x, pos_y from trajectory_data order by id, frame"
        ).fetchall()
    except sqlite3.DatabaseError:
        return {}

    previous: dict[int, tuple[int, float, float]] = {}
    path_lengths: dict[int, float] = {}
    frame_ranges: dict[int, list[int]] = {}
    for agent_id, frame, x, y in rows:
        agent_id = int(agent_id)
        frame = int(frame)
        x = float(x)
        y = float(y)
        frame_ranges.setdefault(agent_id, [frame, frame])
        frame_ranges[agent_id][1] = frame
        prev = previous.get(agent_id)
        if prev is not None:
            _, px, py = prev
            path_lengths[agent_id] = path_lengths.get(agent_id, 0.0) + math.hypot(x - px, y - py)
        else:
            path_lengths.setdefault(agent_id, 0.0)
        previous[agent_id] = (frame, x, y)

    lengths = list(path_lengths.values())
    if not lengths:
        return {}

    result = {
        "avg_path_length_m": round(sum(lengths) / len(lengths), 3),
        "max_path_length_m": round(max(lengths), 3),
        "path_length_distribution": histogram_counts(lengths, bin_size=5, max_value=80),
    }

    if fps > 0:
        speeds = []
        for agent_id, length in path_lengths.items():
            first_frame, last_frame = frame_ranges.get(agent_id, [0, 0])
            duration = (last_frame - first_frame) / fps
            if duration > 0:
                speeds.append(length / duration)
        if speeds:
            result["avg_observed_speed_mps"] = round(sum(speeds) / len(speeds), 3)
            result["max_observed_speed_mps"] = round(max(speeds), 3)
            result["observed_speed_distribution"] = histogram_counts(speeds, bin_size=0.25, max_value=2.5)
    return result


def metadata_float(conn: sqlite3.Connection, key: str, default: float) -> float:
    try:
        row = conn.execute("select value from metadata where key = ?", (key,)).fetchone()
    except sqlite3.DatabaseError:
        return default
    if row is None:
        return default
    try:
        return float(row[0])
    except (TypeError, ValueError):
        return default


def scenario_name_from_path(path: pathlib.Path) -> str:
    name = path.stem
    name = re.sub(r"_seed\d+_run\d+$", "", name)
    name = re.sub(r"_plan$", "", name)
    return name


def find_artifacts(root: pathlib.Path) -> list[pathlib.Path]:
    patterns = [
        "outputs/plans/*.json",
        "outputs/profiles/*.json",
        "outputs/trajectories/*.sqlite",
        "outputs/smoke/*.sqlite",
    ]
    paths: list[pathlib.Path] = []
    for pattern in patterns:
        paths.extend(root.glob(pattern))
    return sorted(paths)


def summarize(path: pathlib.Path) -> dict:
    if path.suffix.lower() in {".sqlite", ".db"}:
        return summarize_sqlite(path)
    if "profiles" in path.parts:
        return summarize_profile(path)
    return summarize_plan(path)


def write_csv(rows: list[dict], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "artifact_type",
        "scenario",
        "runs",
        "path",
        "route_policy",
        "seed",
        "agents",
        "profiles",
        "ttl_removed",
        "ttl_removed_rate",
        "rerouted_due_to_stuck",
        "rerouted_rate",
        "avg_activities",
        "exit_entropy",
        "exit_entropy_normalized",
        "validation_valid",
        "invalid_candidate_exit",
        "invalid_exit_indices",
        "invalid_regions",
        "invalid_wait_seconds",
        "invalid_desired_speed",
        "trajectory_frames",
        "trajectory_agents_observed",
        "trajectory_duration_seconds",
        "avg_observed_travel_seconds",
        "max_observed_travel_seconds",
        "travel_seconds_distribution",
        "avg_path_length_m",
        "max_path_length_m",
        "path_length_distribution",
        "avg_observed_speed_mps",
        "max_observed_speed_mps",
        "observed_speed_distribution",
        "roles",
        "exits",
        "region_visit_distribution",
        "activity_action_distribution",
        "activity_count_distribution",
        "wait_seconds_distribution",
        "tables",
    ]
    extra = sorted({key for row in rows for key in row} - set(preferred))
    fieldnames = [key for key in preferred if any(key in row for row in rows)] + extra
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(row.get(key, ""), ensure_ascii=False)
                if isinstance(row.get(key), (dict, list))
                else row.get(key, "")
                for key in fieldnames
            })


def aggregate_rows(rows: list[dict]) -> list[dict]:
    numeric_fields = [
        "agents",
        "ttl_removed",
        "ttl_removed_rate",
        "rerouted_due_to_stuck",
        "rerouted_rate",
        "avg_activities",
        "exit_entropy",
        "exit_entropy_normalized",
        "invalid_candidate_exit",
        "invalid_regions",
        "trajectory_agents_observed",
        "trajectory_duration_seconds",
        "avg_observed_travel_seconds",
        "max_observed_travel_seconds",
        "avg_path_length_m",
        "max_path_length_m",
        "avg_observed_speed_mps",
        "max_observed_speed_mps",
    ]
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row.get("artifact_type", ""), row.get("scenario", "")), []).append(row)

    summaries = []
    for (artifact_type, scenario), group_rows in sorted(groups.items()):
        summary = {
            "artifact_type": artifact_type,
            "scenario": scenario,
            "runs": len(group_rows),
        }
        for field in numeric_fields:
            values = [
                float(row[field])
                for row in group_rows
                if isinstance(row.get(field), (int, float))
            ]
            if not values:
                continue
            summary[f"{field}_mean"] = round(statistics.mean(values), 4)
            summary[f"{field}_std"] = round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def comparison_rows(rows: list[dict], baseline: str) -> list[dict]:
    plan_rows = [row for row in rows if row.get("artifact_type") == "plan"]
    baseline_rows = [row for row in plan_rows if row.get("scenario") == baseline]
    if not baseline_rows:
        return []

    distribution_fields = [
        "roles",
        "exits",
        "region_visit_distribution",
        "activity_action_distribution",
        "activity_count_distribution",
        "wait_seconds_distribution",
    ]
    baseline_dist = {
        field: merge_count_dicts([row.get(field, {}) for row in baseline_rows])
        for field in distribution_fields
    }

    comparisons = []
    for row in plan_rows:
        scenario = row.get("scenario", "")
        if scenario == baseline:
            continue
        comparison = {
            "artifact_type": "plan_comparison",
            "scenario": scenario,
            "baseline": baseline,
        }
        for field in distribution_fields:
            current = row.get(field, {})
            if isinstance(current, dict):
                comparison[f"{field}_jsd"] = js_divergence_from_counts(current, baseline_dist[field])
        comparisons.append(comparison)
    return comparisons


def merge_count_dicts(items: list[dict]) -> dict[str, float]:
    counter: Counter[str] = Counter()
    for item in items:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            try:
                counter[str(key)] += float(value)
            except (TypeError, ValueError):
                continue
    return dict(counter)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="*", type=pathlib.Path)
    parser.add_argument("--output-json", type=pathlib.Path, default=PROJECT_ROOT / "outputs" / "summaries" / "experiment_summary.json")
    parser.add_argument("--output-csv", type=pathlib.Path, default=PROJECT_ROOT / "outputs" / "summaries" / "experiment_summary.csv")
    parser.add_argument("--aggregate-json", type=pathlib.Path, default=PROJECT_ROOT / "outputs" / "summaries" / "experiment_summary_aggregate.json")
    parser.add_argument("--aggregate-csv", type=pathlib.Path, default=PROJECT_ROOT / "outputs" / "summaries" / "experiment_summary_aggregate.csv")
    parser.add_argument("--comparison-json", type=pathlib.Path, default=PROJECT_ROOT / "outputs" / "summaries" / "experiment_summary_comparison.json")
    parser.add_argument("--comparison-csv", type=pathlib.Path, default=PROJECT_ROOT / "outputs" / "summaries" / "experiment_summary_comparison.csv")
    parser.add_argument("--comparison-baseline", default="baseline_random")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = args.artifacts or find_artifacts(PROJECT_ROOT)
    rows = [summarize(path) for path in paths if path.exists()]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"artifacts": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(rows, args.output_csv)
    aggregate = aggregate_rows(rows)
    args.aggregate_json.parent.mkdir(parents=True, exist_ok=True)
    args.aggregate_json.write_text(json.dumps({"groups": aggregate}, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(aggregate, args.aggregate_csv)
    comparisons = comparison_rows(rows, args.comparison_baseline)
    args.comparison_json.parent.mkdir(parents=True, exist_ok=True)
    args.comparison_json.write_text(
        json.dumps({"baseline": args.comparison_baseline, "comparisons": comparisons}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(comparisons, args.comparison_csv)
    print(f"Summarized {len(rows)} artifact(s)")
    print(f"JSON: {args.output_json}")
    print(f"CSV : {args.output_csv}")
    print(f"Aggregate JSON: {args.aggregate_json}")
    print(f"Aggregate CSV : {args.aggregate_csv}")
    print(f"Comparison JSON: {args.comparison_json}")
    print(f"Comparison CSV : {args.comparison_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
