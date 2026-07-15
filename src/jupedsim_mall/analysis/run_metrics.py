"""Extract traceable run- and agent-level metrics from isolated artifacts."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import pathlib
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Sequence
from statistics import mean
from typing import Any

from shapely.geometry import Point, Polygon

from jupedsim_mall.analysis.metrics import entropy_from_counts, histogram_counts
from jupedsim_mall.experiments.verify_runs import discover_manifests, verify_manifest
from jupedsim_mall.project import PROJECT_ROOT

SCHEMA_VERSION = "1.0"
SLOW_SPEED_MPS = 0.2
STOP_MIN_SECONDS = 2.0
DENSITY_RADIUS_M = 1.0
CONGESTION_DENSITY_PER_M2 = 1.5

METRIC_DEFINITIONS = {
    "completion_rate": {"unit": "ratio", "denominator": "spawned_agents", "missing": "null if no spawned agents"},
    "ttl_removal_rate": {"unit": "ratio", "denominator": "spawned_agents", "missing": "null if no spawned agents"},
    "failure_rate": {"unit": "ratio", "denominator": "planned_agents", "missing": "null if no planned agents"},
    "fallback_rate": {"unit": "ratio", "denominator": "planned_agents", "missing": "null if no planned agents"},
    "mean_travel_time_seconds": {"unit": "s", "denominator": "agents with at least two observed frames", "missing": "null"},
    "mean_path_length_m": {"unit": "m", "denominator": "observed agents", "missing": "null"},
    "mean_speed_mps": {"unit": "m/s", "denominator": "agents with positive observed duration", "missing": "null"},
    "mean_directness": {"unit": "ratio", "denominator": "agents with positive path length", "missing": "null"},
    "mean_slow_ratio": {"unit": "ratio", "denominator": "observed movement time per agent", "missing": "null"},
    "mean_wait_seconds": {"unit": "s", "denominator": "spawned agents", "missing": "0 when no wait event exists"},
    "mean_local_density_per_m2": {"unit": "agents/m2", "denominator": "observed agent-frames", "missing": "null"},
    "peak_local_density_per_m2": {"unit": "agents/m2", "denominator": "maximum observed agent-frame", "missing": "null"},
    "congestion_duration_seconds": {"unit": "s", "denominator": "frames with any density above threshold", "missing": "0"},
    "reroute_rate": {"unit": "ratio", "denominator": "spawned_agents", "missing": "null if no spawned agents"},
    "reroute_success_rate": {"unit": "ratio", "denominator": "rerouted agents", "missing": "null if no rerouted agents"},
    "mean_reroute_extra_path_m": {"unit": "m", "denominator": "rerouted agents with trajectory after reroute", "missing": "null"},
    "exit_entropy": {"unit": "nats", "denominator": "planned agents with final exit", "missing": "0"},
}


def _atomic_write(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _artifact_path(manifest_path: pathlib.Path, value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    if path.is_absolute():
        return path
    project_path = PROJECT_ROOT / path
    return project_path if project_path.exists() else manifest_path.parent / path


def detect_trajectory_schema(connection: sqlite3.Connection) -> dict:
    tables = {
        row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    aliases = {
        "agent_id": ("id", "agent_id", "pedestrian_id"),
        "frame": ("frame", "frame_id", "step", "iteration"),
        "x": ("pos_x", "x", "position_x"),
        "y": ("pos_y", "y", "position_y"),
    }
    for table in ("trajectory_data", "trajectory", "positions"):
        if table not in tables:
            continue
        columns = {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}
        resolved = {}
        for role, candidates in aliases.items():
            resolved[role] = next((name for name in candidates if name in columns), None)
        if all(resolved.values()):
            return {"table": table, "columns": resolved, "tables": sorted(tables)}
    raise ValueError(f"no supported trajectory table; available tables={sorted(tables)}")


def _fps(connection: sqlite3.Connection) -> float:
    try:
        row = connection.execute("SELECT value FROM metadata WHERE key='fps'").fetchone()
        value = float(row[0]) if row else 0.0
    except (sqlite3.Error, TypeError, ValueError):
        value = 0.0
    return value if value > 0 else 25.0


def load_trajectories(path: pathlib.Path) -> tuple[dict, float, dict[int, list[tuple[int, float, float]]]]:
    with sqlite3.connect(path) as connection:
        schema = detect_trajectory_schema(connection)
        columns = schema["columns"]
        query = (
            f'SELECT "{columns["agent_id"]}", "{columns["frame"]}", '
            f'"{columns["x"]}", "{columns["y"]}" FROM "{schema["table"]}" '
            f'ORDER BY "{columns["agent_id"]}", "{columns["frame"]}"'
        )
        trajectories: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
        for agent_id, frame, x, y in connection.execute(query):
            trajectories[int(agent_id)].append((int(frame), float(x), float(y)))
        return schema, _fps(connection), dict(trajectories)


def load_events(path: pathlib.Path) -> list[dict]:
    events: list[dict] = []
    if not path.is_file():
        return events
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def load_regions(path: pathlib.Path) -> list[tuple[str, Polygon]]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("regions", payload if isinstance(payload, list) else [])
    regions = []
    for item in items:
        points = item.get("points_world") or item.get("points")
        if isinstance(points, list) and len(points) >= 3:
            polygon = Polygon(points)
            if polygon.is_valid and not polygon.is_empty:
                regions.append((str(item.get("name", "unknown")), polygon))
    return regions


def _region_at(x: float, y: float, regions: list[tuple[str, Polygon]]) -> str:
    point = Point(x, y)
    for name, polygon in regions:
        if polygon.covers(point):
            return name
    return "unmapped"


def _trajectory_metrics(points: list[tuple[int, float, float]], fps: float) -> dict:
    if not points:
        return {}
    path_length = 0.0
    slow_seconds = 0.0
    stop_count = 0
    stop_seconds = 0.0
    current_stop = 0.0
    for (frame_a, x_a, y_a), (frame_b, x_b, y_b) in zip(points, points[1:]):
        elapsed = max(0.0, (frame_b - frame_a) / fps)
        distance = math.hypot(x_b - x_a, y_b - y_a)
        path_length += distance
        speed = distance / elapsed if elapsed > 0 else 0.0
        if elapsed > 0 and speed < SLOW_SPEED_MPS:
            slow_seconds += elapsed
            current_stop += elapsed
        elif current_stop:
            if current_stop >= STOP_MIN_SECONDS:
                stop_count += 1
                stop_seconds += current_stop
            current_stop = 0.0
    if current_stop >= STOP_MIN_SECONDS:
        stop_count += 1
        stop_seconds += current_stop
    duration = max(0.0, (points[-1][0] - points[0][0]) / fps)
    displacement = math.hypot(points[-1][1] - points[0][1], points[-1][2] - points[0][2])
    return {
        "first_frame": points[0][0],
        "last_frame": points[-1][0],
        "travel_time_seconds": round(duration, 6),
        "path_length_m": round(path_length, 6),
        "mean_speed_mps": round(path_length / duration, 6) if duration > 0 else None,
        "directness": round(displacement / path_length, 6) if path_length > 0 else None,
        "slow_ratio": round(slow_seconds / duration, 6) if duration > 0 else None,
        "stop_count": stop_count,
        "stop_duration_seconds": round(stop_seconds, 6),
        "start_x": points[0][1],
        "start_y": points[0][2],
        "end_x": points[-1][1],
        "end_y": points[-1][2],
    }


def _reroute_extra_path(points: list[tuple[int, float, float]], reroute_time: float, fps: float) -> float | None:
    """Return observed post-reroute path minus the straight remaining distance."""
    if len(points) < 2 or fps <= 0:
        return None
    reroute_frame = reroute_time * fps
    start_index = next((index for index, point in enumerate(points) if point[0] >= reroute_frame), None)
    if start_index is None or start_index >= len(points) - 1:
        return None
    remaining = points[start_index:]
    actual = sum(
        math.hypot(right[1] - left[1], right[2] - left[2])
        for left, right in zip(remaining, remaining[1:])
    )
    direct = math.hypot(remaining[-1][1] - remaining[0][1], remaining[-1][2] - remaining[0][2])
    return round(max(0.0, actual - direct), 6)


def _density_metrics(trajectories: dict[int, list[tuple[int, float, float]]], fps: float) -> tuple[dict, dict[int, float]]:
    frames: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for agent_id, points in trajectories.items():
        for frame, x, y in points:
            frames[frame].append((agent_id, x, y))
    area = math.pi * DENSITY_RADIUS_M ** 2
    all_densities = []
    per_agent: dict[int, list[float]] = defaultdict(list)
    congested_frames = 0
    congestion_locations: Counter[str] = Counter()
    for frame_points in frames.values():
        frame_congested = False
        for agent_id, x, y in frame_points:
            neighbors = sum(
                1 for _, ox, oy in frame_points
                if math.hypot(x - ox, y - oy) <= DENSITY_RADIUS_M
            )
            density = neighbors / area
            all_densities.append(density)
            per_agent[agent_id].append(density)
            if density >= CONGESTION_DENSITY_PER_M2:
                frame_congested = True
                congestion_locations[f"{round(x / 2) * 2:g},{round(y / 2) * 2:g}"] += 1
        congested_frames += int(frame_congested)
    metrics = {
        "mean_local_density_per_m2": round(mean(all_densities), 6) if all_densities else None,
        "peak_local_density_per_m2": round(max(all_densities), 6) if all_densities else None,
        "congestion_duration_seconds": round(congested_frames / fps, 6) if fps > 0 else 0.0,
        "congestion_hotspots": dict(congestion_locations.most_common(20)),
    }
    return metrics, {agent_id: round(mean(values), 6) for agent_id, values in per_agent.items()}


def _mean(values) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return round(mean(cleaned), 6) if cleaned else None


def extract_run_metrics(manifest_path: pathlib.Path, regions_path: pathlib.Path | None = None) -> dict:
    verification = verify_manifest(manifest_path)
    if not verification["valid"]:
        raise ValueError(f"run failed completeness verification: {verification['errors']}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    trajectory_path = _artifact_path(manifest_path, artifacts["trajectory"])
    plans_path = _artifact_path(manifest_path, artifacts["plans"])
    events_path = _artifact_path(manifest_path, artifacts["events"])
    schema, fps, trajectories = load_trajectories(trajectory_path)
    plans_payload = json.loads(plans_path.read_text(encoding="utf-8"))
    planned = plans_payload.get("agents", [])
    events = load_events(events_path)
    regions_path = regions_path or PROJECT_ROOT / "data" / "map" / "localization_grid_regions.json"
    regions = load_regions(regions_path)

    event_by_type: dict[str, list[dict]] = defaultdict(list)
    sim_to_agent: dict[int, str] = {}
    for event in events:
        event_by_type[event.get("event_type", "unknown")].append(event)
        if event.get("simulation_agent_id") is not None and event.get("agent_id"):
            sim_to_agent[int(event["simulation_agent_id"])] = str(event["agent_id"])
    plan_by_agent = {str(item.get("agent_id")): item for item in planned}
    trajectory_by_agent = {
        sim_to_agent.get(sim_id, f"simulation_agent_{sim_id}"): points
        for sim_id, points in trajectories.items()
    }
    waits: dict[str, float] = defaultdict(float)
    wait_started: dict[tuple[Any, Any], float] = {}
    for event in events:
        agent_id = event.get("agent_id")
        key = (agent_id, event.get("region"))
        if event.get("event_type") == "wait_started":
            wait_started[key] = float(event.get("simulation_time", 0))
        elif event.get("event_type") == "wait_released" and key in wait_started:
            waits[str(agent_id)] += max(0.0, float(event.get("simulation_time", 0)) - wait_started.pop(key))

    completed_ids = {str(event.get("agent_id")) for event in event_by_type["agent_completed"]}
    ttl_ids = {str(event.get("agent_id")) for event in event_by_type["agent_ttl_removed"]}
    failed_ids = {str(event.get("agent_id")) for event in event_by_type["agent_spawn_failed"]}
    reroute_ids = {str(event.get("agent_id")) for event in event_by_type["agent_rerouted"]}
    reroute_times = {
        str(event.get("agent_id")): float(event.get("simulation_time", 0.0))
        for event in event_by_type["agent_rerouted"]
    }
    density, agent_density = _density_metrics(trajectories, fps)
    agent_rows: list[dict[str, Any]] = []
    region_visits: Counter[str] = Counter()
    region_dwell: dict[str, float] = defaultdict(float)
    transitions: Counter[tuple[str, str]] = Counter()
    trajectory_rows: list[dict[str, Any]] = []
    all_agent_ids = sorted(set(plan_by_agent) | set(trajectory_by_agent))
    for agent_id in all_agent_ids:
        plan = plan_by_agent.get(agent_id, {})
        points = trajectory_by_agent.get(agent_id, [])
        row: dict[str, Any] = {
            "run_id": manifest["run_id"],
            "scenario_name": manifest["scenario_name"],
            "seed": manifest["seed"],
            "agent_id": agent_id,
            "role": plan.get("role", plan.get("profile", {}).get("role", "unknown")),
            "profile_source": plan.get("profile", {}).get("source", "unknown"),
            "planner": plan.get("planner", plans_payload.get("metadata", {}).get("route_policy", "unknown")),
            "final_exit": plan.get("final_exit"),
            "fallback": bool(plan.get("fallback_reason")),
            "terminal_state": (
                "ttl_removed" if agent_id in ttl_ids else
                "failed" if agent_id in failed_ids else
                "completed" if agent_id in completed_ids else
                plan.get("terminal_state", "unfinished")
            ),
            "wait_seconds": round(waits.get(agent_id, 0.0), 6),
            "rerouted": agent_id in reroute_ids or bool(plan.get("rerouted_due_to_stuck")),
            "reroute_count": len(plan.get("congestion_detours", [])) + int(bool(plan.get("rerouted_due_to_stuck"))),
            "reroute_extra_path_m": _reroute_extra_path(points, reroute_times[agent_id], fps)
            if agent_id in reroute_times else None,
            "mean_local_density_per_m2": agent_density.get(next(
                (sim_id for sim_id, mapped in sim_to_agent.items() if mapped == agent_id), -1
            )),
            **_trajectory_metrics(points, fps),
        }
        previous_region = None
        for (frame, x, y), next_point in zip(points, points[1:] + [points[-1]] if points else []):
            region = _region_at(x, y, regions)
            trajectory_rows.append({
                "run_id": manifest["run_id"],
                "scenario_name": manifest["scenario_name"],
                "seed": manifest["seed"],
                "agent_id": agent_id,
                "frame": frame,
                "time_seconds": round(frame / fps, 6),
                "x_m": x,
                "y_m": y,
                "region": region,
            })
            elapsed = max(0.0, (next_point[0] - frame) / fps)
            region_dwell[region] += elapsed
            if region != previous_region:
                region_visits[region] += 1
                if previous_region is not None:
                    transitions[(previous_region, region)] += 1
                previous_region = region
        agent_rows.append(row)

    spawned = len(event_by_type["agent_spawned"]) or len(trajectory_by_agent)
    completed = len(completed_ids)
    ttl_removed = len(ttl_ids)
    failed = len(failed_ids)
    unfinished = max(0, spawned - completed - ttl_removed - failed)
    fallback_count = sum(bool(item.get("fallback_reason")) for item in planned)
    exit_distribution = Counter(item.get("final_exit", "unknown") for item in planned)
    rerouted = sum(bool(row["rerouted"]) for row in agent_rows)
    reroute_success = sum(bool(row["rerouted"]) and row["terminal_state"] == "completed" for row in agent_rows)
    exit_windows: Counter[str] = Counter()
    for event in event_by_type["agent_completed"]:
        agent_id = str(event.get("agent_id"))
        exit_name = plan_by_agent.get(agent_id, {}).get("final_exit", "unknown")
        window = int(float(event.get("simulation_time", 0)) // 10) * 10
        exit_windows[f"{exit_name}@{window}-{window + 10}s"] += 1

    scalar_metrics = {
        "completion_rate": round(completed / spawned, 6) if spawned else None,
        "ttl_removal_rate": round(ttl_removed / spawned, 6) if spawned else None,
        "failure_rate": round(failed / len(planned), 6) if planned else None,
        "fallback_rate": round(fallback_count / len(planned), 6) if planned else None,
        "mean_travel_time_seconds": _mean(row.get("travel_time_seconds") for row in agent_rows),
        "mean_path_length_m": _mean(row.get("path_length_m") for row in agent_rows),
        "mean_speed_mps": _mean(row.get("mean_speed_mps") for row in agent_rows),
        "mean_directness": _mean(row.get("directness") for row in agent_rows),
        "mean_slow_ratio": _mean(row.get("slow_ratio") for row in agent_rows),
        "mean_wait_seconds": _mean(row.get("wait_seconds") for row in agent_rows) or 0.0,
        "mean_stop_count": _mean(row.get("stop_count") for row in agent_rows),
        "mean_stop_duration_seconds": _mean(row.get("stop_duration_seconds") for row in agent_rows),
        "reroute_rate": round(rerouted / spawned, 6) if spawned else None,
        "reroute_success_rate": round(reroute_success / rerouted, 6) if rerouted else None,
        "mean_reroute_extra_path_m": _mean(row.get("reroute_extra_path_m") for row in agent_rows),
        "exit_entropy": entropy_from_counts(exit_distribution),
        "peak_exit_load_per_10s": max(exit_windows.values(), default=0),
        **{key: value for key, value in density.items() if key != "congestion_hotspots"},
    }
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for field in ("role", "profile_source", "terminal_state"):
        grouped[field] = {}
        values = sorted({str(row.get(field, "unknown")) for row in agent_rows})
        for value in values:
            subset = [row for row in agent_rows if str(row.get(field, "unknown")) == value]
            grouped[field][value] = {
                "agents": len(subset),
                "mean_travel_time_seconds": _mean(row.get("travel_time_seconds") for row in subset),
                "mean_path_length_m": _mean(row.get("path_length_m") for row in subset),
                "mean_speed_mps": _mean(row.get("mean_speed_mps") for row in subset),
                "completion_rate": round(
                    sum(row["terminal_state"] == "completed" for row in subset) / len(subset), 6
                ) if subset else None,
            }
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "scenario_name": manifest["scenario_name"],
        "seed": manifest["seed"],
        "matrix": manifest.get("matrix", {}),
        "generated_at": dt.datetime.now().astimezone().isoformat(),
        "source_manifest": str(manifest_path),
        "trajectory_schema": schema,
        "fps": fps,
        "planned_agents": len(planned),
        "spawned_agents": spawned,
        "completed_agents": completed,
        "ttl_removed_agents": ttl_removed,
        "failed_agents": failed,
        "unfinished_agents": unfinished,
        "metrics": scalar_metrics,
        "distributions": {
            "exit": dict(exit_distribution),
            "travel_time_seconds": histogram_counts(
                [float(row["travel_time_seconds"]) for row in agent_rows if row.get("travel_time_seconds") is not None], 10, 300
            ),
            "path_length_m": histogram_counts(
                [float(row["path_length_m"]) for row in agent_rows if row.get("path_length_m") is not None], 5, 100
            ),
            "mean_speed_mps": histogram_counts(
                [float(row["mean_speed_mps"]) for row in agent_rows if row.get("mean_speed_mps") is not None], 0.2, 2.4
            ),
            "slow_ratio": histogram_counts(
                [float(row["slow_ratio"]) for row in agent_rows if row.get("slow_ratio") is not None], 0.1, 1.0
            ),
            "region_visits": dict(region_visits),
            "region_dwell_seconds": {key: round(value, 6) for key, value in region_dwell.items()},
            "exit_load_windows": dict(exit_windows),
            "congestion_hotspots": density["congestion_hotspots"],
        },
        "region_transitions": [
            {"from_region": source, "to_region": target, "count": count}
            for (source, target), count in sorted(transitions.items())
        ],
        "grouped_metrics": grouped,
        "metric_definitions": METRIC_DEFINITIONS,
        "agent_rows": agent_rows,
        "trajectory_rows": trajectory_rows,
    }


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        _atomic_write(path, "")
        return
    fields = sorted({key for row in rows for key in row})
    from io import StringIO
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write(path, stream.getvalue())


def write_run_metrics(manifest_path: pathlib.Path, output_dir: pathlib.Path | None = None) -> dict:
    result = extract_run_metrics(manifest_path)
    output_dir = output_dir or manifest_path.parent
    agent_rows = result.pop("agent_rows")
    trajectory_rows = result.pop("trajectory_rows")
    transitions = result["region_transitions"]
    metrics_path = output_dir / "metrics.json"
    _atomic_write(metrics_path, json.dumps(result, indent=2, ensure_ascii=False))
    _write_csv(output_dir / "agent_metrics.csv", agent_rows)
    _write_csv(output_dir / "trajectory_points.csv", trajectory_rows)
    run_row = {
        "run_id": result["run_id"],
        "scenario_name": result["scenario_name"],
        "seed": result["seed"],
        "spawned_agents": result["spawned_agents"],
        "completed_agents": result["completed_agents"],
        "ttl_removed_agents": result["ttl_removed_agents"],
        "failed_agents": result["failed_agents"],
        "unfinished_agents": result["unfinished_agents"],
        **result["metrics"],
    }
    _write_csv(output_dir / "run_metrics.csv", [run_row])
    _write_csv(output_dir / "region_transitions.csv", transitions)
    return {"metrics": str(metrics_path), "agents": len(agent_rows), "run_id": result["run_id"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", help="Run directories, manifests, or a parent runs directory.")
    parser.add_argument("--output-root", type=pathlib.Path)
    args = parser.parse_args(argv)
    manifests = discover_manifests(args.paths)
    if not manifests:
        raise SystemExit("No per-run manifests found.")
    completed = 0
    for manifest in manifests:
        output = args.output_root / manifest.parent.name if args.output_root else None
        try:
            result = write_run_metrics(manifest, output)
        except ValueError as exc:
            print(f"SKIP {manifest}: {exc}")
            continue
        print(f"OK   {result['run_id']}: {result['agents']} agent row(s)")
        completed += 1
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
