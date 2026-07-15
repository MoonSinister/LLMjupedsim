#!/usr/bin/env python3
"""Run JuPedSim on the drawn map.

The simulation uses:
  - data/map/geometry.wkt as the walkable geometry
  - data/map/stages.json for manual exits and entrances, when present

Agents are spawned one by one over time, which works better for small
entrance polygons than placing the full crowd at once.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import jupedsim as jps
import shapely
from jupedsim_mall.geometry import map_io
from jupedsim_mall.planning.route_planner import (
    BaselineRoutePlanner,
    LLMPlannerConfig,
    LLMExitOnlyRoutePlanner,
    LLMSemanticRoutePlanner,
    ReplayRoutePlanner,
)
from jupedsim_mall.profiles.agent_model import create_agent_queue, homogenize_agent_profiles
from jupedsim_mall.profiles.llmob_adapter import normalize_wait_seconds
from jupedsim_mall.profiles.providers import build_profile_provider
from jupedsim_mall.simulation.congestion_control import update_congestion_controls as run_congestion_control
from jupedsim_mall.simulation.event_recorder import create_event_recorder
from jupedsim_mall.simulation.journey_builder import (
    build_llm_journeys as create_semantic_journeys,
    update_waiting_controls as run_waiting_controls,
)
from jupedsim_mall.simulation.lifecycle import (
    mark_spawn_failures,
    mark_unfinished_agents,
    record_completed_agents,
    update_agent_lifetimes as run_lifecycle_control,
)
from jupedsim_mall.simulation.movement import MovementConfig, create_movement_model as build_movement_model
from jupedsim_mall.simulation.spawn_scheduler import (
    next_ready_spawn_plan as select_ready_spawn_plan,
    spawn_one_agent as run_spawn,
)
from shapely.geometry import Point, Polygon
from shapely.ops import triangulate, unary_union


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def geometry_parts(geometry):
    return list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]


def largest_polygon(geometry):
    parts = [g for g in geometry_parts(geometry) if isinstance(g, Polygon) and not g.is_empty]
    return max(parts, key=lambda p: p.area) if parts else None


def convex_exit_polygon(exit_area):
    """JuPedSim exit stages must be convex polygons.

    Use the largest triangle fully covered by the drawn/clipped exit area.
    This keeps the exit inside the user-drawn region even when clipping by
    walls turns it into a concave polygon.
    """
    exit_area = largest_polygon(exit_area)
    if exit_area is None or exit_area.area <= 0:
        return None

    hull = exit_area.convex_hull
    if exit_area.covers(hull) or abs(hull.area - exit_area.area) <= max(1e-6, exit_area.area * 0.01):
        return hull

    triangles = [
        tri
        for tri in triangulate(exit_area)
        if isinstance(tri, Polygon) and tri.area > 1e-6 and exit_area.covers(tri)
    ]
    if triangles:
        return max(triangles, key=lambda tri: tri.area)

    point = exit_area.representative_point()
    minx, miny, maxx, maxy = exit_area.bounds
    size = min(maxx - minx, maxy - miny, 0.4)
    while size > 0.02:
        half = size / 2
        square = Polygon([
            (point.x - half, point.y - half),
            (point.x + half, point.y - half),
            (point.x + half, point.y + half),
            (point.x - half, point.y + half),
        ])
        if exit_area.covers(square):
            return square
        size *= 0.5

    return None


def keep_largest_connected_area(geometry):
    """JuPedSim requires one connected accessible area."""
    return map_io.keep_largest_connected_area(geometry)


def load_geometry(path):
    return map_io.load_geometry(path)


def load_manual_stages(geometry, stages_path=pathlib.Path("data/map/stages.json")):
    return map_io.load_manual_stages(geometry, stages_path)


def load_regions(geometry, regions_path=pathlib.Path("data/map/localization_grid_regions.json")):
    return map_io.load_regions(geometry, regions_path)


def find_exit_positions(geometry, n_exits=3, exit_size=1.0):
    shrunk = geometry.buffer(-1.0)
    if shrunk.is_empty:
        shrunk = geometry.buffer(-0.3)

    polys = [g for g in geometry_parts(shrunk) if isinstance(g, Polygon) and g.area > 2.0]
    if not polys:
        polys = [g for g in geometry_parts(geometry) if isinstance(g, Polygon)]
        exit_size = 0.3

    candidates = []
    half = exit_size / 2
    for poly in polys:
        minx, miny, maxx, maxy = poly.bounds
        tests = [
            ("left", minx + 1.5, (miny + maxy) / 2),
            ("right", maxx - 1.5, (miny + maxy) / 2),
            ("bottom", (minx + maxx) / 2, miny + 1.5),
            ("top", (minx + maxx) / 2, maxy - 1.5),
        ]
        for label, tx, ty in tests:
            point = Point(tx, ty)
            if not poly.contains(point):
                continue
            exit_poly = Polygon([
                (tx - half, ty - half),
                (tx + half, ty - half),
                (tx + half, ty + half),
                (tx - half, ty + half),
            ])
            if poly.contains(exit_poly):
                candidates.append((exit_poly, label, tx, ty))

    unique = []
    for exit_poly, label, cx, cy in candidates:
        if all(((cx - ux) ** 2 + (cy - uy) ** 2) ** 0.5 >= exit_size * 5 for _, _, ux, uy in unique):
            unique.append((exit_poly, label, cx, cy))

    return [(exit_poly, label) for exit_poly, label, _, _ in unique[:n_exits]]


def find_spawn_areas(geometry, exits, n_areas=2, area_size=3.0):
    spawn_areas = []
    for poly in [g for g in geometry_parts(geometry) if isinstance(g, Polygon)]:
        centroid = poly.centroid
        minx, miny, maxx, maxy = poly.bounds
        offsets = [
            (0, 0, "center"),
            (-(maxx - minx) * 0.15, (maxy - miny) * 0.15, "upper_left"),
            ((maxx - minx) * 0.15, -(maxy - miny) * 0.15, "lower_right"),
        ]
        for dx, dy, label in offsets:
            sx, sy = centroid.x + dx, centroid.y + dy
            spawn_poly = Polygon([
                (sx - area_size / 2, sy - area_size / 2),
                (sx + area_size / 2, sy - area_size / 2),
                (sx + area_size / 2, sy + area_size / 2),
                (sx - area_size / 2, sy + area_size / 2),
            ])
            inter = spawn_poly.intersection(poly)
            if inter.is_empty:
                continue
            for exit_poly, _ in exits:
                inter = inter.difference(exit_poly.buffer(0.5))
                if inter.is_empty:
                    break
            inter = largest_polygon(inter)
            if inter is not None and inter.area > area_size ** 2 * 0.2:
                spawn_areas.append((inter, label))
                break
    return spawn_areas[:n_areas]


def candidate_exits_for_spawn(spawn_area, exits):
    """Return exits that do not overlap this spawn area."""
    candidates = []
    for idx, (exit_area, _) in enumerate(exits):
        overlap = spawn_area.intersection(exit_area).area
        overlap_ratio = overlap / max(spawn_area.area, 1e-9)
        if overlap_ratio < 0.02:
            candidates.append(idx)

    if candidates:
        return candidates

    scored = []
    spawn_centroid = spawn_area.centroid
    for idx, (exit_area, _) in enumerate(exits):
        overlap = spawn_area.intersection(exit_area).area
        overlap_ratio = overlap / max(spawn_area.area, 1e-9)
        distance = spawn_centroid.distance(exit_area.centroid)
        scored.append((overlap_ratio, -distance, idx))
    scored.sort()
    return [idx for _, _, idx in scored[: max(1, min(3, len(scored)))]]


def clean_spawn_area_for_single_agent(spawn_area, exits):
    """Remove overlapping exit pieces only if enough spawn area remains."""
    overlapping = [
        exit_area.buffer(0.03)
        for exit_area, _ in exits
        if spawn_area.intersection(exit_area).area > 0
    ]
    if overlapping:
        cleaned = largest_polygon(spawn_area.difference(unary_union(overlapping)))
        if cleaned is not None and cleaned.area >= 0.02:
            return cleaned

    return spawn_area


def build_spawn_plans(spawns, exits, total_wanted):
    plans = []
    if not spawns:
        return plans

    base = total_wanted // len(spawns)
    remainder = total_wanted % len(spawns)
    for idx, (spawn_area, label) in enumerate(spawns):
        candidate_exit_indices = candidate_exits_for_spawn(spawn_area, exits)
        cleaned_spawn = clean_spawn_area_for_single_agent(spawn_area, exits)
        plans.append({
            "label": label,
            "polygon": cleaned_spawn,
            "candidate_exit_indices": candidate_exit_indices,
            "remaining": base + (1 if idx < remainder else 0),
            "spawned": 0,
            "failed_attempts": 0,
        })
    return plans


def validate_saved_routes(spawn_plans, exits, regions=None):
    exit_labels = {label for _, label in exits}
    region_names = {region["name"] for region in (regions or [])}
    report = {
        "total_agents": 0,
        "planned_agents": 0,
        "invalid_exit_indices": 0,
        "invalid_candidate_exit": 0,
        "invalid_final_exit_label": 0,
        "invalid_regions": 0,
        "invalid_wait_seconds": 0,
        "invalid_desired_speed": 0,
    }
    for plan in spawn_plans:
        candidate_indices = set(plan.get("candidate_exit_indices", []))
        for agent in plan.get("agent_queue", []):
            report["total_agents"] += 1
            route = agent.get("route")
            if route is None:
                continue
            report["planned_agents"] += 1
            exit_idx = route.get("exit_idx")
            if not isinstance(exit_idx, int) or exit_idx < 0 or exit_idx >= len(exits):
                report["invalid_exit_indices"] += 1
            elif exit_idx not in candidate_indices:
                report["invalid_candidate_exit"] += 1
            if isinstance(exit_idx, int) and 0 <= exit_idx < len(exits):
                final_exit = exits[exit_idx][1]
                if final_exit not in exit_labels:
                    report["invalid_final_exit_label"] += 1
            for region_name in route.get("regions", []):
                if region_names and region_name not in region_names:
                    report["invalid_regions"] += 1
            for activity in route.get("activities", []):
                region_name = activity.get("region")
                if region_names and region_name not in region_names:
                    report["invalid_regions"] += 1
                wait_seconds = activity.get("wait_seconds", 0)
                if not isinstance(wait_seconds, (int, float)) or wait_seconds < 0 or wait_seconds > 180:
                    report["invalid_wait_seconds"] += 1
            desired_speed = route.get("desired_speed_mps", 1.34)
            if not isinstance(desired_speed, (int, float)) or desired_speed < 0.7 or desired_speed > 1.9:
                report["invalid_desired_speed"] += 1
    report["valid"] = all(
        report[key] == 0
        for key in (
            "invalid_exit_indices",
            "invalid_candidate_exit",
            "invalid_final_exit_label",
            "invalid_regions",
            "invalid_wait_seconds",
            "invalid_desired_speed",
        )
    )
    return report


def save_llm_plan(spawn_plans, exits, output_path, metadata=None, regions=None):
    if not output_path:
        return

    plans = []
    for plan in spawn_plans:
        for agent in plan.get("agent_queue", []):
            route = agent.get("route")
            if route is None:
                continue
            plans.append({
                "agent_id": agent["agent_id"],
                "spawn": plan["label"],
                "spawn_order": agent["spawn_order"],
                "scheduled_iteration": agent["scheduled_iteration"],
                "spawn_iteration": agent.get("spawn_iteration"),
                "candidate_exits": [exits[idx][1] for idx in plan.get("candidate_exit_indices", [])],
                "profile": agent.get("profile", {}),
                "role": route.get("role", "visitor"),
                "subtype": route.get("subtype"),
                "intent": route.get("intent", ""),
                "regions": route.get("regions", []),
                "activities": route.get("activities", []),
                "desired_speed_mps": route.get("desired_speed_mps"),
                "exit_idx": route["exit_idx"],
                "final_exit": exits[route["exit_idx"]][1],
                "rerouted_due_to_stuck": agent.get("rerouted_due_to_stuck", False),
                "reroute_iteration": agent.get("reroute_iteration"),
                "reroute_exit": agent.get("reroute_exit"),
                "released_waiting_controls": agent.get("released_waiting_controls", 0),
                "congestion_detours": agent.get("congestion_detours", []),
                "removed_due_to_ttl": agent.get("removed_due_to_ttl", False),
                "ttl_removed_iteration": agent.get("ttl_removed_iteration"),
                "ttl_age_seconds": agent.get("ttl_age_seconds"),
                "ttl_limit_seconds": agent.get("ttl_limit_seconds"),
                "terminal_state": agent.get("terminal_state"),
                "completed_iteration": agent.get("completed_iteration"),
                "spawn_failed_iteration": agent.get("spawn_failed_iteration"),
            })

    if not plans:
        return

    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validation = validate_saved_routes(spawn_plans, exits, regions=regions)
    payload = json.dumps(
        {
            "schema_version": "1.0",
            "metadata": metadata or {},
            "validation": validation,
            "agents": plans,
        },
        ensure_ascii=False,
        indent=2,
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)
    print(f"  Plan saved: {path.resolve()} validation_valid={validation['valid']}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", "-g", default="data/map/geometry.wkt")
    parser.add_argument("--num-agents", "-n", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2026, help="Base random seed for reproducible experiments.")
    parser.add_argument(
        "--spawn-interval",
        type=int,
        default=60,
        help="Base iterations between agents from the same entrance.",
    )
    parser.add_argument(
        "--spawn-jitter",
        type=int,
        default=40,
        help="Random extra iterations added to each agent's scheduled spawn time.",
    )
    parser.add_argument("--max-iters", type=int, default=30000)
    parser.add_argument("--simulation-dt", type=float, default=0.01)
    parser.add_argument(
        "--movement-model",
        choices=["cfsv3", "cfsv2", "cfs", "avm"],
        default="cfsv3",
        help="Pedestrian movement model. Use avm when opposing flows deadlock frequently.",
    )
    parser.add_argument("--agent-radius", type=float, default=0.12)
    parser.add_argument("--agent-time-gap", type=float, default=0.7)
    parser.add_argument("--neighbor-repulsion-strength", type=float, default=8.0)
    parser.add_argument("--neighbor-repulsion-range", type=float, default=0.12)
    parser.add_argument("--geometry-repulsion-strength", type=float, default=4.0)
    parser.add_argument("--geometry-repulsion-range", type=float, default=0.03)
    parser.add_argument("--cfsv3-range-x-scale", type=float, default=20.0)
    parser.add_argument("--cfsv3-range-y-scale", type=float, default=8.0)
    parser.add_argument("--cfsv3-theta-max", type=float, default=1.35)
    parser.add_argument("--cfsv3-agent-buffer", type=float, default=0.0)
    parser.add_argument("--avm-wall-buffer-distance", type=float, default=0.08)
    parser.add_argument("--avm-anticipation-time", type=float, default=1.0)
    parser.add_argument("--avm-reaction-time", type=float, default=0.3)
    parser.add_argument(
        "--max-active-agents",
        type=int,
        default=90,
        help="Delay new spawns when too many agents are active in the scene.",
    )
    parser.add_argument(
        "--agent-max-lifetime-seconds",
        type=float,
        default=300.0,
        help="Remove an agent after this many simulated seconds; set <= 0 to disable.",
    )
    parser.add_argument("--disable-stuck-reroute", action="store_true")
    parser.add_argument("--congestion-check-interval", type=int, default=300)
    parser.add_argument("--congestion-grace-iterations", type=int, default=900)
    parser.add_argument("--stuck-distance-threshold", type=float, default=0.06)
    parser.add_argument("--stuck-checks-before-reroute", type=int, default=4)
    parser.add_argument("--congestion-release-radius", type=float, default=2.0)
    parser.add_argument(
        "--enable-congestion-detour",
        action="store_true",
        help="Enable an optional emergency side-step journey for persistent local deadlocks.",
    )
    parser.add_argument("--congestion-pair-radius", type=float, default=0.7)
    parser.add_argument("--congestion-detour-max-attempts", type=int, default=1)
    parser.add_argument("--congestion-detour-distance", type=float, default=1.0)
    parser.add_argument("--congestion-detour-wall-margin", type=float, default=0.08)
    parser.add_argument("--congestion-detour-wait-seconds", type=float, default=4.0)
    parser.add_argument("--output", default="outputs/trajectories/demo_map.sqlite", help="Output SQLite trajectory file.")
    parser.add_argument("--regions", default="data/map/localization_grid_regions.json")
    parser.add_argument(
        "--profile-source",
        choices=["mall", "llmob", "atc"],
        default="mall",
        help="Use built-in mall personas, LLMob pickle profiles, or ATC tracking profiles.",
    )
    parser.add_argument(
        "--homogeneous-profiles",
        action="store_true",
        help="Use one identical control profile for the no-profile-variation ablation.",
    )
    parser.add_argument(
        "--llmob-data-root",
        default=os.environ.get("LLMOB_DATA_ROOT", ""),
        help="Path to the LLMob data folder that contains 2019/2021/20192021.",
    )
    parser.add_argument(
        "--llmob-dataset",
        choices=["2019", "2021", "20192021"],
        default=os.environ.get("LLMOB_DATASET", "2019"),
        help="LLMob dataset split used for profile identification.",
    )
    parser.add_argument(
        "--llmob-max-persons",
        type=int,
        default=int(os.environ.get("LLMOB_MAX_PERSONS", "0")),
        help="Maximum LLMob person pickle files to identify; 0 means all.",
    )
    parser.add_argument(
        "--llmob-profile-cache",
        default=os.environ.get("LLMOB_PROFILE_CACHE", "outputs/profiles/demo_map_llmob_profiles.json"),
        help="Where to save identified LLMob indoor profiles; empty disables cache output.",
    )
    parser.add_argument(
        "--atc-raw-path",
        default=os.environ.get("ATC_RAW_PATH", "data/atc-20121114/atc-20121114.csv"),
        help="ATC raw or processed CSV file/directory used for trajectory-profile identification.",
    )
    parser.add_argument(
        "--atc-regions",
        default=os.environ.get("ATC_REGIONS", "data/map/localization_grid_regions.json"),
        help="ATC region JSON for mapping trajectory points to semantic areas; empty disables region mapping.",
    )
    parser.add_argument(
        "--atc-max-persons",
        type=int,
        default=int(os.environ.get("ATC_MAX_PERSONS", "200")),
        help="Maximum ATC person trajectories to aggregate for profile identification; 0 means all.",
    )
    parser.add_argument(
        "--atc-max-rows",
        type=int,
        default=int(os.environ.get("ATC_MAX_ROWS", "0")),
        help="Maximum ATC CSV rows to scan; 0 means all selected rows/files.",
    )
    parser.add_argument(
        "--atc-min-points",
        type=int,
        default=int(os.environ.get("ATC_MIN_POINTS", "300")),
        help="Minimum tracking points required before an ATC person trajectory can become a profile.",
    )
    parser.add_argument(
        "--atc-profile-cache",
        default=os.environ.get("ATC_PROFILE_CACHE", "outputs/profiles/demo_map_atc_profiles.json"),
        help="Where to save identified ATC indoor profiles; empty disables cache output.",
    )
    parser.add_argument(
        "--atc-partition",
        choices=["train", "tuning", "evaluation", "all"],
        default=os.environ.get("ATC_PARTITION", "train"),
        help="Stable person-level ATC split; profile experiments should use train.",
    )
    parser.add_argument("--llm-routing", action="store_true", help="Use local LLM to assign an exit to each agent.")
    parser.add_argument("--replay-plan", default="", help="Replay routes from a saved plan JSON without calling an LLM.")
    parser.add_argument(
        "--baseline-routing",
        choices=["random", "nearest"],
        default="random",
        help="Routing policy used when --llm-routing is disabled or no usable LLM routes are returned.",
    )
    parser.add_argument(
        "--llm-base-url",
        default=os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:8600/v1"),
        help="OpenAI-compatible base URL, for example http://HOST:8600/v1.",
    )
    parser.add_argument(
        "--llm-model",
        default=os.environ.get("LOCAL_LLM_MODEL", "qwen3.6-27b:q8"),
        help="Model name passed to the local LLM server.",
    )
    parser.add_argument("--llm-timeout", type=float, default=60.0)
    parser.add_argument("--llm-max-tokens", type=int, default=4096)
    parser.add_argument("--llm-temperature", type=float, default=0.1)
    parser.add_argument("--llm-max-retries", type=int, default=1)
    parser.add_argument("--llm-batch-size", type=int, default=20)
    parser.add_argument("--llm-max-regions-per-agent", type=int, default=3)
    parser.add_argument("--llm-plan-output", default="outputs/plans/demo_map_llm_plan.json")
    parser.add_argument("--llm-cache-dir", default="outputs/llm_cache")
    parser.add_argument("--llm-calls-output", default="outputs/runs/llm_calls.jsonl")
    parser.add_argument("--run-id", default="", help="Stable run identifier written to lifecycle events.")
    parser.add_argument("--event-output", default="", help="Optional JSONL lifecycle event output.")
    parser.add_argument("--region-target-margin", type=float, default=0.2)
    parser.add_argument(
        "--waypoint-distance",
        type=float,
        default=1.8,
        help="Waypoint arrival tolerance in meters; larger values reduce clustering at exact target points.",
    )
    parser.add_argument(
        "--disable-routing-waypoints",
        action="store_true",
        help="Do not insert RoutingEngine waypoints between semantic stages.",
    )
    parser.add_argument(
        "--disable-activity-waiting",
        action="store_true",
        help="Keep semantic activity targets but set all activity waiting time to zero.",
    )
    parser.add_argument("--routing-waypoint-distance", type=float, default=1.4)
    parser.add_argument("--routing-waypoint-max-per-leg", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    MovementConfig.from_args(args)
    event_recorder = create_event_recorder(
        args.event_output,
        run_id=args.run_id,
        simulation_dt=args.simulation_dt,
    )
    event_recorder.record("run_started", iteration=0, seed=args.seed, movement_model=args.movement_model)

    geometry = keep_largest_connected_area(load_geometry(args.geometry))
    manual_exits, manual_spawns = load_manual_stages(geometry)
    regions = load_regions(geometry, pathlib.Path(args.regions))
    print(f"Geometry area: {geometry.area:.0f} m2")

    exits = manual_exits or find_exit_positions(geometry, n_exits=3, exit_size=1.0)
    if not exits:
        print("No suitable exits found. Draw exits in src/manual_draw_geometry.py or expand the walkable area.")
        return

    print(f"\n{'Manual' if manual_exits else 'Automatic'} exits: {len(exits)}")
    for exit_area, label in exits:
        cx, cy = exit_area.centroid.x, exit_area.centroid.y
        print(f"  {label}: ({cx:.1f}, {cy:.1f}) [{'OK' if geometry.contains(exit_area) else 'PARTIAL'}]")

    spawns = manual_spawns or find_spawn_areas(geometry, exits, n_areas=2, area_size=3.0)
    if not spawns:
        print("No suitable entrances/spawn areas found. Draw entrances in src/manual_draw_geometry.py.")
        return

    print(f"\n{'Manual' if manual_spawns else 'Automatic'} spawn areas: {len(spawns)}")
    for spawn_area, label in spawns:
        print(f"  {label}: area {spawn_area.area:.2f} m2")

    trajectory_file = pathlib.Path(args.output)
    trajectory_file.parent.mkdir(parents=True, exist_ok=True)
    trajectory_writer = jps.SqliteTrajectoryWriter(output_file=trajectory_file)
    simulation = jps.Simulation(
        model=build_movement_model(args.movement_model),
        geometry=geometry,
        dt=args.simulation_dt,
        trajectory_writer=trajectory_writer,
    )
    print(
        f"Movement model: {args.movement_model}, radius={args.agent_radius}, "
        f"time_gap={args.agent_time_gap}, max_active={args.max_active_agents}"
    )

    exit_ids = []
    journey_ids = []
    for exit_area, _ in exits:
        exit_id = simulation.add_exit_stage(exit_area)
        exit_ids.append(exit_id)
        journey_ids.append(simulation.add_journey(jps.JourneyDescription([exit_id])))

    spawn_plans = build_spawn_plans(spawns, exits, args.num_agents)
    if not spawn_plans:
        print("No usable spawn plans.")
        return

    profile_data_path = ""
    profile_cache = ""
    if args.profile_source == "llmob":
        profile_data_path = args.llmob_data_root
        profile_cache = args.llmob_profile_cache
    elif args.profile_source == "atc":
        profile_data_path = args.atc_raw_path
        profile_cache = args.atc_profile_cache
    profile_result = build_profile_provider(
        args.profile_source,
        seed=args.seed,
        data_path=profile_data_path,
        regions_path=args.atc_regions,
        dataset=args.llmob_dataset,
        max_persons=args.llmob_max_persons if args.profile_source == "llmob" else args.atc_max_persons,
        max_rows=args.atc_max_rows,
        min_points=args.atc_min_points,
        cache_output=profile_cache,
        allow_fallback=True,
        atc_partition=args.atc_partition,
    )
    profile_provider = profile_result.provider
    profile_report = profile_result.report.to_json()
    print(
        f"\nProfile provider: requested={profile_report['requested_source']}, "
        f"active={profile_report['active_source']}, profiles={profile_report['profile_count']}, "
        f"cache_hit={profile_report['cache_hit']}"
    )
    if profile_report["fallback_reason"]:
        print(f"  profile fallback: {profile_report['fallback_reason']}")

    create_agent_queue(
        spawn_plans,
        spawn_interval=args.spawn_interval,
        spawn_jitter=args.spawn_jitter,
        seed=args.seed,
        profile_provider=profile_provider,
    )
    if args.homogeneous_profiles:
        homogenize_agent_profiles(spawn_plans)

    if args.replay_plan:
        planner = ReplayRoutePlanner(args.replay_plan)
        planning_result = planner.plan(spawn_plans, exits, regions)
        if planning_result.planned_agents != planning_result.total_agents:
            raise ValueError(
                f"replay plan covered {planning_result.planned_agents}/{planning_result.total_agents} agents "
                f"with {planning_result.validation_errors} validation error(s)"
            )
    elif args.llm_routing:
        llm_config = LLMPlannerConfig(
                base_url=args.llm_base_url,
                model=args.llm_model,
                timeout=args.llm_timeout,
                max_tokens=args.llm_max_tokens,
                batch_size=args.llm_batch_size,
                max_regions_per_agent=args.llm_max_regions_per_agent,
                temperature=args.llm_temperature,
                max_retries=args.llm_max_retries,
                cache_dir=args.llm_cache_dir,
                calls_output=args.llm_calls_output,
            )
        planner_type = LLMExitOnlyRoutePlanner if args.llm_max_regions_per_agent == 0 else LLMSemanticRoutePlanner
        planner = planner_type(llm_config, seed=args.seed)
        planning_result = planner.plan(spawn_plans, exits, regions)
        if not planning_result.applied:
            planning_result = BaselineRoutePlanner(args.baseline_routing, args.seed).plan(spawn_plans, exits, regions)
    else:
        planning_result = BaselineRoutePlanner(args.baseline_routing, args.seed).plan(spawn_plans, exits, regions)

    route_policy = planning_result.policy
    llm_routing_applied = route_policy in {"llm_semantic", "llm_exit_only"}
    replay_source_policy = planning_result.metadata.get("source_route_policy", "")
    semantic_routing_applied = route_policy in {"llm_semantic", "llm_exit_only"} or (
        route_policy == "replay" and replay_source_policy in {"llm_semantic", "llm_exit_only"}
    )

    waiting_controls = []
    if semantic_routing_applied:
        routing_engine = None
        if not args.disable_routing_waypoints:
            try:
                routing_engine = jps.RoutingEngine(geometry)
                print("  RoutingEngine waypoints: enabled")
            except RuntimeError as exc:
                print(f"  RoutingEngine waypoints disabled: {exc}")
        waiting_controls = create_semantic_journeys(
            simulation,
            spawn_plans,
            regions,
            exit_ids,
            exits,
            routing_engine,
            simulation_dt=args.simulation_dt,
            region_target_margin=args.region_target_margin,
            waypoint_distance=args.waypoint_distance,
            routing_waypoint_distance=args.routing_waypoint_distance,
            routing_waypoint_max_per_leg=args.routing_waypoint_max_per_leg,
            disable_activity_waiting=args.disable_activity_waiting,
        )
        save_llm_plan(spawn_plans, exits, args.llm_plan_output, regions=regions)
    elif args.llm_plan_output:
        save_llm_plan(
            spawn_plans,
            exits,
            args.llm_plan_output,
            metadata={
                "route_policy": route_policy,
                "seed": args.seed,
                "llm_routing": False,
                "baseline_routing": args.baseline_routing,
            },
            regions=regions,
        )

    print(f"\nPreparing to spawn {args.num_agents} agents over time")
    for plan in spawn_plans:
        route_label = "llm_exits" if llm_routing_applied and plan.get("planned_exit_indices") else f"{args.baseline_routing}_exits"
        route_indices = plan.get("planned_exit_indices") or plan["candidate_exit_indices"]
        print(
            f"  {plan['label']}: {plan['remaining']} people, "
            f"{route_label}={[exits[idx][1] for idx in route_indices]}, "
            f"spawn_area={plan['polygon'].area:.2f} m2"
        )
        for agent in plan.get("agent_queue", [])[:3]:
            route = agent.get("route")
            if route is None:
                continue
            via = " -> ".join(route.get("regions", [])) or "direct"
            print(f"    {agent['agent_id']} {route.get('role', 'visitor')}: {via} -> {exits[route['exit_idx']][1]}")

    print("Running simulation...")
    spawned_total = 0
    seed = args.seed * 1000
    last_active = simulation.agent_count()
    congestion_state = {}
    previous_active_ids = set()

    while simulation.iteration_count() < args.max_iters:
        waiting = sum(plan["remaining"] for plan in spawn_plans)

        if waiting > 0 and simulation.agent_count() < args.max_active_agents:
            for _ in range(len(spawn_plans)):
                if simulation.agent_count() >= args.max_active_agents:
                    break
                plan = select_ready_spawn_plan(spawn_plans, simulation.iteration_count())
                if plan is None:
                    break
                if plan["failed_attempts"] > 120:
                    print(f"  {plan['label']}: repeated spawn failures; skipping remaining {plan['remaining']} agents")
                    mark_spawn_failures(plan, simulation.iteration_count(), event_recorder)
                    plan["remaining"] = 0
                    break
                if run_spawn(simulation, plan, exit_ids, journey_ids, exits, seed, args, event_recorder):
                    spawned_total += 1
                    seed += 1
                    continue
                seed += 1
                break

        if waiting == 0 and simulation.agent_count() == 0:
            break

        previous_active_ids = {agent.id for agent in simulation.agents()}
        simulation.iterate()
        previous_active_ids = record_completed_agents(
            previous_active_ids, simulation, spawn_plans, event_recorder
        )
        run_waiting_controls(waiting_controls, simulation.iteration_count(), event_recorder)
        run_congestion_control(
            simulation,
            spawn_plans,
            geometry,
            exits,
            exit_ids,
            journey_ids,
            waiting_controls,
            congestion_state,
            args,
            event_recorder,
        )
        ttl_removed = run_lifecycle_control(simulation, spawn_plans, args, event_recorder)
        if ttl_removed:
            print(
                f"  ttl: removed {ttl_removed} expired agents at "
                f"iter {simulation.iteration_count()}"
            )

        if simulation.iteration_count() % 500 == 0:
            active = simulation.agent_count()
            waiting = sum(plan["remaining"] for plan in spawn_plans)
            if active != last_active or waiting > 0:
                print(
                    f"  iter {simulation.iteration_count():>6}, "
                    f"time {simulation.elapsed_time():>6.1f}s, "
                    f"active: {active}, spawned: {spawned_total}, waiting: {waiting}"
                )
                last_active = active

    trajectory_writer.close()
    mark_unfinished_agents(spawn_plans)
    if args.llm_plan_output:
        save_llm_plan(
            spawn_plans,
            exits,
            args.llm_plan_output,
            metadata={
                "route_policy": route_policy,
                "seed": args.seed,
                "llm_routing": llm_routing_applied,
                "baseline_routing": args.baseline_routing if not llm_routing_applied else None,
                "profile_provider": profile_report,
                "planning": planning_result.to_json(),
            },
            regions=regions,
        )

    print()
    print("=" * 50)
    print("Simulation complete")
    print(f"  iterations : {simulation.iteration_count()}")
    print(f"  duration   : {simulation.elapsed_time():.1f}s")
    print(f"  spawned    : {spawned_total}")
    print(f"  area       : {geometry.area:.0f} m2")
    print(f"  output     : {trajectory_file.resolve()}")
    print("=" * 50)
    print()
    print("Visualization:")
    print(f"  python src/demo_visualize.py {trajectory_file}")
    event_recorder.record(
        "run_finished",
        iteration=simulation.iteration_count(),
        spawned_agents=spawned_total,
        active_agents=simulation.agent_count(),
    )
    event_recorder.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
