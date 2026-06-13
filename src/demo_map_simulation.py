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
import math
import os
import pathlib
import random
import sys
import urllib.error
import urllib.request

import jupedsim as jps
import shapely
from agent_model import create_agent_queue, fallback_route, planned_agents, set_agent_route
from llmob_training_adapter import build_atc_profile_sampler, build_llmob_profile_sampler
from llmob_adapter import normalize_wait_seconds
from llm_prompts import build_agent_routing_messages
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
    parts = [g for g in geometry_parts(geometry) if isinstance(g, Polygon) and not g.is_empty]
    if len(parts) <= 1:
        return geometry

    parts.sort(key=lambda p: p.area, reverse=True)
    largest = parts[0]
    dropped_area = sum(p.area for p in parts[1:])
    print(
        f"Note: JuPedSim requires one connected walkable area; "
        f"kept largest area {largest.area:.1f} m2 and ignored "
        f"{len(parts) - 1} smaller areas ({dropped_area:.1f} m2)."
    )
    return largest


def load_geometry(path):
    wkt_path = pathlib.Path(path)
    if wkt_path.exists():
        return shapely.from_wkt(wkt_path.read_text(encoding="utf-8"))

    py_path = pathlib.Path("data/map/geometry.py")
    if py_path.exists():
        sys.path.insert(0, str(py_path.parent))
        from geometry import get_geometry

        return get_geometry()

    raise FileNotFoundError("No geometry file. Run: python src/manual_draw_geometry.py and press 's'.")


def load_manual_stages(geometry, stages_path=pathlib.Path("data/map/stages.json")):
    if not stages_path.exists():
        return [], []

    data = json.loads(stages_path.read_text(encoding="utf-8"))
    exits = []
    spawns = []

    for i, item in enumerate(data.get("exits", [])):
        poly = Polygon(item["polygon"])
        clipped = convex_exit_polygon(poly.intersection(geometry))
        if clipped is None or clipped.area < 0.01:
            print(f"  skip exit {item.get('name', i)}: outside walkable area")
            continue
        exits.append((clipped, item.get("name", f"exit_{i}")))

    for i, item in enumerate(data.get("entrances", data.get("spawns", []))):
        poly = Polygon(item["polygon"])
        clipped = largest_polygon(poly.intersection(geometry))
        if clipped is None or clipped.area < 0.02:
            print(f"  skip entrance {item.get('name', i)}: outside walkable area or too small")
            continue
        spawns.append((clipped, item.get("name", f"entrance_{i}")))

    if exits or spawns:
        print(f"\nLoaded manual stages: {stages_path}")
        print(f"  exits: {len(exits)}")
        print(f"  entrances/spawns: {len(spawns)}")

    return exits, spawns


def load_regions(geometry, regions_path=pathlib.Path("data/map/localization_grid_regions.json")):
    if not regions_path.exists():
        return []

    data = json.loads(regions_path.read_text(encoding="utf-8"))
    regions = []
    for item in data.get("regions", []):
        points = item.get("points_world") or []
        if len(points) < 3:
            continue

        polygon = Polygon(points)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty:
            continue

        clipped = largest_polygon(polygon.intersection(geometry))
        region_geometry = clipped or polygon
        center = region_geometry.representative_point()
        if not geometry.contains(center):
            nearest = geometry.representative_point()
            center = nearest

        regions.append({
            "name": item.get("name", f"region_{len(regions) + 1}"),
            "description": item.get("description", ""),
            "color": item.get("color", ""),
            "polygon": region_geometry,
            "position": (center.x, center.y),
            "area": region_geometry.area,
        })

    if regions:
        print(f"\nLoaded semantic regions: {regions_path}")
        print(f"  regions: {len(regions)}")
    return regions


def sample_region_target(region, seed, min_wall_margin=0.15, max_attempts=120):
    """Pick a stable target inside a semantic region instead of its center."""
    rng = random.Random(seed)
    polygon = region["polygon"]
    sample_polygon = polygon
    for margin in (min_wall_margin, 0.08, 0.03):
        shrunk = polygon.buffer(-margin)
        candidate = largest_polygon(shrunk)
        if candidate is not None and candidate.area > 0.01:
            sample_polygon = candidate
            break

    minx, miny, maxx, maxy = sample_polygon.bounds
    for _ in range(max_attempts):
        point = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if sample_polygon.contains(point):
            return (point.x, point.y)

    point = sample_polygon.representative_point()
    return (point.x, point.y)


def add_routing_waypoint_stages(
    simulation,
    routing_engine,
    waypoint_stage_ids,
    agent_id,
    start_position,
    end_position,
    waypoint_distance,
    max_waypoints,
):
    if routing_engine is None:
        return []
    if not routing_engine.is_routable(start_position) or not routing_engine.is_routable(end_position):
        return []

    try:
        waypoints = routing_engine.compute_waypoints(start_position, end_position)
    except RuntimeError:
        return []

    stage_ids = []
    for waypoint in waypoints[1:-1]:
        if distance_between(waypoint, start_position) <= waypoint_distance:
            continue
        if distance_between(waypoint, end_position) <= waypoint_distance:
            continue
        cache_key = (agent_id, "routing", round(waypoint[0], 2), round(waypoint[1], 2))
        if cache_key not in waypoint_stage_ids:
            waypoint_stage_ids[cache_key] = simulation.add_waypoint_stage(
                waypoint,
                distance=waypoint_distance,
            )
        stage_ids.append(waypoint_stage_ids[cache_key])
        if len(stage_ids) >= max_waypoints:
            break
    return stage_ids


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


def extract_json_object(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM response does not contain a JSON object")
    return json.loads(text[start : end + 1])


def chat_completion(base_url, model, messages, timeout, max_tokens):
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        url = base_url
    else:
        url = f"{base_url}/chat/completions"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("LOCAL_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")

    result = json.loads(body)
    message = result["choices"][0]["message"]
    return message.get("content") or message.get("reasoning") or ""


def chunked(items, size):
    size = max(size, 1)
    for start in range(0, len(items), size):
        yield items[start : start + size]


def create_movement_model(model_name):
    if model_name == "cfs":
        return jps.CollisionFreeSpeedModel()
    if model_name == "cfsv2":
        return jps.CollisionFreeSpeedModelV2()
    if model_name == "cfsv3":
        return jps.CollisionFreeSpeedModelV3()
    if model_name == "avm":
        return jps.AnticipationVelocityModel()
    raise ValueError(f"Unsupported movement model: {model_name}")


def create_agent_parameters(model_name, *, journey_id, stage_id, position, desired_speed, args):
    common = {
        "journey_id": journey_id,
        "stage_id": stage_id,
        "position": position,
        "radius": args.agent_radius,
        "desired_speed": desired_speed,
        "time_gap": args.agent_time_gap,
    }
    if model_name == "cfs":
        return jps.CollisionFreeSpeedModelAgentParameters(**common)
    if model_name == "cfsv2":
        return jps.CollisionFreeSpeedModelV2AgentParameters(
            **common,
            strength_neighbor_repulsion=args.neighbor_repulsion_strength,
            range_neighbor_repulsion=args.neighbor_repulsion_range,
            strength_geometry_repulsion=args.geometry_repulsion_strength,
            range_geometry_repulsion=args.geometry_repulsion_range,
        )
    if model_name == "cfsv3":
        return jps.CollisionFreeSpeedModelV3AgentParameters(
            **common,
            strength_neighbor_repulsion=args.neighbor_repulsion_strength,
            range_neighbor_repulsion=args.neighbor_repulsion_range,
            strength_geometry_repulsion=args.geometry_repulsion_strength,
            range_geometry_repulsion=args.geometry_repulsion_range,
            range_x_scale=args.cfsv3_range_x_scale,
            range_y_scale=args.cfsv3_range_y_scale,
            theta_max_upper_bound=args.cfsv3_theta_max,
            agent_buffer=args.cfsv3_agent_buffer,
        )
    if model_name == "avm":
        return jps.AnticipationVelocityModelAgentParameters(
            **common,
            strength_neighbor_repulsion=args.neighbor_repulsion_strength,
            range_neighbor_repulsion=args.neighbor_repulsion_range,
            wall_buffer_distance=args.avm_wall_buffer_distance,
            anticipation_time=args.avm_anticipation_time,
            reaction_time=args.avm_reaction_time,
        )
    raise ValueError(f"Unsupported movement model: {model_name}")


def route_agents_with_llm(spawn_plans, exits, regions, args):
    if not args.llm_routing:
        return False

    exit_infos = []
    for idx, (exit_area, label) in enumerate(exits):
        centroid = exit_area.centroid
        exit_infos.append({
            "id": idx,
            "name": label,
            "centroid": [round(centroid.x, 2), round(centroid.y, 2)],
            "area_m2": round(exit_area.area, 2),
        })

    agent_context = {}
    all_agent_infos = []
    for plan in spawn_plans:
        centroid = plan["polygon"].centroid
        candidate_set = set(plan["candidate_exit_indices"])
        candidate_exits = [
            {
                "name": exits[idx][1],
                "distance_m": round(centroid.distance(exits[idx][0].centroid), 2),
            }
            for idx in plan["candidate_exit_indices"]
        ]
        for agent in plan.get("agent_queue", []):
            all_agent_infos.append({
                "agent_id": agent["agent_id"],
                "spawn": plan["label"],
                "spawn_order": agent["spawn_order"],
                "scheduled_iteration": agent["scheduled_iteration"],
                "profile": agent.get("profile", {}),
                "spawn_centroid": [round(centroid.x, 2), round(centroid.y, 2)],
                "spawn_area_m2": round(plan["polygon"].area, 2),
                "candidate_exits": candidate_exits,
            })
            agent_context[agent["agent_id"]] = {
                "agent": agent,
                "candidate_exit_indices": candidate_set,
            }

    region_infos = []
    for region in regions:
        x, y = region["position"]
        region_infos.append({
            "name": region["name"],
            "description": region["description"],
            "centroid": [round(x, 2), round(y, 2)],
            "area_m2": round(region["area"], 2),
        })

    exit_label_to_idx = {label: idx for idx, (_, label) in enumerate(exits)}
    region_names = {region["name"] for region in regions}
    applied = 0
    failed_batches = 0

    batches = list(chunked(all_agent_infos, args.llm_batch_size))
    print(
        f"\nRequesting LLM batched routing: {args.llm_base_url} "
        f"model={args.llm_model}, batches={len(batches)}, batch_size={args.llm_batch_size}"
    )
    for batch_index, batch in enumerate(batches, start=1):
        messages = build_agent_routing_messages(batch, region_infos, exit_infos)
        try:
            content = chat_completion(
                args.llm_base_url,
                args.llm_model,
                messages,
                args.llm_timeout,
                args.llm_max_tokens,
            )
            data = extract_json_object(content)
        except (OSError, urllib.error.URLError, json.JSONDecodeError, KeyError, ValueError) as exc:
            failed_batches += 1
            print(f"  batch {batch_index}/{len(batches)} failed; agents in this batch will use fallback: {exc}")
            continue

        batch_applied = 0
        for item in data.get("plans", []):
            agent_id = item.get("agent_id")
            context = agent_context.get(agent_id)
            if context is None:
                continue
            exit_name = item.get("final_exit") or item.get("exit")
            idx = exit_label_to_idx.get(exit_name)
            if idx not in context["candidate_exit_indices"]:
                continue

            profile = context["agent"].get("profile", {})
            activities = []
            raw_activities = item.get("activities")
            if not isinstance(raw_activities, list):
                raw_activities = [{"region": name} for name in item.get("regions", [])]
            for activity in raw_activities:
                if not isinstance(activity, dict):
                    continue
                region_name = activity.get("region")
                if region_name not in region_names:
                    continue
                activities.append({
                    "region": region_name,
                    "action": activity.get("action", "visit"),
                    "wait_seconds": normalize_wait_seconds(
                        activity.get("wait_seconds"),
                        profile.get("default_wait_seconds", 0),
                    ),
                })
                if len(activities) >= args.llm_max_regions_per_agent:
                    break
            set_agent_route(context["agent"], {
                "agent_id": agent_id,
                "exit_idx": idx,
                "regions": [activity["region"] for activity in activities],
                "activities": activities,
                "role": item.get("role", "visitor"),
                "subtype": item.get("subtype") or profile.get("subtype"),
                "intent": item.get("intent", ""),
                "desired_speed_mps": float(item.get("desired_speed_mps") or profile.get("desired_speed_mps", 1.34)),
            })
            applied += 1
            batch_applied += 1

        print(f"  batch {batch_index}/{len(batches)}: planned {batch_applied}/{len(batch)} agents")

    if applied == 0:
        print("  LLM returned no usable routes; falling back to random exits.")
        return False

    for plan in spawn_plans:
        rng = random.Random(f"llm-fallback-{plan['label']}")
        for agent in plan.get("agent_queue", []):
            if agent.get("route") is None:
                set_agent_route(
                    agent,
                    fallback_route(agent, rng.choice(plan["candidate_exit_indices"])),
                )
        plan["planned_routes"] = [agent["route"] for agent in plan.get("agent_queue", [])]
        plan["planned_exit_indices"] = [route["exit_idx"] for route in plan["planned_routes"]]

    print(f"  LLM routing applied: {applied}/{len(all_agent_infos)} agents, failed_batches={failed_batches}")
    for plan in spawn_plans:
        planned = plan.get("planned_exit_indices")
        if not planned:
            continue
        counts = {}
        for idx in planned:
            counts[exits[idx][1]] = counts.get(exits[idx][1], 0) + 1
        print(f"    {plan['label']}: {counts}")
        for agent in plan.get("agent_queue", [])[:3]:
            route = agent["route"]
            via = " -> ".join(route["regions"]) if route["regions"] else "direct"
            print(
                f"      {agent['agent_id']} {route['role']}: {via} -> {exits[route['exit_idx']][1]} "
                f"({route['intent']})"
            )
    return True


def build_llm_journeys(
    simulation,
    spawn_plans,
    regions,
    exit_ids,
    exits,
    routing_engine,
    simulation_dt,
    region_target_margin,
    waypoint_distance,
    routing_waypoint_distance,
    routing_waypoint_max_per_leg,
):
    region_by_name = {region["name"]: region for region in regions}
    waypoint_stage_ids = {}
    waiting_controls = []
    journey_cache = {}

    for plan in spawn_plans:
        spawn_point = plan["polygon"].representative_point()
        spawn_position = (spawn_point.x, spawn_point.y)
        for agent in plan.get("agent_queue", []):
            route = agent.get("route")
            if route is None:
                continue

            stage_ids = []
            current_position = spawn_position
            activities = route.get("activities") or [
                {"region": region_name, "action": "visit", "wait_seconds": 0}
                for region_name in route.get("regions", [])
            ]
            for activity in activities:
                region_name = activity.get("region")
                region = region_by_name.get(region_name)
                if region is None:
                    continue
                wait_seconds = normalize_wait_seconds(activity.get("wait_seconds"), 0)
                target_position = sample_region_target(
                    region,
                    seed=f"{agent['agent_id']}-{region_name}-{activity.get('action', 'visit')}",
                    min_wall_margin=region_target_margin,
                )
                activity["target_position"] = [round(target_position[0], 3), round(target_position[1], 3)]
                stage_ids.extend(add_routing_waypoint_stages(
                    simulation,
                    routing_engine,
                    waypoint_stage_ids,
                    agent["agent_id"],
                    current_position,
                    target_position,
                    routing_waypoint_distance,
                    routing_waypoint_max_per_leg,
                ))
                if wait_seconds > 0:
                    stage_id = simulation.add_waiting_set_stage([target_position])
                    waiting_stage = simulation.get_stage(stage_id)
                    waiting_stage.state = jps.WaitingSetState.ACTIVE
                    waiting_controls.append({
                        "agent_id": agent["agent_id"],
                        "region": region_name,
                        "action": activity.get("action", "visit"),
                        "target_position": target_position,
                        "stage": waiting_stage,
                        "wait_iterations": max(1, int(wait_seconds / simulation_dt)),
                        "started_iteration": None,
                        "released": False,
                    })
                    stage_ids.append(stage_id)
                else:
                    cache_key = (agent["agent_id"], region_name, activity.get("action", "visit"), "waypoint")
                    if cache_key not in waypoint_stage_ids:
                        waypoint_stage_ids[cache_key] = simulation.add_waypoint_stage(
                            target_position,
                            distance=waypoint_distance,
                        )
                    stage_ids.append(waypoint_stage_ids[cache_key])
                current_position = target_position

            for region_name in route.get("regions", []):
                if region_name in [activity.get("region") for activity in activities]:
                    continue
                region = region_by_name.get(region_name)
                if region is None:
                    continue
                target_position = sample_region_target(
                    region,
                    seed=f"{agent['agent_id']}-{region_name}-extra-waypoint",
                    min_wall_margin=region_target_margin,
                )
                stage_ids.extend(add_routing_waypoint_stages(
                    simulation,
                    routing_engine,
                    waypoint_stage_ids,
                    agent["agent_id"],
                    current_position,
                    target_position,
                    routing_waypoint_distance,
                    routing_waypoint_max_per_leg,
                ))
                cache_key = (agent["agent_id"], region_name, "extra-waypoint")
                if cache_key not in waypoint_stage_ids:
                    waypoint_stage_ids[cache_key] = simulation.add_waypoint_stage(
                        target_position,
                        distance=waypoint_distance,
                    )
                stage_ids.append(waypoint_stage_ids[cache_key])
                current_position = target_position

            exit_idx = route["exit_idx"]
            exit_stage_id = exit_ids[exit_idx]
            exit_centroid = exits[exit_idx][0].centroid
            exit_position = (exit_centroid.x, exit_centroid.y)
            stage_ids.extend(add_routing_waypoint_stages(
                simulation,
                routing_engine,
                waypoint_stage_ids,
                agent["agent_id"],
                current_position,
                exit_position,
                routing_waypoint_distance,
                routing_waypoint_max_per_leg,
            ))
            stage_ids.append(exit_stage_id)
            cache_key = tuple(stage_ids)
            if cache_key not in journey_cache:
                journey = jps.JourneyDescription(stage_ids)
                for current_stage, next_stage in zip(stage_ids, stage_ids[1:]):
                    journey.set_transition_for_stage(
                        current_stage,
                        jps.Transition.create_fixed_transition(next_stage),
                    )
                journey_cache[cache_key] = simulation.add_journey(journey)

            route["stage_id"] = stage_ids[0]
            route["journey_id"] = journey_cache[cache_key]

    if waypoint_stage_ids:
        print(f"  Created semantic/routing waypoint stages: {len(waypoint_stage_ids)}")
    if waiting_controls:
        print(f"  Created activity waiting stages: {len(waiting_controls)}")
    return waiting_controls




def update_waiting_controls(waiting_controls, iteration):
    for control in waiting_controls:
        if control["released"]:
            continue
        stage = control["stage"]
        if control["started_iteration"] is None and stage.count_waiting() > 0:
            control["started_iteration"] = iteration
        if control["started_iteration"] is None:
            continue
        if iteration - control["started_iteration"] >= control["wait_iterations"]:
            stage.state = jps.WaitingSetState.INACTIVE
            control["released"] = True


def agent_records_by_simulation_id(spawn_plans):
    records = {}
    for plan in spawn_plans:
        for agent in plan.get("agent_queue", []):
            simulation_agent_id = agent.get("simulation_agent_id")
            if simulation_agent_id is not None:
                records[simulation_agent_id] = (plan, agent)
    return records


def nearest_exit_index(position, candidate_exit_indices, exits):
    px, py = position
    scored = []
    for idx in candidate_exit_indices:
        centroid = exits[idx][0].centroid
        dist = ((px - centroid.x) ** 2 + (py - centroid.y) ** 2) ** 0.5
        scored.append((dist, idx))
    scored.sort()
    return scored[0][1]


def active_waiting_agent_ids(waiting_controls):
    waiting_ids = set()
    for control in waiting_controls:
        if not control["released"]:
            waiting_ids.update(control["stage"].waiting())
    return waiting_ids


def release_nearby_waiting_controls(waiting_controls, position, radius, iteration):
    released = 0
    px, py = position
    for control in waiting_controls:
        if control["released"]:
            continue
        waiting = control["stage"].waiting()
        if not waiting:
            continue
        sx, sy = control.get("target_position", position)
        if ((px - sx) ** 2 + (py - sy) ** 2) ** 0.5 <= radius:
            control["stage"].state = jps.WaitingSetState.INACTIVE
            control["released"] = True
            control["forced_release_iteration"] = iteration
            released += 1
    return released


def distance_between(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def find_congestion_detour_position(position, nearby_positions, geometry, args, seed):
    """Find a small side-step point for an agent caught in a local deadlock."""
    px, py = position
    if nearby_positions:
        cx = sum(p[0] for p in nearby_positions) / len(nearby_positions)
        cy = sum(p[1] for p in nearby_positions) / len(nearby_positions)
        base_dx = px - cx
        base_dy = py - cy
    else:
        rng = random.Random(seed)
        angle = rng.uniform(0, 6.283185307179586)
        base_dx = math.cos(angle)
        base_dy = math.sin(angle)

    length = (base_dx * base_dx + base_dy * base_dy) ** 0.5
    if length < 1e-6:
        rng = random.Random(seed)
        angle = rng.uniform(0, 6.283185307179586)
        base_dx = math.cos(angle)
        base_dy = math.sin(angle)
    else:
        base_dx /= length
        base_dy /= length

    allowed_geometry = geometry
    margin = args.agent_radius + args.congestion_detour_wall_margin
    if margin > 0:
        shrunk = geometry.buffer(-margin)
        if not shrunk.is_empty:
            allowed_geometry = shrunk

    directions = [
        (base_dx, base_dy),
        (-base_dy, base_dx),
        (base_dy, -base_dx),
        (base_dx * 0.7 - base_dy * 0.7, base_dy * 0.7 + base_dx * 0.7),
        (base_dx * 0.7 + base_dy * 0.7, base_dy * 0.7 - base_dx * 0.7),
    ]
    distances = [
        args.congestion_detour_distance,
        args.congestion_detour_distance * 0.7,
        args.congestion_detour_distance * 1.3,
    ]

    candidates = []
    for dist in distances:
        for dx, dy in directions:
            dlen = (dx * dx + dy * dy) ** 0.5
            if dlen < 1e-6:
                continue
            candidate = (px + dx / dlen * dist, py + dy / dlen * dist)
            point = Point(candidate)
            if not allowed_geometry.covers(point):
                continue
            min_neighbor_dist = min(
                [distance_between(candidate, other) for other in nearby_positions] or [999.0]
            )
            candidates.append((min_neighbor_dist, candidate))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def switch_agent_to_congestion_detour(
    simulation,
    sim_id,
    agent,
    candidate_exit_indices,
    position,
    nearby_positions,
    geometry,
    exits,
    exit_ids,
    journey_ids,
    waiting_controls,
    iteration,
    args,
):
    detour_position = find_congestion_detour_position(
        position,
        nearby_positions,
        geometry,
        args,
        seed=f"{agent['agent_id']}-{iteration}-detour",
    )
    if detour_position is None:
        return False

    target_exit_idx = nearest_exit_index(position, candidate_exit_indices, exits)
    stage_id = simulation.add_waiting_set_stage([detour_position])
    waiting_stage = simulation.get_stage(stage_id)
    waiting_stage.state = jps.WaitingSetState.ACTIVE
    wait_iterations = max(1, int(args.congestion_detour_wait_seconds / args.simulation_dt))
    waiting_controls.append({
        "agent_id": agent["agent_id"],
        "region": "congestion_detour",
        "action": "yield_to_opposing_flow",
        "target_position": detour_position,
        "stage": waiting_stage,
        "wait_iterations": wait_iterations,
        "started_iteration": None,
        "released": False,
    })

    journey = jps.JourneyDescription([stage_id, exit_ids[target_exit_idx]])
    journey.set_transition_for_stage(
        stage_id,
        jps.Transition.create_fixed_transition(exit_ids[target_exit_idx]),
    )
    detour_journey_id = simulation.add_journey(journey)
    simulation.switch_agent_journey(sim_id, detour_journey_id, stage_id)

    detours = agent.setdefault("congestion_detours", [])
    detours.append({
        "iteration": iteration,
        "target_position": [round(detour_position[0], 3), round(detour_position[1], 3)],
        "final_exit": exits[target_exit_idx][1],
    })
    agent["congestion_detour_count"] = len(detours)
    return True


def update_congestion_controls(
    simulation,
    spawn_plans,
    geometry,
    exits,
    exit_ids,
    journey_ids,
    waiting_controls,
    congestion_state,
    args,
):
    if args.disable_stuck_reroute:
        return
    iteration = simulation.iteration_count()
    if iteration % max(args.congestion_check_interval, 1) != 0:
        return

    records = agent_records_by_simulation_id(spawn_plans)
    waiting_ids = active_waiting_agent_ids(waiting_controls)
    positions = {sim_agent.id: sim_agent.position for sim_agent in simulation.agents()}
    active_ids = set()

    for sim_agent in simulation.agents():
        sim_id = sim_agent.id
        active_ids.add(sim_id)
        if sim_id in waiting_ids:
            congestion_state.pop(sim_id, None)
            continue
        plan_agent = records.get(sim_id)
        if plan_agent is None:
            continue
        plan, agent = plan_agent
        if iteration - agent.get("spawn_iteration", iteration) < args.congestion_grace_iterations:
            continue
        if agent.get("rerouted_due_to_stuck"):
            continue

        position = sim_agent.position
        state = congestion_state.get(sim_id)
        if state is None:
            congestion_state[sim_id] = {
                "position": position,
                "stuck_checks": 0,
            }
            continue

        last_position = state["position"]
        moved = ((position[0] - last_position[0]) ** 2 + (position[1] - last_position[1]) ** 2) ** 0.5
        if moved < args.stuck_distance_threshold:
            state["stuck_checks"] += 1
        else:
            state["stuck_checks"] = 0
            state["position"] = position
            continue

        if state["stuck_checks"] < args.stuck_checks_before_reroute:
            continue

        nearby_positions = [
            other_position
            for other_id, other_position in positions.items()
            if other_id != sim_id
            and distance_between(position, other_position) <= args.congestion_pair_radius
        ]
        if (
            args.enable_congestion_detour
            and nearby_positions
            and agent.get("congestion_detour_count", 0) < args.congestion_detour_max_attempts
        ):
            if switch_agent_to_congestion_detour(
                simulation,
                sim_id,
                agent,
                plan["candidate_exit_indices"],
                position,
                nearby_positions,
                geometry,
                exits,
                exit_ids,
                journey_ids,
                waiting_controls,
                iteration,
                args,
            ):
                state["stuck_checks"] = 0
                state["position"] = position
                print(
                    f"  congestion: {agent['agent_id']} yield detour, "
                    f"nearby_agents={len(nearby_positions)}"
                )
                continue

        released = release_nearby_waiting_controls(
            waiting_controls,
            position,
            args.congestion_release_radius,
            iteration,
        )
        target_exit_idx = nearest_exit_index(position, plan["candidate_exit_indices"], exits)
        simulation.switch_agent_journey(
            sim_id,
            journey_ids[target_exit_idx],
            exit_ids[target_exit_idx],
        )
        agent["rerouted_due_to_stuck"] = True
        agent["reroute_iteration"] = iteration
        agent["reroute_exit"] = exits[target_exit_idx][1]
        agent["released_waiting_controls"] = released
        print(
            f"  congestion: {agent['agent_id']} stuck, reroute -> "
            f"{exits[target_exit_idx][1]}, released_waiting={released}"
        )

    for sim_id in list(congestion_state):
        if sim_id not in active_ids:
            congestion_state.pop(sim_id, None)


def update_agent_lifetimes(simulation, spawn_plans, args):
    max_lifetime_seconds = args.agent_max_lifetime_seconds
    if max_lifetime_seconds <= 0:
        return 0

    max_lifetime_iterations = int(max_lifetime_seconds / args.simulation_dt)
    if max_lifetime_iterations <= 0:
        return 0

    active_ids = {agent.id for agent in simulation.agents()}
    iteration = simulation.iteration_count()
    removed = 0

    for plan in spawn_plans:
        for agent in plan.get("agent_queue", []):
            sim_id = agent.get("simulation_agent_id")
            if sim_id is None or sim_id not in active_ids:
                continue
            if agent.get("removed_due_to_ttl"):
                continue
            spawn_iteration = agent.get("spawn_iteration")
            if spawn_iteration is None:
                continue
            age_iterations = iteration - spawn_iteration
            if age_iterations < max_lifetime_iterations:
                continue

            if simulation.mark_agent_for_removal(sim_id):
                agent["removed_due_to_ttl"] = True
                agent["ttl_removed_iteration"] = iteration
                agent["ttl_age_seconds"] = round(age_iterations * args.simulation_dt, 2)
                agent["ttl_limit_seconds"] = max_lifetime_seconds
                removed += 1

    return removed


def save_llm_plan(spawn_plans, exits, output_path):
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
                "profile": agent.get("profile", {}),
                "role": route.get("role", "visitor"),
                "subtype": route.get("subtype"),
                "intent": route.get("intent", ""),
                "regions": route.get("regions", []),
                "activities": route.get("activities", []),
                "desired_speed_mps": route.get("desired_speed_mps"),
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
            })

    if not plans:
        return

    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"agents": plans}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  LLM plan saved: {path.resolve()}")


def sample_single_position(polygon, seed, max_attempts=200):
    rng = random.Random(seed)
    sample_polygon = polygon
    for margin in (0.08, 0.03):
        shrunk = polygon.buffer(-margin)
        candidate = largest_polygon(shrunk)
        if candidate is not None and candidate.area > 0.01:
            sample_polygon = candidate
            break

    minx, miny, maxx, maxy = sample_polygon.bounds
    for _ in range(max_attempts):
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        point = Point(x, y)
        if sample_polygon.contains(point):
            return (x, y)
    return None


def spawn_one_agent(simulation, plan, exit_ids, journey_ids, exits, seed, args):
    rng = random.Random(seed * 7919 + plan["spawned"])
    for offset in range(20):
        position = sample_single_position(plan["polygon"], seed + offset)
        if position is None:
            continue

        agent = None
        route = None
        agent_queue = plan.get("agent_queue", [])
        next_agent_index = plan.get("next_agent_index", plan["spawned"])
        if next_agent_index < len(agent_queue):
            agent = agent_queue[next_agent_index]
            route = agent.get("route")

        if route is not None:
            target_exit_idx = route["exit_idx"]
            journey_id = route.get("journey_id", journey_ids[target_exit_idx])
            stage_id = route.get("stage_id", exit_ids[target_exit_idx])
            desired_speed = max(0.7, min(1.9, float(route.get("desired_speed_mps", 1.34))))
        else:
            target_exit_idx = rng.choice(plan["candidate_exit_indices"])
            journey_id = journey_ids[target_exit_idx]
            stage_id = exit_ids[target_exit_idx]
            desired_speed = 1.34
        try:
            simulation_agent_id = simulation.add_agent(
                create_agent_parameters(
                    args.movement_model,
                    journey_id=journey_id,
                    stage_id=stage_id,
                    position=position,
                    desired_speed=desired_speed,
                    args=args,
                )
            )
        except Exception:
            continue

        if agent is not None:
            agent["simulation_agent_id"] = simulation_agent_id
            agent["spawn_iteration"] = simulation.iteration_count()
        plan["remaining"] -= 1
        plan["spawned"] += 1
        plan["next_agent_index"] = next_agent_index + 1
        plan["failed_attempts"] = 0
        if plan["spawned"] <= 3:
            if agent is not None and route is not None:
                via = " -> ".join(route.get("regions", [])) or "direct"
                print(
                    f"  {plan['label']}: {agent['agent_id']} "
                    f"{route.get('role', 'visitor')} {via} -> {exits[target_exit_idx][1]}"
                )
            else:
                print(f"  {plan['label']}: agent {plan['spawned']} -> {exits[target_exit_idx][1]}")
        return True

    plan["failed_attempts"] += 1
    return False


def next_ready_spawn_plan(spawn_plans, iteration):
    ready = []
    for plan in spawn_plans:
        if plan["remaining"] <= 0:
            continue
        agent_queue = plan.get("agent_queue", [])
        next_agent_index = plan.get("next_agent_index", plan["spawned"])
        if next_agent_index >= len(agent_queue):
            continue
        agent = agent_queue[next_agent_index]
        if agent["scheduled_iteration"] <= iteration:
            ready.append((agent["scheduled_iteration"], agent["agent_id"], plan))

    if not ready:
        return None
    ready.sort(key=lambda item: (item[0], item[1]))
    return ready[0][2]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", "-g", default="data/map/geometry.wkt")
    parser.add_argument("--num-agents", "-n", type=int, default=120)
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
        "--llmob-data-root",
        default=os.environ.get("LLMOB_DATA_ROOT", r"D:\AAAWorkSpace\code\LLMob\LLMob\data"),
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
        default=os.environ.get(
            "ATC_RAW_PATH",
            r"D:\AAAWorkSpace\file\毕业论文材料\ATC-map\data\raw\atc-tracking-part1",
        ),
        help="ATC raw or processed CSV file/directory used for trajectory-profile identification.",
    )
    parser.add_argument(
        "--atc-regions",
        default=os.environ.get(
            "ATC_REGIONS",
            r"D:\AAAWorkSpace\file\毕业论文材料\ATC-map\data\map\localization_grid_regions.json",
        ),
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
    parser.add_argument("--llm-routing", action="store_true", help="Use local LLM to assign an exit to each agent.")
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
    parser.add_argument("--llm-batch-size", type=int, default=20)
    parser.add_argument("--llm-max-regions-per-agent", type=int, default=3)
    parser.add_argument("--llm-plan-output", default="outputs/plans/demo_map_llm_plan.json")
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
    parser.add_argument("--routing-waypoint-distance", type=float, default=1.4)
    parser.add_argument("--routing-waypoint-max-per-leg", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()

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
    simulation = jps.Simulation(
        model=create_movement_model(args.movement_model),
        geometry=geometry,
        dt=args.simulation_dt,
        trajectory_writer=jps.SqliteTrajectoryWriter(output_file=trajectory_file),
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

    profile_provider = None
    if args.profile_source == "llmob":
        profile_provider, identified_profiles = build_llmob_profile_sampler(
            args.llmob_data_root,
            dataset=args.llmob_dataset,
            max_persons=args.llmob_max_persons,
            cache_output=args.llmob_profile_cache or None,
        )
        role_counts = {}
        for profile in identified_profiles:
            role = profile.get("role", "visitor")
            role_counts[role] = role_counts.get(role, 0) + 1
        print(
            f"\nLoaded LLMob profiles: dataset={args.llmob_dataset}, "
            f"persons={len(identified_profiles)}, roles={role_counts}"
        )
        if args.llmob_profile_cache:
            print(f"  LLMob profile cache: {pathlib.Path(args.llmob_profile_cache).resolve()}")
    elif args.profile_source == "atc":
        profile_provider, identified_profiles = build_atc_profile_sampler(
            args.atc_raw_path,
            regions_path=args.atc_regions or None,
            max_persons=args.atc_max_persons,
            max_rows=args.atc_max_rows,
            min_points=args.atc_min_points,
            cache_output=args.atc_profile_cache or None,
        )
        role_counts = {}
        for profile in identified_profiles:
            role = profile.get("role", "visitor")
            role_counts[role] = role_counts.get(role, 0) + 1
        print(
            f"\nLoaded ATC profiles: persons={len(identified_profiles)}, "
            f"roles={role_counts}"
        )
        print(f"  ATC source: {pathlib.Path(args.atc_raw_path).resolve()}")
        if args.atc_regions:
            print(f"  ATC regions: {pathlib.Path(args.atc_regions).resolve()}")
        if args.atc_profile_cache:
            print(f"  ATC profile cache: {pathlib.Path(args.atc_profile_cache).resolve()}")

    create_agent_queue(
        spawn_plans,
        spawn_interval=args.spawn_interval,
        spawn_jitter=args.spawn_jitter,
        profile_provider=profile_provider,
    )

    llm_routing_applied = route_agents_with_llm(spawn_plans, exits, regions, args)
    waiting_controls = []
    if llm_routing_applied:
        routing_engine = None
        if not args.disable_routing_waypoints:
            try:
                routing_engine = jps.RoutingEngine(geometry)
                print("  RoutingEngine waypoints: enabled")
            except RuntimeError as exc:
                print(f"  RoutingEngine waypoints disabled: {exc}")
        waiting_controls = build_llm_journeys(
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
        )
        save_llm_plan(spawn_plans, exits, args.llm_plan_output)

    print(f"\nPreparing to spawn {args.num_agents} agents over time")
    for plan in spawn_plans:
        route_label = "llm_exits" if llm_routing_applied and plan.get("planned_exit_indices") else "random_exits"
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
    seed = 1000
    last_active = simulation.agent_count()
    congestion_state = {}

    while simulation.iteration_count() < args.max_iters:
        waiting = sum(plan["remaining"] for plan in spawn_plans)

        if waiting > 0 and simulation.agent_count() < args.max_active_agents:
            for _ in range(len(spawn_plans)):
                if simulation.agent_count() >= args.max_active_agents:
                    break
                plan = next_ready_spawn_plan(spawn_plans, simulation.iteration_count())
                if plan is None:
                    break
                if plan["failed_attempts"] > 120:
                    print(f"  {plan['label']}: repeated spawn failures; skipping remaining {plan['remaining']} agents")
                    plan["remaining"] = 0
                    break
                if spawn_one_agent(simulation, plan, exit_ids, journey_ids, exits, seed, args):
                    spawned_total += 1
                    seed += 1
                    continue
                seed += 1
                break

        if waiting == 0 and simulation.agent_count() == 0:
            break

        simulation.iterate()
        update_waiting_controls(waiting_controls, simulation.iteration_count())
        update_congestion_controls(
            simulation,
            spawn_plans,
            geometry,
            exits,
            exit_ids,
            journey_ids,
            waiting_controls,
            congestion_state,
            args,
        )
        ttl_removed = update_agent_lifetimes(simulation, spawn_plans, args)
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

    simulation._writer.close()
    if llm_routing_applied:
        save_llm_plan(spawn_plans, exits, args.llm_plan_output)

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


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
