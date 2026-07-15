"""Translate planned semantic routes into JuPedSim stages and journeys."""

from __future__ import annotations

import random

import jupedsim as jps
from shapely.geometry import Point

from jupedsim_mall.profiles.llmob_adapter import normalize_wait_seconds


def _largest_polygon(geometry):
    parts = list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]
    parts = [part for part in parts if not part.is_empty]
    return max(parts, key=lambda part: part.area) if parts else None


def _distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def sample_region_target(region, seed, min_wall_margin=0.15, max_attempts=120):
    rng = random.Random(seed)
    polygon = region["polygon"]
    sample_polygon = polygon
    for margin in (min_wall_margin, 0.08, 0.03):
        candidate = _largest_polygon(polygon.buffer(-margin))
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


def add_routing_waypoint_stages(simulation, routing_engine, stage_cache, agent_id, start_position,
                                end_position, waypoint_distance, max_waypoints):
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
        if _distance(waypoint, start_position) <= waypoint_distance or _distance(waypoint, end_position) <= waypoint_distance:
            continue
        key = (agent_id, "routing", round(waypoint[0], 2), round(waypoint[1], 2))
        if key not in stage_cache:
            stage_cache[key] = simulation.add_waypoint_stage(waypoint, distance=waypoint_distance)
        stage_ids.append(stage_cache[key])
        if len(stage_ids) >= max_waypoints:
            break
    return stage_ids


def build_llm_journeys(simulation, spawn_plans, regions, exit_ids, exits, routing_engine, simulation_dt,
                       region_target_margin, waypoint_distance, routing_waypoint_distance,
                       routing_waypoint_max_per_leg, disable_activity_waiting=False):
    region_by_name = {region["name"]: region for region in regions}
    stage_cache, waiting_controls, journey_cache = {}, [], {}
    for plan in spawn_plans:
        spawn_point = plan["polygon"].representative_point()
        spawn_position = (spawn_point.x, spawn_point.y)
        for agent in plan.get("agent_queue", []):
            route = agent.get("route")
            if route is None:
                continue
            stage_ids, current_position = [], spawn_position
            activities = route.get("activities") or [
                {"region": name, "action": "visit", "wait_seconds": 0} for name in route.get("regions", [])
            ]
            for activity in activities:
                region_name = activity.get("region")
                region = region_by_name.get(region_name)
                if region is None:
                    continue
                wait_seconds = 0 if disable_activity_waiting else normalize_wait_seconds(
                    activity.get("wait_seconds"), 0
                )
                target = sample_region_target(region, f"{agent['agent_id']}-{region_name}-{activity.get('action', 'visit')}", region_target_margin)
                activity["target_position"] = [round(target[0], 3), round(target[1], 3)]
                stage_ids.extend(add_routing_waypoint_stages(
                    simulation, routing_engine, stage_cache, agent["agent_id"], current_position, target,
                    routing_waypoint_distance, routing_waypoint_max_per_leg,
                ))
                if wait_seconds > 0:
                    stage_id = simulation.add_waiting_set_stage([target])
                    stage = simulation.get_stage(stage_id)
                    stage.state = jps.WaitingSetState.ACTIVE
                    waiting_controls.append({
                        "agent_id": agent["agent_id"], "region": region_name,
                        "action": activity.get("action", "visit"), "target_position": target,
                        "stage": stage, "wait_iterations": max(1, int(wait_seconds / simulation_dt)),
                        "started_iteration": None, "released": False,
                    })
                    stage_ids.append(stage_id)
                else:
                    key = (agent["agent_id"], region_name, activity.get("action", "visit"), "waypoint")
                    if key not in stage_cache:
                        stage_cache[key] = simulation.add_waypoint_stage(target, distance=waypoint_distance)
                    stage_ids.append(stage_cache[key])
                current_position = target
            activity_regions = {activity.get("region") for activity in activities}
            for region_name in route.get("regions", []):
                if region_name in activity_regions or region_name not in region_by_name:
                    continue
                target = sample_region_target(region_by_name[region_name], f"{agent['agent_id']}-{region_name}-extra-waypoint", region_target_margin)
                stage_ids.extend(add_routing_waypoint_stages(
                    simulation, routing_engine, stage_cache, agent["agent_id"], current_position, target,
                    routing_waypoint_distance, routing_waypoint_max_per_leg,
                ))
                key = (agent["agent_id"], region_name, "extra-waypoint")
                if key not in stage_cache:
                    stage_cache[key] = simulation.add_waypoint_stage(target, distance=waypoint_distance)
                stage_ids.append(stage_cache[key])
                current_position = target
            exit_idx = route["exit_idx"]
            centroid = exits[exit_idx][0].centroid
            exit_position = (centroid.x, centroid.y)
            stage_ids.extend(add_routing_waypoint_stages(
                simulation, routing_engine, stage_cache, agent["agent_id"], current_position, exit_position,
                routing_waypoint_distance, routing_waypoint_max_per_leg,
            ))
            stage_ids.append(exit_ids[exit_idx])
            key = tuple(stage_ids)
            if key not in journey_cache:
                journey = jps.JourneyDescription(stage_ids)
                for current_stage, next_stage in zip(stage_ids, stage_ids[1:]):
                    journey.set_transition_for_stage(current_stage, jps.Transition.create_fixed_transition(next_stage))
                journey_cache[key] = simulation.add_journey(journey)
            route["stage_id"], route["journey_id"] = stage_ids[0], journey_cache[key]
    if stage_cache:
        print(f"  Created semantic/routing waypoint stages: {len(stage_cache)}")
    if waiting_controls:
        print(f"  Created activity waiting stages: {len(waiting_controls)}")
    return waiting_controls


def update_waiting_controls(waiting_controls, iteration, recorder=None):
    for control in waiting_controls:
        if control["released"]:
            continue
        stage = control["stage"]
        if control["started_iteration"] is None and stage.count_waiting() > 0:
            control["started_iteration"] = iteration
            if recorder is not None:
                recorder.record("wait_started", iteration=iteration, agent_id=control.get("agent_id"),
                                region=control.get("region"), action=control.get("action"))
        if control["started_iteration"] is None:
            continue
        if iteration - control["started_iteration"] >= control["wait_iterations"]:
            stage.state = jps.WaitingSetState.INACTIVE
            control["released"] = True
            if recorder is not None:
                recorder.record("wait_released", iteration=iteration, agent_id=control.get("agent_id"),
                                region=control.get("region"), forced=False)
