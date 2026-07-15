"""Deterministic agent spawn queue scheduling."""

from __future__ import annotations

import random

from shapely.geometry import Point

from jupedsim_mall.simulation.movement import create_agent_parameters


def _largest_polygon(geometry):
    parts = list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]
    parts = [part for part in parts if not part.is_empty]
    return max(parts, key=lambda part: part.area) if parts else None


def sample_single_position(polygon, seed, max_attempts=200):
    rng = random.Random(seed)
    sample_polygon = polygon
    for margin in (0.08, 0.03):
        candidate = _largest_polygon(polygon.buffer(-margin))
        if candidate is not None and candidate.area > 0.01:
            sample_polygon = candidate
            break
    minx, miny, maxx, maxy = sample_polygon.bounds
    for _ in range(max_attempts):
        point = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if sample_polygon.contains(point):
            return (point.x, point.y)
    return None


def spawn_one_agent(simulation, plan, exit_ids, journey_ids, exits, seed, args, recorder=None):
    rng = random.Random(seed * 7919 + plan["spawned"])
    for offset in range(20):
        position = sample_single_position(plan["polygon"], seed + offset)
        if position is None:
            continue
        queue = plan.get("agent_queue", [])
        next_index = plan.get("next_agent_index", plan["spawned"])
        agent = queue[next_index] if next_index < len(queue) else None
        route = agent.get("route") if agent is not None else None
        if route is not None:
            exit_idx = route["exit_idx"]
            journey_id = route.get("journey_id", journey_ids[exit_idx])
            stage_id = route.get("stage_id", exit_ids[exit_idx])
            desired_speed = max(0.7, min(1.9, float(route.get("desired_speed_mps", 1.34))))
        else:
            exit_idx = rng.choice(plan["candidate_exit_indices"])
            journey_id, stage_id, desired_speed = journey_ids[exit_idx], exit_ids[exit_idx], 1.34
        try:
            sim_id = simulation.add_agent(create_agent_parameters(
                args.movement_model, journey_id=journey_id, stage_id=stage_id,
                position=position, desired_speed=desired_speed, args=args,
            ))
        except Exception:
            continue
        iteration = simulation.iteration_count()
        if agent is not None:
            agent["simulation_agent_id"] = sim_id
            agent["spawn_iteration"] = iteration
        plan["remaining"] -= 1
        plan["spawned"] += 1
        plan["next_agent_index"] = next_index + 1
        plan["failed_attempts"] = 0
        if recorder is not None:
            recorder.record("agent_spawned", iteration=iteration, agent_id=agent.get("agent_id") if agent else None,
                            simulation_agent_id=sim_id, entrance=plan["label"], exit=exits[exit_idx][1], position=position)
        if plan["spawned"] <= 3:
            if agent is not None and route is not None:
                via = " -> ".join(route.get("regions", [])) or "direct"
                print(f"  {plan['label']}: {agent['agent_id']} {route.get('role', 'visitor')} {via} -> {exits[exit_idx][1]}")
            else:
                print(f"  {plan['label']}: agent {plan['spawned']} -> {exits[exit_idx][1]}")
        return True
    plan["failed_attempts"] += 1
    if recorder is not None:
        recorder.record("spawn_attempt_failed", iteration=simulation.iteration_count(), entrance=plan["label"],
                        consecutive_failures=plan["failed_attempts"])
    return False


def next_ready_spawn_plan(spawn_plans, iteration):
    ready = []
    for plan in spawn_plans:
        if plan["remaining"] <= 0:
            continue
        queue = plan.get("agent_queue", [])
        next_index = plan.get("next_agent_index", plan["spawned"])
        if next_index >= len(queue):
            continue
        agent = queue[next_index]
        if agent["scheduled_iteration"] <= iteration:
            ready.append((agent["scheduled_iteration"], agent["agent_id"], plan))
    if not ready:
        return None
    ready.sort(key=lambda item: (item[0], item[1]))
    return ready[0][2]
