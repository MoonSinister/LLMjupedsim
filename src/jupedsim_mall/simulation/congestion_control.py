"""Stuck-agent detection, waiting release and congestion rerouting."""

from __future__ import annotations

import math
import random

import jupedsim as jps
from shapely.geometry import Point


def _records(spawn_plans):
    result = {}
    for plan in spawn_plans:
        for agent in plan.get("agent_queue", []):
            if agent.get("simulation_agent_id") is not None:
                result[agent["simulation_agent_id"]] = (plan, agent)
    return result


def _distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _nearest_exit(position, candidates, exits):
    return min(candidates, key=lambda idx: _distance(position, (exits[idx][0].centroid.x, exits[idx][0].centroid.y)))


def _waiting_ids(controls):
    result = set()
    for control in controls:
        if not control["released"]:
            result.update(control["stage"].waiting())
    return result


def _release_nearby(controls, position, radius, iteration, recorder=None):
    released = 0
    for control in controls:
        if control["released"] or not control["stage"].waiting():
            continue
        if _distance(position, control.get("target_position", position)) <= radius:
            control["stage"].state = jps.WaitingSetState.INACTIVE
            control["released"] = True
            control["forced_release_iteration"] = iteration
            released += 1
            if recorder is not None:
                recorder.record("wait_released", iteration=iteration, agent_id=control.get("agent_id"),
                                region=control.get("region"), forced=True)
    return released


def _detour_position(position, nearby, geometry, args, seed):
    px, py = position
    if nearby:
        base_dx = px - sum(p[0] for p in nearby) / len(nearby)
        base_dy = py - sum(p[1] for p in nearby) / len(nearby)
    else:
        angle = random.Random(seed).uniform(0, 2 * math.pi)
        base_dx, base_dy = math.cos(angle), math.sin(angle)
    length = math.hypot(base_dx, base_dy)
    if length < 1e-6:
        angle = random.Random(seed).uniform(0, 2 * math.pi)
        base_dx, base_dy, length = math.cos(angle), math.sin(angle), 1.0
    base_dx, base_dy = base_dx / length, base_dy / length
    allowed = geometry
    margin = args.agent_radius + args.congestion_detour_wall_margin
    if margin > 0 and not geometry.buffer(-margin).is_empty:
        allowed = geometry.buffer(-margin)
    directions = [(base_dx, base_dy), (-base_dy, base_dx), (base_dy, -base_dx),
                  (base_dx * .7 - base_dy * .7, base_dy * .7 + base_dx * .7),
                  (base_dx * .7 + base_dy * .7, base_dy * .7 - base_dx * .7)]
    candidates = []
    for distance in (args.congestion_detour_distance, args.congestion_detour_distance * .7, args.congestion_detour_distance * 1.3):
        for dx, dy in directions:
            dlen = math.hypot(dx, dy)
            candidate = (px + dx / dlen * distance, py + dy / dlen * distance)
            if allowed.covers(Point(candidate)):
                candidates.append((min([_distance(candidate, other) for other in nearby] or [999.0]), candidate))
    return max(candidates)[1] if candidates else None


def _switch_to_detour(simulation, sim_id, agent, candidates, position, nearby, geometry, exits, exit_ids,
                      waiting_controls, iteration, args, recorder=None):
    target = _detour_position(position, nearby, geometry, args, f"{agent['agent_id']}-{iteration}-detour")
    if target is None:
        return False
    exit_idx = _nearest_exit(position, candidates, exits)
    stage_id = simulation.add_waiting_set_stage([target])
    stage = simulation.get_stage(stage_id)
    stage.state = jps.WaitingSetState.ACTIVE
    waiting_controls.append({
        "agent_id": agent["agent_id"], "region": "congestion_detour", "action": "yield_to_opposing_flow",
        "target_position": target, "stage": stage,
        "wait_iterations": max(1, int(args.congestion_detour_wait_seconds / args.simulation_dt)),
        "started_iteration": None, "released": False,
    })
    journey = jps.JourneyDescription([stage_id, exit_ids[exit_idx]])
    journey.set_transition_for_stage(stage_id, jps.Transition.create_fixed_transition(exit_ids[exit_idx]))
    simulation.switch_agent_journey(sim_id, simulation.add_journey(journey), stage_id)
    detours = agent.setdefault("congestion_detours", [])
    detours.append({"iteration": iteration, "target_position": [round(target[0], 3), round(target[1], 3)], "final_exit": exits[exit_idx][1]})
    agent["congestion_detour_count"] = len(detours)
    if recorder is not None:
        recorder.record("congestion_detour", iteration=iteration, agent_id=agent["agent_id"],
                        simulation_agent_id=sim_id, target_position=target, exit=exits[exit_idx][1])
    return True


def update_congestion_controls(simulation, spawn_plans, geometry, exits, exit_ids, journey_ids,
                               waiting_controls, congestion_state, args, recorder=None):
    if args.disable_stuck_reroute:
        return
    iteration = simulation.iteration_count()
    if iteration % max(args.congestion_check_interval, 1) != 0:
        return
    records = _records(spawn_plans)
    waiting_ids = _waiting_ids(waiting_controls)
    positions = {agent.id: agent.position for agent in simulation.agents()}
    active_ids = set()
    for sim_agent in simulation.agents():
        sim_id, position = sim_agent.id, sim_agent.position
        active_ids.add(sim_id)
        if sim_id in waiting_ids:
            congestion_state.pop(sim_id, None)
            continue
        pair = records.get(sim_id)
        if pair is None:
            continue
        plan, agent = pair
        if iteration - agent.get("spawn_iteration", iteration) < args.congestion_grace_iterations or agent.get("rerouted_due_to_stuck"):
            continue
        state = congestion_state.get(sim_id)
        if state is None:
            congestion_state[sim_id] = {"position": position, "stuck_checks": 0}
            continue
        moved = _distance(position, state["position"])
        if moved < args.stuck_distance_threshold:
            state["stuck_checks"] += 1
        else:
            state.update(position=position, stuck_checks=0)
            continue
        if state["stuck_checks"] < args.stuck_checks_before_reroute:
            continue
        nearby = [other for other_id, other in positions.items() if other_id != sim_id and _distance(position, other) <= args.congestion_pair_radius]
        if args.enable_congestion_detour and nearby and agent.get("congestion_detour_count", 0) < args.congestion_detour_max_attempts:
            if _switch_to_detour(simulation, sim_id, agent, plan["candidate_exit_indices"], position, nearby,
                                  geometry, exits, exit_ids, waiting_controls, iteration, args, recorder):
                state.update(position=position, stuck_checks=0)
                print(f"  congestion: {agent['agent_id']} yield detour, nearby_agents={len(nearby)}")
                continue
        released = _release_nearby(waiting_controls, position, args.congestion_release_radius, iteration, recorder)
        exit_idx = _nearest_exit(position, plan["candidate_exit_indices"], exits)
        simulation.switch_agent_journey(sim_id, journey_ids[exit_idx], exit_ids[exit_idx])
        agent.update(rerouted_due_to_stuck=True, reroute_iteration=iteration,
                     reroute_exit=exits[exit_idx][1], released_waiting_controls=released)
        if recorder is not None:
            recorder.record("agent_rerouted", iteration=iteration, agent_id=agent["agent_id"],
                            simulation_agent_id=sim_id, exit=exits[exit_idx][1], released_waiting_controls=released)
        print(f"  congestion: {agent['agent_id']} stuck, reroute -> {exits[exit_idx][1]}, released_waiting={released}")
    for sim_id in list(congestion_state):
        if sim_id not in active_ids:
            congestion_state.pop(sim_id, None)
