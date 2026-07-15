"""Agent completion and removal lifecycle tracking."""

from __future__ import annotations

from jupedsim_mall.simulation.event_recorder import TerminalState


def agent_records_by_simulation_id(spawn_plans):
    return {
        agent["simulation_agent_id"]: agent
        for plan in spawn_plans
        for agent in plan.get("agent_queue", [])
        if agent.get("simulation_agent_id") is not None
    }


def update_agent_lifetimes(simulation, spawn_plans, args, recorder=None):
    if args.agent_max_lifetime_seconds <= 0:
        return 0
    limit = int(args.agent_max_lifetime_seconds / args.simulation_dt)
    if limit <= 0:
        return 0
    active_ids = {agent.id for agent in simulation.agents()}
    iteration = simulation.iteration_count()
    removed = 0
    for agent in agent_records_by_simulation_id(spawn_plans).values():
        sim_id = agent["simulation_agent_id"]
        spawn_iteration = agent.get("spawn_iteration")
        if sim_id not in active_ids or agent.get("removed_due_to_ttl") or spawn_iteration is None:
            continue
        age = iteration - spawn_iteration
        if age < limit:
            continue
        if simulation.mark_agent_for_removal(sim_id):
            agent["removed_due_to_ttl"] = True
            agent["terminal_state"] = TerminalState.TTL_REMOVED.value
            agent["ttl_removed_iteration"] = iteration
            agent["ttl_age_seconds"] = round(age * args.simulation_dt, 2)
            agent["ttl_limit_seconds"] = args.agent_max_lifetime_seconds
            removed += 1
            if recorder is not None:
                recorder.record("agent_ttl_removed", iteration=iteration, agent_id=agent.get("agent_id"), simulation_agent_id=sim_id,
                                age_seconds=agent["ttl_age_seconds"], limit_seconds=args.agent_max_lifetime_seconds)
    return removed


def record_completed_agents(previous_active_ids, simulation, spawn_plans, recorder=None):
    active_ids = {agent.id for agent in simulation.agents()}
    records = agent_records_by_simulation_id(spawn_plans)
    iteration = simulation.iteration_count()
    for sim_id in previous_active_ids - active_ids:
        agent = records.get(sim_id)
        if agent is None or agent.get("terminal_state"):
            continue
        agent["terminal_state"] = TerminalState.COMPLETED.value
        agent["completed_iteration"] = iteration
        if recorder is not None:
            recorder.record("agent_completed", iteration=iteration, agent_id=agent.get("agent_id"), simulation_agent_id=sim_id)
    return active_ids


def mark_unfinished_agents(spawn_plans):
    for plan in spawn_plans:
        for agent in plan.get("agent_queue", []):
            if agent.get("simulation_agent_id") is not None and not agent.get("terminal_state"):
                agent["terminal_state"] = TerminalState.UNFINISHED.value


def mark_spawn_failures(plan, iteration, recorder=None):
    queue = plan.get("agent_queue", [])
    next_index = plan.get("next_agent_index", plan.get("spawned", 0))
    for agent in queue[next_index:]:
        if agent.get("simulation_agent_id") is not None:
            continue
        agent["terminal_state"] = TerminalState.SPAWN_FAILED.value
        agent["spawn_failed_iteration"] = iteration
        if recorder is not None:
            recorder.record(
                "agent_spawn_failed",
                iteration=iteration,
                agent_id=agent.get("agent_id"),
                entrance=plan.get("label"),
                reason="spawn_retry_limit",
            )
