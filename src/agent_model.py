"""Agent identity and route-plan helpers for map simulation."""

from __future__ import annotations

from llmob_adapter import build_agent_profile


DEFAULT_AGENT_ROLES = [
    {
        "name": "commuter",
        "description": "Commuter: tends to move efficiently through the hall toward transit, parking, or an exit.",
    },
    {
        "name": "shopper",
        "description": "Shopper: may visit shops, restaurants, or convenience areas before leaving.",
    },
    {
        "name": "staff",
        "description": "Staff: may head toward office elevators, service corridors, shops, or internal areas.",
    },
    {
        "name": "visitor",
        "description": "Visitor: may explore landmarks, shops, restaurants, stairs, or escalators before choosing an exit.",
    },
]


def create_agent_queue(
    spawn_plans,
    spawn_interval,
    spawn_jitter,
    seed=2026,
    profile_provider=None,
):
    """Attach stable per-agent identities to each spawn plan."""
    import random

    rng = random.Random(seed)
    profile_provider = profile_provider or build_agent_profile
    next_id = 1
    for plan in spawn_plans:
        agents = []
        for spawn_order in range(1, plan["remaining"] + 1):
            jitter = rng.randint(0, max(spawn_jitter, 0)) if spawn_jitter > 0 else 0
            agent_id = f"agent_{next_id:04d}"
            agents.append({
                "agent_id": agent_id,
                "spawn": plan["label"],
                "spawn_order": spawn_order,
                "scheduled_iteration": (spawn_order - 1) * max(spawn_interval, 1) + jitter,
                "profile": profile_provider(agent_id, plan["label"], spawn_order, seed=seed),
                "route": None,
            })
            next_id += 1
        agents.sort(key=lambda agent: (agent["scheduled_iteration"], agent["agent_id"]))
        plan["agent_queue"] = agents
        plan["next_agent_index"] = 0
    return next_id - 1


def planned_agents(spawn_plans):
    for plan in spawn_plans:
        yield from plan.get("agent_queue", [])


def set_agent_route(agent, route):
    agent["route"] = route


def fallback_route(agent, exit_idx, intent="fallback random exit"):
    return {
        "agent_id": agent["agent_id"],
        "exit_idx": exit_idx,
        "regions": [],
        "activities": [],
        "role": "visitor",
        "intent": intent,
        "desired_speed_mps": agent.get("profile", {}).get("desired_speed_mps", 1.34),
    }
