"""Shared data contracts for the JuPedSim mall simulation.

The current simulation still passes dictionaries through most code paths. These
dataclasses define the stable shape those dictionaries should converge to, and
provide small helpers for validating JSON-compatible payloads.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


AGENT_ROLES = ("commuter", "shopper", "staff", "visitor")
SCHEMA_VERSION = "1.0"
RUN_STATUSES = ("planned", "running", "completed", "failed", "interrupted")
AGENT_TERMINAL_STATES = ("completed", "ttl_removed", "failed", "unfinished")


@dataclass(frozen=True)
class SemanticRegion:
    name: str
    points_world: list[list[float]]
    description: str = ""
    category: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StageDefinition:
    name: str
    kind: str
    polygon: list[list[float]]
    candidate_exit_indices: list[int] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentProfile:
    profile_id: str
    role: str = "visitor"
    subtype: str | None = None
    attribute: str = ""
    motivation: str = ""
    preferences: list[str] = field(default_factory=list)
    personal_variation: str = ""
    desired_speed_mps: float = 1.34
    default_wait_seconds: int = 0
    source: str = "mall"
    history_hint: str = ""
    intent_hint: str = ""
    memory_hint: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RouteActivity:
    region: str
    action: str = "visit"
    wait_seconds: int = 0
    target_position: list[float] | None = None
    schema_version: str = SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentRoute:
    agent_id: str
    exit_idx: int
    regions: list[str] = field(default_factory=list)
    activities: list[RouteActivity] = field(default_factory=list)
    role: str = "visitor"
    subtype: str | None = None
    intent: str = ""
    desired_speed_mps: float = 1.34
    planner: str = "unknown"
    fallback_reason: str = ""
    schema_version: str = SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["activities"] = [activity.to_json() for activity in self.activities]
        return payload


@dataclass(frozen=True)
class ExperimentScenario:
    name: str
    description: str = ""
    enabled: bool = True
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpawnPlan:
    agent_id: str
    profile: AgentProfile
    route: AgentRoute
    entrance_idx: int
    spawn_time_seconds: float = 0.0
    seed: int | None = None
    schema_version: str = SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    scenario_name: str
    seed: int
    status: str = "planned"
    started_at: str | None = None
    finished_at: str | None = None
    returncode: int | None = None
    config_path: str = ""
    scenario_snapshot: dict[str, Any] = field(default_factory=dict)
    reproducibility: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    failure: dict[str, Any] | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status not in RUN_STATUSES:
            raise ValueError(f"invalid run status: {self.status}")

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunMetrics:
    run_id: str
    scenario_name: str
    seed: int
    spawned_agents: int = 0
    completed_agents: int = 0
    ttl_removed_agents: int = 0
    failed_agents: int = 0
    unfinished_agents: int = 0
    metrics: dict[str, float | int | None] = field(default_factory=dict)
    distributions: dict[str, dict[str, int | float]] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat())
    schema_version: str = SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def normalize_role(value: Any, default: str = "visitor") -> str:
    role = str(value or "").strip().lower()
    return role if role in AGENT_ROLES else default


def clamp_float(value: Any, default: float, min_value: float, max_value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(max_value, parsed))


def clamp_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(max_value, parsed))


def profile_from_mapping(data: dict[str, Any]) -> AgentProfile:
    return AgentProfile(
        profile_id=str(data.get("profile_id") or "profile_unknown"),
        role=normalize_role(data.get("role")),
        subtype=data.get("subtype"),
        attribute=str(data.get("attribute") or ""),
        motivation=str(data.get("motivation") or ""),
        preferences=[str(item) for item in data.get("preferences", []) if item],
        personal_variation=str(data.get("personal_variation") or ""),
        desired_speed_mps=clamp_float(data.get("desired_speed_mps"), 1.34, 0.8, 1.8),
        default_wait_seconds=clamp_int(data.get("default_wait_seconds"), 0, 0, 180),
        source=str(data.get("source") or "mall"),
        history_hint=str(data.get("history_hint") or ""),
        intent_hint=str(data.get("intent_hint") or ""),
        memory_hint=str(data.get("memory_hint") or ""),
        schema_version=str(data.get("schema_version") or SCHEMA_VERSION),
        metadata={key: value for key, value in data.items() if key not in {
            "profile_id",
            "role",
            "subtype",
            "attribute",
            "motivation",
            "preferences",
            "personal_variation",
            "desired_speed_mps",
            "default_wait_seconds",
            "source",
            "history_hint",
            "intent_hint",
            "memory_hint",
            "schema_version",
        }},
    )


def activity_from_mapping(data: dict[str, Any]) -> RouteActivity | None:
    region = str(data.get("region") or "").strip()
    if not region:
        return None
    target = data.get("target_position")
    if isinstance(target, tuple):
        target = list(target)
    return RouteActivity(
        region=region,
        action=str(data.get("action") or "visit"),
        wait_seconds=clamp_int(data.get("wait_seconds"), 0, 0, 180),
        target_position=target if isinstance(target, list) else None,
        schema_version=str(data.get("schema_version") or SCHEMA_VERSION),
    )


def scenario_from_mapping(data: dict[str, Any]) -> ExperimentScenario:
    return ExperimentScenario(
        name=str(data.get("name") or "").strip(),
        description=str(data.get("description") or "").strip(),
        enabled=bool(data.get("enabled", True)),
        args=[str(item) for item in data.get("args", [])],
        env={str(key): str(value) for key, value in data.get("env", {}).items()},
        schema_version=str(data.get("schema_version") or SCHEMA_VERSION),
    )


def route_from_mapping(data: dict[str, Any]) -> AgentRoute:
    activities = []
    for item in data.get("activities", []):
        if isinstance(item, dict):
            activity = activity_from_mapping(item)
            if activity is not None:
                activities.append(activity)
    return AgentRoute(
        agent_id=str(data.get("agent_id") or "agent_unknown"),
        exit_idx=clamp_int(data.get("exit_idx"), 0, 0, 1_000_000),
        regions=[str(item) for item in data.get("regions", []) if item],
        activities=activities,
        role=normalize_role(data.get("role")),
        subtype=data.get("subtype"),
        intent=str(data.get("intent") or ""),
        desired_speed_mps=clamp_float(data.get("desired_speed_mps"), 1.34, 0.8, 1.8),
        planner=str(data.get("planner") or "unknown"),
        fallback_reason=str(data.get("fallback_reason") or ""),
        schema_version=str(data.get("schema_version") or SCHEMA_VERSION),
    )


def migrate_payload(kind: str, data: dict[str, Any]) -> dict[str, Any]:
    """Normalize a legacy JSON object to the current top-level schema version."""
    if not isinstance(data, dict):
        raise TypeError("payload must be a mapping")
    migrated = dict(data)
    migrated["schema_version"] = SCHEMA_VERSION
    if kind == "scenario":
        return scenario_from_mapping(migrated).to_json()
    if kind == "profile":
        return profile_from_mapping(migrated).to_json()
    if kind == "route":
        return route_from_mapping(migrated).to_json()
    if kind in {"plan", "map_metadata", "manifest", "metrics"}:
        return migrated
    raise ValueError(f"unsupported payload kind: {kind}")
