"""Unified baseline, LLM, and replay route planners."""

from __future__ import annotations

import hashlib
import json
import os
import random
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from jupedsim_mall.experiments.sim_schema import route_from_mapping
from jupedsim_mall.planning.llm_prompts import PROMPT_VERSION, build_agent_routing_messages, prompt_hash
from jupedsim_mall.profiles.agent_model import fallback_route, set_agent_route
from jupedsim_mall.profiles.llmob_adapter import normalize_wait_seconds
from jupedsim_mall.project import SCHEMA_DIR


LLM_RESPONSE_SCHEMA = json.loads(
    (SCHEMA_DIR / "llm_route_response.schema.json").read_text(encoding="utf-8")
)
LLM_RESPONSE_VALIDATOR = Draft202012Validator(LLM_RESPONSE_SCHEMA)


@dataclass
class PlanningResult:
    policy: str
    total_agents: int
    planned_agents: int
    failed_batches: int = 0
    fallback_agents: int = 0
    validation_errors: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def applied(self) -> bool:
        return self.planned_agents > 0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class RoutePlanner(ABC):
    policy = "unknown"

    @abstractmethod
    def plan(self, spawn_plans: list[dict], exits: list, regions: list[dict]) -> PlanningResult:
        raise NotImplementedError


def _finalize_spawn_plans(spawn_plans: list[dict]) -> None:
    for spawn_plan in spawn_plans:
        routes = [agent["route"] for agent in spawn_plan.get("agent_queue", []) if agent.get("route") is not None]
        spawn_plan["planned_routes"] = routes
        spawn_plan["planned_exit_indices"] = [route["exit_idx"] for route in routes]


def validate_route(route: dict[str, Any], candidate_exit_indices: set[int], exit_count: int, region_names: set[str]) -> list[str]:
    errors = []
    exit_idx = route.get("exit_idx")
    if not isinstance(exit_idx, int) or not 0 <= exit_idx < exit_count:
        errors.append("invalid exit_idx")
    elif exit_idx not in candidate_exit_indices:
        errors.append("exit_idx is not a candidate for the spawn")
    for region in route.get("regions", []):
        if region not in region_names:
            errors.append(f"unknown region: {region}")
    for activity in route.get("activities", []):
        if activity.get("region") not in region_names:
            errors.append(f"unknown activity region: {activity.get('region')}")
        wait = activity.get("wait_seconds", 0)
        if not isinstance(wait, (int, float)) or not 0 <= wait <= 180:
            errors.append("activity wait_seconds must be between 0 and 180")
    speed = route.get("desired_speed_mps", 1.34)
    if not isinstance(speed, (int, float)) or not 0.8 <= speed <= 1.8:
        errors.append("desired_speed_mps must be between 0.8 and 1.8")
    return errors


class BaselineRoutePlanner(RoutePlanner):
    def __init__(self, policy: str = "random", seed: int = 2026):
        if policy not in {"random", "nearest"}:
            raise ValueError(f"unsupported baseline policy: {policy}")
        self.baseline_policy = policy
        self.policy = f"baseline_{policy}"
        self.seed = seed

    def plan(self, spawn_plans: list[dict], exits: list, regions: list[dict]) -> PlanningResult:
        applied = 0
        for spawn_plan in spawn_plans:
            rng = random.Random(f"baseline-{self.baseline_policy}-{self.seed}-{spawn_plan['label']}")
            spawn_centroid = spawn_plan["polygon"].centroid
            for agent in spawn_plan.get("agent_queue", []):
                candidates = spawn_plan["candidate_exit_indices"]
                if self.baseline_policy == "nearest":
                    exit_idx = min(candidates, key=lambda idx: spawn_centroid.distance(exits[idx][0].centroid))
                    intent = "baseline nearest candidate exit"
                else:
                    exit_idx = rng.choice(candidates)
                    intent = "baseline random candidate exit"
                route = fallback_route(agent, exit_idx, intent=intent)
                profile = agent.get("profile", {})
                route.update({
                    "role": profile.get("role", route["role"]),
                    "subtype": profile.get("subtype"),
                    "planner": self.policy,
                    "fallback_reason": "",
                    "schema_version": "1.0",
                })
                set_agent_route(agent, route)
                applied += 1
        _finalize_spawn_plans(spawn_plans)
        print(f"  Baseline routing applied: policy={self.baseline_policy}, agents={applied}")
        return PlanningResult(self.policy, applied, applied)


class ReplayRoutePlanner(RoutePlanner):
    policy = "replay"

    def __init__(self, plan_path: str | Path):
        self.plan_path = Path(plan_path)

    def plan(self, spawn_plans: list[dict], exits: list, regions: list[dict]) -> PlanningResult:
        payload = json.loads(self.plan_path.read_text(encoding="utf-8"))
        source_route_policy = str(payload.get("metadata", {}).get("route_policy") or "unknown")
        saved = {str(item.get("agent_id")): item for item in payload.get("agents", [])}
        region_names = {region["name"] for region in regions}
        planned = 0
        validation_errors = 0
        total = 0
        for spawn_plan in spawn_plans:
            candidates = set(spawn_plan["candidate_exit_indices"])
            for agent in spawn_plan.get("agent_queue", []):
                total += 1
                item = saved.get(agent["agent_id"])
                if item is None:
                    continue
                route = route_from_mapping({**item, "planner": "replay"}).to_json()
                errors = validate_route(route, candidates, len(exits), region_names)
                if errors:
                    validation_errors += len(errors)
                    continue
                set_agent_route(agent, route)
                planned += 1
        _finalize_spawn_plans(spawn_plans)
        return PlanningResult(
            self.policy,
            total,
            planned,
            validation_errors=validation_errors,
            metadata={
                "replay_plan": str(self.plan_path),
                "source_route_policy": source_route_policy,
            },
        )


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM response does not contain a JSON object")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM response root must be an object")
    return payload


def validate_llm_response(payload: dict[str, Any]) -> None:
    errors = sorted(LLM_RESPONSE_VALIDATOR.iter_errors(payload), key=lambda error: list(error.absolute_path))
    if errors:
        error = errors[0]
        field = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise ValueError(f"LLM response schema {field}: {error.message}")


@dataclass(frozen=True)
class LLMPlannerConfig:
    base_url: str
    model: str
    timeout: float = 60.0
    max_tokens: int = 4096
    batch_size: int = 20
    max_regions_per_agent: int = 3
    temperature: float = 0.1
    max_retries: int = 1
    cache_dir: str = "outputs/llm_cache"
    calls_output: str = "outputs/runs/llm_calls.jsonl"


class OpenAICompatibleClient:
    def __init__(self, config: LLMPlannerConfig):
        self.config = config

    def _request_key(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "prompt_version": PROMPT_VERSION,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "max_retries": self.config.max_retries,
            "messages": messages,
        }
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _append_call(self, record: dict[str, Any]) -> None:
        if not self.config.calls_output:
            return
        path = Path(self.config.calls_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def complete(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        key = self._request_key(messages)
        cache_path = Path(self.config.cache_dir) / f"{key}.json" if self.config.cache_dir else None
        started = datetime.now(timezone.utc)
        if cache_path is not None and cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            content = str(cached["content"])
            record = {
                "schema_version": "1.0",
                "request_key": key,
                "prompt_version": PROMPT_VERSION,
                "prompt_hash": prompt_hash(messages),
                "model": self.config.model,
                "cache_hit": True,
                "status": "completed",
                "started_at": started.isoformat(),
                "duration_seconds": 0.0,
                "retries": 0,
            }
            self._append_call(record)
            return content, record

        base_url = self.config.base_url.rstrip("/")
        url = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("LOCAL_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        status = "completed"
        error = ""
        content = ""
        usage: dict[str, Any] = {}
        retries = 0
        try:
            result = None
            for attempt in range(self.config.max_retries + 1):
                try:
                    with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                        result = json.loads(response.read().decode("utf-8"))
                    retries = attempt
                    break
                except (OSError, urllib.error.URLError, TimeoutError):
                    retries = attempt
                    if attempt >= self.config.max_retries:
                        raise
            if result is None:
                raise RuntimeError("LLM request completed without a response")
            message = result["choices"][0]["message"]
            content = message.get("content") or message.get("reasoning") or ""
            usage = result.get("usage", {})
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps({"content": content, "usage": usage}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                temporary.replace(cache_path)
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            duration = (datetime.now(timezone.utc) - started).total_seconds()
            record = {
                "schema_version": "1.0",
                "request_key": key,
                "prompt_version": PROMPT_VERSION,
                "prompt_hash": prompt_hash(messages),
                "model": self.config.model,
                "base_url": base_url,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "cache_hit": False,
                "status": status,
                "error": error,
                "usage": usage,
                "started_at": started.isoformat(),
                "duration_seconds": round(duration, 6),
                "retries": retries,
            }
            self._append_call(record)
        return content, record


def _chunked(items: list, size: int):
    size = max(size, 1)
    for start in range(0, len(items), size):
        yield items[start : start + size]


class LLMSemanticRoutePlanner(RoutePlanner):
    policy = "llm_semantic"

    def __init__(self, config: LLMPlannerConfig, seed: int = 2026):
        self.config = config
        self.seed = seed
        self.client = OpenAICompatibleClient(config)

    def plan(self, spawn_plans: list[dict], exits: list, regions: list[dict]) -> PlanningResult:
        exit_infos = []
        for index, (exit_area, label) in enumerate(exits):
            centroid = exit_area.centroid
            exit_infos.append({
                "id": index,
                "name": label,
                "centroid": [round(centroid.x, 2), round(centroid.y, 2)],
                "area_m2": round(exit_area.area, 2),
            })
        contexts = {}
        agent_infos = []
        for spawn_plan in spawn_plans:
            centroid = spawn_plan["polygon"].centroid
            candidates = set(spawn_plan["candidate_exit_indices"])
            candidate_exits = [
                {
                    "name": exits[index][1],
                    "distance_m": round(centroid.distance(exits[index][0].centroid), 2),
                }
                for index in spawn_plan["candidate_exit_indices"]
            ]
            for agent in spawn_plan.get("agent_queue", []):
                agent_infos.append({
                    "agent_id": agent["agent_id"],
                    "spawn": spawn_plan["label"],
                    "spawn_order": agent["spawn_order"],
                    "scheduled_iteration": agent["scheduled_iteration"],
                    "profile": agent.get("profile", {}),
                    "spawn_centroid": [round(centroid.x, 2), round(centroid.y, 2)],
                    "spawn_area_m2": round(spawn_plan["polygon"].area, 2),
                    "candidate_exits": candidate_exits,
                })
                contexts[agent["agent_id"]] = {"agent": agent, "candidates": candidates}
        region_infos = [
            {
                "name": region["name"],
                "description": region["description"],
                "centroid": [round(region["position"][0], 2), round(region["position"][1], 2)],
                "area_m2": round(region["area"], 2),
            }
            for region in regions
        ]
        labels = {label: index for index, (_, label) in enumerate(exits)}
        region_names = {region["name"] for region in regions}
        applied = 0
        failed_batches = 0
        validation_errors = 0
        batches = list(_chunked(agent_infos, self.config.batch_size))
        print(
            f"\nRequesting LLM batched routing: {self.config.base_url} "
            f"model={self.config.model}, batches={len(batches)}, batch_size={self.config.batch_size}"
        )
        for batch_index, batch in enumerate(batches, start=1):
            messages = build_agent_routing_messages(batch, region_infos, exit_infos)
            try:
                content, _ = self.client.complete(messages)
                response = extract_json_object(content)
                validate_llm_response(response)
            except (OSError, urllib.error.URLError, json.JSONDecodeError, KeyError, ValueError) as exc:
                failed_batches += 1
                print(f"  batch {batch_index}/{len(batches)} failed; fallback will be used: {exc}")
                continue
            batch_applied = 0
            for item in response.get("plans", []):
                if not isinstance(item, dict):
                    continue
                agent_id = str(item.get("agent_id", ""))
                context = contexts.get(agent_id)
                if context is None:
                    continue
                exit_idx = labels.get(item.get("final_exit") or item.get("exit"))
                if exit_idx not in context["candidates"]:
                    validation_errors += 1
                    continue
                profile = context["agent"].get("profile", {})
                raw_activities = item.get("activities")
                if not isinstance(raw_activities, list):
                    raw_activities = [{"region": name} for name in item.get("regions", [])]
                activities = []
                for activity in raw_activities:
                    if not isinstance(activity, dict) or len(activities) >= self.config.max_regions_per_agent:
                        continue
                    region = activity.get("region")
                    if region not in region_names:
                        validation_errors += 1
                        continue
                    activities.append({
                        "region": region,
                        "action": str(activity.get("action") or "visit"),
                        "wait_seconds": normalize_wait_seconds(
                            activity.get("wait_seconds"),
                            profile.get("default_wait_seconds", 0),
                        ),
                    })
                route = route_from_mapping({
                    "agent_id": agent_id,
                    "exit_idx": exit_idx,
                    "regions": [activity["region"] for activity in activities],
                    "activities": activities,
                    "role": item.get("role") or profile.get("role", "visitor"),
                    "subtype": item.get("subtype") or profile.get("subtype"),
                    "intent": item.get("intent", ""),
                    "desired_speed_mps": item.get("desired_speed_mps") or profile.get("desired_speed_mps", 1.34),
                    "planner": self.policy,
                }).to_json()
                errors = validate_route(route, context["candidates"], len(exits), region_names)
                if errors:
                    validation_errors += len(errors)
                    continue
                set_agent_route(context["agent"], route)
                applied += 1
                batch_applied += 1
            print(f"  batch {batch_index}/{len(batches)}: planned {batch_applied}/{len(batch)} agents")

        fallback_agents = 0
        if applied:
            for spawn_plan in spawn_plans:
                rng = random.Random(f"llm-fallback-{spawn_plan['label']}")
                for agent in spawn_plan.get("agent_queue", []):
                    if agent.get("route") is not None:
                        continue
                    route = fallback_route(agent, rng.choice(spawn_plan["candidate_exit_indices"]))
                    route.update({
                        "planner": self.policy,
                        "fallback_reason": "missing_or_invalid_llm_plan",
                        "schema_version": "1.0",
                    })
                    set_agent_route(agent, route)
                    fallback_agents += 1
        _finalize_spawn_plans(spawn_plans)
        total = len(agent_infos)
        print(f"  LLM routing applied: {applied}/{total} agents, failed_batches={failed_batches}")
        return PlanningResult(
            self.policy,
            total,
            applied,
            failed_batches=failed_batches,
            fallback_agents=fallback_agents,
            validation_errors=validation_errors,
            metadata={
                "prompt_version": PROMPT_VERSION,
                "model": self.config.model,
                "cache_dir": self.config.cache_dir,
                "calls_output": self.config.calls_output,
            },
        )


class LLMExitOnlyRoutePlanner(LLMSemanticRoutePlanner):
    policy = "llm_exit_only"

    def __init__(self, config: LLMPlannerConfig, seed: int = 2026):
        if config.max_regions_per_agent != 0:
            config = LLMPlannerConfig(**{**asdict(config), "max_regions_per_agent": 0})
        super().__init__(config, seed=seed)
