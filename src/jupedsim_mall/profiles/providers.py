"""Unified agent profile providers and preprocessing provenance."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jupedsim_mall.experiments.sim_schema import profile_from_mapping
from jupedsim_mall.profiles.llmob_adapter import build_agent_profile
from jupedsim_mall.profiles.llmob_training_adapter import (
    LLMobProfileSampler,
    build_atc_profile_sampler,
    build_llmob_profile_sampler,
)


def _normalize_profile(raw: dict[str, Any], source: str) -> dict[str, Any]:
    payload = dict(raw)
    payload.setdefault("source", source)
    return profile_from_mapping(payload).to_json()


class ProfileProvider(ABC):
    source = "unknown"

    @abstractmethod
    def sample(self, agent_id: str, spawn_name: str, spawn_order: int, *, seed: int) -> dict[str, Any]:
        raise NotImplementedError

    def __call__(self, agent_id: str, spawn_name: str, spawn_order: int, seed: int = 2026) -> dict[str, Any]:
        return self.sample(agent_id, spawn_name, spawn_order, seed=seed)


class MallProfileProvider(ProfileProvider):
    source = "mall"

    def sample(self, agent_id: str, spawn_name: str, spawn_order: int, *, seed: int) -> dict[str, Any]:
        return _normalize_profile(build_agent_profile(agent_id, spawn_name, spawn_order, seed=seed), self.source)


class IdentifiedProfileProvider(ProfileProvider):
    def __init__(self, source: str, profiles: list[dict[str, Any]], seed: int):
        if not profiles:
            raise ValueError(f"{source} provider requires at least one profile")
        self.source = source
        self.profiles = [_normalize_profile(profile, source) for profile in profiles]
        self.sampler = LLMobProfileSampler(self.profiles, seed=seed)

    def sample(self, agent_id: str, spawn_name: str, spawn_order: int, *, seed: int) -> dict[str, Any]:
        return _normalize_profile(self.sampler(agent_id, spawn_name, spawn_order, seed=seed), self.source)


@dataclass
class ProfileBuildReport:
    requested_source: str
    active_source: str
    profile_count: int
    role_distribution: dict[str, int] = field(default_factory=dict)
    source_fingerprint: str = ""
    cache_key: str = ""
    cache_hit: bool = False
    filters: dict[str, Any] = field(default_factory=dict)
    fallback_reason: str = ""
    input_count: int = 0
    output_count: int = 0
    rejected_count: int = 0
    missing_rate: float = 0.0
    anomaly_rate: float = 0.0
    distributions: dict[str, dict[str, int]] = field(default_factory=dict)
    schema_version: str = "1.0"

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "requested_source": self.requested_source,
            "active_source": self.active_source,
            "profile_count": self.profile_count,
            "role_distribution": self.role_distribution,
            "source_fingerprint": self.source_fingerprint,
            "cache_key": self.cache_key,
            "cache_hit": self.cache_hit,
            "filters": self.filters,
            "fallback_reason": self.fallback_reason,
            "input_count": self.input_count,
            "output_count": self.output_count,
            "rejected_count": self.rejected_count,
            "missing_rate": self.missing_rate,
            "anomaly_rate": self.anomaly_rate,
            "distributions": self.distributions,
        }


@dataclass
class ProfileProviderResult:
    provider: ProfileProvider
    profiles: list[dict[str, Any]]
    report: ProfileBuildReport


def source_fingerprint(path_value: str | Path) -> str:
    path = Path(path_value).expanduser()
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if path.is_dir():
        for item in sorted((entry for entry in path.rglob("*") if entry.is_file()), key=lambda entry: entry.as_posix()):
            stat = item.stat()
            digest.update(item.relative_to(path).as_posix().encode("utf-8"))
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
        return digest.hexdigest()
    return "missing"


def _cache_key(source: str, fingerprint: str, filters: dict[str, Any]) -> str:
    payload = json.dumps(
        {"source": source, "fingerprint": fingerprint, "filters": filters},
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_matching_cache(path: Path | None, cache_key: str) -> list[dict[str, Any]] | None:
    if path is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("preprocessing", {}).get("cache_key") != cache_key:
        return None
    profiles = payload.get("profiles")
    return profiles if isinstance(profiles, list) and profiles else None


def _write_cache(path: Path | None, profiles: list[dict[str, Any]], report: ProfileBuildReport) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "preprocessing": report.to_json(),
        "profiles": profiles,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _role_distribution(profiles: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(profile.get("role", "visitor")) for profile in profiles))


def _profile_quality(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    required = ("profile_id", "role", "desired_speed_mps", "preferences", "source")
    total_fields = len(profiles) * len(required)
    missing = sum(
        value in (None, "", [])
        for profile in profiles
        for value in (profile.get(field) for field in required)
    )
    anomalies = sum(
        not isinstance(profile.get("desired_speed_mps"), (int, float))
        or not 0.8 <= float(profile.get("desired_speed_mps", 0.0)) <= 1.8
        or not isinstance(profile.get("preferences"), list)
        for profile in profiles
    )
    return {
        "missing_rate": round(missing / total_fields, 6) if total_fields else 0.0,
        "anomaly_rate": round(anomalies / len(profiles), 6) if profiles else 0.0,
        "distributions": {
            "role": _role_distribution(profiles),
            "source": dict(Counter(str(profile.get("source", "unknown")) for profile in profiles)),
        },
    }


def build_profile_provider(
    source: str,
    *,
    seed: int = 2026,
    data_path: str = "",
    regions_path: str = "",
    dataset: str = "2019",
    max_persons: int = 0,
    max_rows: int = 0,
    min_points: int = 300,
    cache_output: str = "",
    allow_fallback: bool = True,
    atc_partition: str = "train",
) -> ProfileProviderResult:
    if source == "mall":
        report = ProfileBuildReport("mall", "mall", 0)
        return ProfileProviderResult(MallProfileProvider(), [], report)

    filters = {
        "seed": seed,
        "dataset": dataset if source == "llmob" else None,
        "max_persons": max_persons,
        "max_rows": max_rows if source == "atc" else None,
        "min_points": min_points if source == "atc" else None,
        "regions_path": regions_path if source == "atc" else None,
        "partition": atc_partition if source == "atc" else None,
    }
    fingerprint = source_fingerprint(data_path)
    cache_key = _cache_key(source, fingerprint, filters)
    cache_path = Path(cache_output) if cache_output else None
    cached = _load_matching_cache(cache_path, cache_key)
    try:
        cache_hit = cached is not None
        if cached is not None:
            profiles = cached
        elif source == "llmob":
            _, profiles = build_llmob_profile_sampler(
                data_path,
                dataset=dataset,
                max_persons=max_persons,
                seed=seed,
                cache_output=None,
            )
        elif source == "atc":
            _, profiles = build_atc_profile_sampler(
                data_path,
                regions_path=regions_path or None,
                max_persons=max_persons,
                max_rows=max_rows,
                min_points=min_points,
                seed=seed,
                cache_output=None,
                partition=atc_partition,
            )
        else:
            raise ValueError(f"unknown profile source: {source}")
        normalized = [_normalize_profile(profile, source) for profile in profiles]
        quality = _profile_quality(normalized)
        report = ProfileBuildReport(
            requested_source=source,
            active_source=source,
            profile_count=len(normalized),
            role_distribution=_role_distribution(normalized),
            source_fingerprint=fingerprint,
            cache_key=cache_key,
            cache_hit=cache_hit,
            filters=filters,
            input_count=len(profiles),
            output_count=len(normalized),
            rejected_count=max(0, len(profiles) - len(normalized)),
            **quality,
        )
        _write_cache(cache_path, normalized, report)
        return ProfileProviderResult(IdentifiedProfileProvider(source, normalized, seed), normalized, report)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        if not allow_fallback:
            raise
        report = ProfileBuildReport(
            requested_source=source,
            active_source="mall",
            profile_count=0,
            source_fingerprint=fingerprint,
            cache_key=cache_key,
            filters=filters,
            fallback_reason=f"{type(exc).__name__}: {exc}",
        )
        return ProfileProviderResult(MallProfileProvider(), [], report)


def partition_overlap(left_ids: set[str], right_ids: set[str]) -> set[str]:
    """Return leaked person/date identifiers shared by two data partitions."""
    return {str(item) for item in left_ids & right_ids}
