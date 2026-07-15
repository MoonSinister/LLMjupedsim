"""Distribution metrics for experiment summaries.

These helpers adapt the JSD-style evaluation idea used by mobility generation
workflows to the indoor JuPedSim artifacts produced by this project.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Iterable, Mapping
import random


def js_divergence_from_counts(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = sorted(set(left) | set(right))
    if not keys:
        return 0.0
    p = [float(left.get(key, 0.0)) for key in keys]
    q = [float(right.get(key, 0.0)) for key in keys]
    return js_divergence(p, q)


def js_divergence(left: Iterable[float], right: Iterable[float]) -> float:
    p = [max(0.0, float(value)) for value in left]
    q = [max(0.0, float(value)) for value in right]
    if len(p) != len(q):
        raise ValueError("JSD inputs must have the same length")
    sp = sum(p)
    sq = sum(q)
    if sp <= 0 and sq <= 0:
        return 0.0
    if sp <= 0 or sq <= 0:
        return round(math.log(2.0), 6)
    p = [value / sp for value in p]
    q = [value / sq for value in q]
    m = [(a + b) / 2.0 for a, b in zip(p, q)]
    return round((kl_divergence(p, m) + kl_divergence(q, m)) / 2.0, 6)


def kl_divergence(left: Iterable[float], right: Iterable[float]) -> float:
    total = 0.0
    for p, q in zip(left, right):
        if p <= 0:
            continue
        total += p * math.log(p / max(q, 1e-12))
    return total


def histogram_counts(values: Iterable[float], bin_size: float, max_value: float) -> dict[str, int]:
    if bin_size <= 0:
        raise ValueError("bin_size must be positive")
    bins = int(math.ceil(max_value / bin_size))
    counts: Counter[str] = Counter()
    for value in values:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed < 0:
            continue
        index = int(parsed // bin_size)
        if index >= bins:
            counts[f">={max_value:g}"] += 1
        else:
            start = index * bin_size
            end = start + bin_size
            counts[f"{start:g}-{end:g}"] += 1
    return dict(counts)


def entropy_from_counts(counts: Mapping[str, float], normalized: bool = False) -> float:
    total = sum(max(0.0, float(value)) for value in counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    active = 0
    for value in counts.values():
        p = max(0.0, float(value)) / total
        if p <= 0:
            continue
        active += 1
        entropy -= p * math.log(p)
    if normalized and active > 1:
        entropy /= math.log(active)
    return round(entropy, 6)


def plan_distribution_metrics(agents: list[dict]) -> dict:
    roles = Counter(agent.get("role", "unknown") for agent in agents)
    exits = Counter(agent.get("final_exit", "unknown") for agent in agents)
    regions: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    wait_seconds = []
    activity_counts = []

    for agent in agents:
        activities = agent.get("activities", [])
        if not isinstance(activities, list):
            activities = []
        activity_counts.append(len(activities))
        for activity in activities:
            if not isinstance(activity, dict):
                continue
            region = activity.get("region")
            action = activity.get("action")
            if region:
                regions[str(region)] += 1
            if action:
                actions[str(action)] += 1
            wait_seconds.append(activity.get("wait_seconds", 0))

    return {
        "role_distribution": dict(roles),
        "exit_distribution": dict(exits),
        "region_visit_distribution": dict(regions),
        "activity_action_distribution": dict(actions),
        "activity_count_distribution": histogram_counts(activity_counts, bin_size=1, max_value=5),
        "wait_seconds_distribution": histogram_counts(wait_seconds, bin_size=30, max_value=180),
    }


def distribution_points(counts: Mapping[str, float]) -> list[tuple[float, float]]:
    """Convert numeric histogram labels to sorted representative points."""
    points = []
    for label, count in counts.items():
        text = str(label)
        try:
            if text.startswith(">="):
                value = float(text[2:])
            elif "-" in text:
                left, right = text.split("-", 1)
                value = (float(left) + float(right)) / 2
            else:
                value = float(text)
        except ValueError:
            continue
        points.append((value, max(0.0, float(count))))
    return sorted(points)


def wasserstein_from_counts(left: Mapping[str, float], right: Mapping[str, float]) -> float | None:
    left_points = distribution_points(left)
    right_points = distribution_points(right)
    if not left_points or not right_points:
        return None
    coordinates = sorted({value for value, _ in left_points} | {value for value, _ in right_points})
    left_total = sum(count for _, count in left_points)
    right_total = sum(count for _, count in right_points)
    if left_total <= 0 or right_total <= 0:
        return None
    left_map, right_map = dict(left_points), dict(right_points)
    left_cdf = right_cdf = distance = 0.0
    for index, value in enumerate(coordinates[:-1]):
        left_cdf += left_map.get(value, 0.0) / left_total
        right_cdf += right_map.get(value, 0.0) / right_total
        distance += abs(left_cdf - right_cdf) * (coordinates[index + 1] - value)
    return round(distance, 6)


def ks_from_counts(left: Mapping[str, float], right: Mapping[str, float]) -> float | None:
    left_points = distribution_points(left)
    right_points = distribution_points(right)
    if not left_points or not right_points:
        return None
    coordinates = sorted({value for value, _ in left_points} | {value for value, _ in right_points})
    left_total = sum(count for _, count in left_points)
    right_total = sum(count for _, count in right_points)
    if left_total <= 0 or right_total <= 0:
        return None
    left_map, right_map = dict(left_points), dict(right_points)
    left_cdf = right_cdf = statistic = 0.0
    for value in coordinates:
        left_cdf += left_map.get(value, 0.0) / left_total
        right_cdf += right_map.get(value, 0.0) / right_total
        statistic = max(statistic, abs(left_cdf - right_cdf))
    return round(statistic, 6)


def bootstrap_jsd_ci(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    iterations: int = 500,
    seed: int = 2026,
    confidence: float = 0.95,
) -> tuple[float, float] | None:
    keys = sorted(set(left) | set(right))
    left_total = int(sum(max(0.0, float(left.get(key, 0))) for key in keys))
    right_total = int(sum(max(0.0, float(right.get(key, 0))) for key in keys))
    if not keys or left_total <= 0 or right_total <= 0 or iterations <= 0:
        return None
    left_population = [key for key in keys for _ in range(int(max(0, float(left.get(key, 0)))))]
    right_population = [key for key in keys for _ in range(int(max(0, float(right.get(key, 0)))))]
    if not left_population or not right_population:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(iterations):
        left_sample = Counter(rng.choice(left_population) for _ in range(left_total))
        right_sample = Counter(rng.choice(right_population) for _ in range(right_total))
        values.append(js_divergence_from_counts(left_sample, right_sample))
    values.sort()
    alpha = (1.0 - confidence) / 2.0
    low = values[max(0, int(alpha * len(values)))]
    high = values[min(len(values) - 1, int((1.0 - alpha) * len(values)) - 1)]
    return round(low, 6), round(high, 6)
