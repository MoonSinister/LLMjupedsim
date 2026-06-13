"""LLMob training and persona-identification adapter for JuPedSim agents.

This module migrates the data side of LLMob into the indoor JuPedSim workflow.
It reads LLMob pickle records, summarizes each person's historical routines,
identifies a mobility role from activity/category evidence, and exposes a
profile sampler that can be plugged into ``create_agent_queue``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import csv
import json
import math
import pathlib
import pickle
import random
import re


ROLE_KEYWORDS = {
    "commuter": [
        "station",
        "train",
        "subway",
        "metro",
        "rail",
        "bus",
        "airport",
        "parking",
        "transport",
        "commute",
        "office",
        "work",
    ],
    "shopper": [
        "shop",
        "store",
        "mall",
        "market",
        "retail",
        "restaurant",
        "cafe",
        "coffee",
        "food",
        "bar",
        "bakery",
        "grocery",
    ],
    "staff": [
        "office",
        "work",
        "company",
        "service",
        "bank",
        "school",
        "university",
        "hospital",
        "government",
    ],
    "visitor": [
        "hotel",
        "park",
        "museum",
        "theater",
        "cinema",
        "tourist",
        "entertainment",
        "landmark",
        "gym",
        "stadium",
    ],
}


ROLE_DEFAULTS = {
    "commuter": {
        "desired_speed_mps": 1.45,
        "default_wait_seconds": 0,
        "motivation": "Move efficiently through the building toward transit, parking, or a direct exit.",
        "preferences": ["direct exit", "train corridor", "parking connection", "wide corridor"],
    },
    "shopper": {
        "desired_speed_mps": 1.15,
        "default_wait_seconds": 55,
        "motivation": "Visit shopping or dining areas before leaving through a convenient exit.",
        "preferences": ["shops", "restaurant", "convenience store", "main corridor"],
    },
    "staff": {
        "desired_speed_mps": 1.30,
        "default_wait_seconds": 25,
        "motivation": "Move through familiar work-related or service-oriented areas.",
        "preferences": ["office elevator", "service corridor", "utility access", "convenience store"],
    },
    "visitor": {
        "desired_speed_mps": 1.20,
        "default_wait_seconds": 35,
        "motivation": "Explore visible landmarks or public areas before choosing an exit.",
        "preferences": ["visible landmarks", "shops", "restaurant", "stairs", "escalator"],
    },
}


SUBTYPE_BY_ROLE = {
    "commuter": ["train_connection", "parking_connection", "hurried_passenger"],
    "shopper": ["window_shopper", "restaurant_visitor", "quick_buyer"],
    "staff": ["office_worker", "retail_employee", "maintenance_staff"],
    "visitor": ["first_time_visitor", "lost_visitor", "meeting_visitor"],
}


PERSONAL_VARIATIONS = [
    "prefers shortest plausible path",
    "accepts a small detour for interesting places",
    "avoids unnecessary stops",
    "may pause briefly at the intended destination",
    "prefers a different stop than nearby agents with the same role",
    "chooses a less crowded-looking alternative when plausible",
]


@dataclass
class LLMobPersonRecord:
    person_id: str
    dataset: str
    train_routine_list: list
    test_routine_list: list
    source_attribute: str
    cat: object
    domain_knowledge: str
    neg_routines: list
    activity_area: list
    area_freq: object
    loc_cat: object


@dataclass
class ATCPersonTrack:
    person_id: str
    source: str
    count: int = 0
    first_time: float | None = None
    last_time: float | None = None
    first_position: tuple[float, float] | None = None
    last_position: tuple[float, float] | None = None
    previous_position: tuple[float, float] | None = None
    path_length_m: float = 0.0
    speed_sum: float = 0.0
    slow_count: int = 0
    region_counter: Counter | None = None

    def __post_init__(self):
        if self.region_counter is None:
            self.region_counter = Counter()

    def add(self, time_s, x_m, y_m, speed_m_s, region=""):
        position = (x_m, y_m)
        if self.count == 0:
            self.first_time = time_s
            self.first_position = position
        if self.previous_position is not None:
            self.path_length_m += _distance(self.previous_position, position)
        self.previous_position = position
        self.last_time = time_s
        self.last_position = position
        self.speed_sum += max(0.0, speed_m_s)
        if speed_m_s < 0.35:
            self.slow_count += 1
        if region:
            self.region_counter[region] += 1
        self.count += 1


def _safe_item(items, index, default=None):
    try:
        return items[index]
    except (IndexError, TypeError):
        return default


def load_llmob_records(data_root, dataset="2019", max_persons=0):
    """Load LLMob pickle records using the field layout from LLMob/generate.py."""
    folder = pathlib.Path(data_root) / dataset
    if not folder.exists():
        raise FileNotFoundError(f"LLMob dataset folder not found: {folder}")

    records = []
    for path in sorted(folder.glob("*.pkl"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
        with path.open("rb") as handle:
            raw = pickle.load(handle)
        records.append(LLMobPersonRecord(
            person_id=path.stem,
            dataset=dataset,
            train_routine_list=list(_safe_item(raw, 0, []) or []),
            test_routine_list=list(_safe_item(raw, 1, []) or []),
            source_attribute=str(_safe_item(raw, 2, "") or ""),
            cat=_safe_item(raw, 4, None),
            domain_knowledge=str(_safe_item(raw, 5, "") or ""),
            neg_routines=list(_safe_item(raw, 6, []) or []),
            activity_area=list(_safe_item(raw, 7, []) or []),
            area_freq=_safe_item(raw, 8, None),
            loc_cat=_safe_item(raw, 11, None),
        ))
        if max_persons and len(records) >= max_persons:
            break
    return records


def _clean_activity_name(value):
    text = str(value)
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\bat\s+\d{1,2}:\d{2}(?::\d{2})?\b", "", text, flags=re.IGNORECASE)
    text = text.replace("Activities", "")
    text = text.replace("Go to", "")
    text = text.replace("Visit", "")
    text = text.replace("Head to", "")
    text = text.strip(" -,:;.")
    return re.sub(r"\s+", " ", text).strip()


def extract_activity_names(routine):
    """Extract approximate activity/location names from an LLMob routine string."""
    if not routine:
        return []
    text = str(routine)
    if ": " in text:
        text = text.split(": ", 1)[1]

    activities = []
    paren_matches = re.findall(r"([^,]+?)\s*\([^)]*\)", text)
    for item in paren_matches:
        cleaned = _clean_activity_name(item)
        if cleaned:
            activities.append(cleaned)

    if not activities:
        for item in re.split(r",|->| at ", text):
            cleaned = _clean_activity_name(item)
            if cleaned and not re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", cleaned):
                activities.append(cleaned)
    return activities


def _lookup_category(activity, loc_cat):
    if not loc_cat:
        return ""

    candidates = [activity, activity.lower(), activity.split("#", 1)[0], activity.split("#", 1)[0].lower()]
    if isinstance(loc_cat, dict):
        for key in candidates:
            value = loc_cat.get(key)
            if value:
                return str(value)
        activity_lower = activity.lower()
        for key, value in loc_cat.items():
            key_text = str(key).lower()
            if activity_lower in key_text or key_text in activity_lower:
                return str(value)
    return ""


def _counter_from_area_freq(area_freq):
    counter = Counter()
    if isinstance(area_freq, dict):
        for key, value in area_freq.items():
            try:
                counter[str(key)] += float(value)
            except (TypeError, ValueError):
                counter[str(key)] += 1
    return counter


def summarize_person(record):
    activities = []
    category_counter = Counter()
    for routine in record.train_routine_list:
        for activity in extract_activity_names(routine):
            activities.append(activity)
            category = _lookup_category(activity, record.loc_cat)
            if category:
                category_counter[category] += 1

    activity_counter = Counter(activities)
    area_counter = _counter_from_area_freq(record.area_freq)
    for area in record.activity_area:
        if area:
            area_counter[str(area)] += 1

    evidence_text = " ".join(
        list(activity_counter.keys())
        + list(category_counter.keys())
        + list(area_counter.keys())
        + [record.domain_knowledge, record.source_attribute]
    ).lower()
    role_scores = {}
    for role, keywords in ROLE_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            score += evidence_text.count(keyword)
        role_scores[role] = score

    for activity, count in activity_counter.items():
        lowered = activity.lower()
        for role, keywords in ROLE_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                role_scores[role] += count

    return {
        "top_activities": activity_counter.most_common(10),
        "top_categories": category_counter.most_common(10),
        "top_areas": area_counter.most_common(10),
        "role_scores": role_scores,
        "train_routine_count": len(record.train_routine_list),
        "test_routine_count": len(record.test_routine_list),
        "negative_routine_count": len(record.neg_routines),
    }


def identify_profile(record):
    """Identify one indoor-compatible JuPedSim profile from an LLMob person."""
    summary = summarize_person(record)
    role_scores = summary["role_scores"]
    role = max(role_scores, key=lambda key: (role_scores[key], key))
    if role_scores[role] <= 0:
        role = "visitor"

    defaults = ROLE_DEFAULTS[role]
    top_categories = [name for name, _ in summary["top_categories"][:5]]
    top_activities = [name for name, _ in summary["top_activities"][:5]]
    evidence = " ".join(top_categories + top_activities).lower()

    subtype_options = SUBTYPE_BY_ROLE[role]
    subtype_index = _stable_index(f"{record.person_id}-{evidence}", len(subtype_options))
    subtype = subtype_options[subtype_index]

    preferences = list(defaults["preferences"])
    preferences.extend(_indoor_preferences_from_evidence(evidence))
    preferences = list(dict.fromkeys(preferences))[:8]

    wait = defaults["default_wait_seconds"]
    speed = defaults["desired_speed_mps"]
    richness = summary["train_routine_count"] + len(summary["top_activities"])
    if role == "commuter":
        wait = max(0, min(15, int(wait + richness % 6)))
        speed += 0.03
    elif role == "shopper":
        wait = max(20, min(120, int(wait + (richness % 8) * 5)))
        speed -= 0.03
    elif role == "staff":
        wait = max(5, min(70, int(wait + (richness % 5) * 4)))
    else:
        wait = max(10, min(90, int(wait + (richness % 7) * 5)))

    attribute = record.source_attribute or _build_attribute_sentence(role, top_activities, top_categories)
    motivation = _build_motivation(role, top_activities, top_categories, defaults["motivation"])

    return {
        "profile_id": f"llmob_{record.dataset}_{record.person_id}",
        "source": "llmob",
        "llmob_person_id": record.person_id,
        "llmob_dataset": record.dataset,
        "role": role,
        "subtype": subtype,
        "attribute": attribute,
        "motivation": motivation,
        "preferences": preferences,
        "personal_variation": PERSONAL_VARIATIONS[_stable_index(record.person_id, len(PERSONAL_VARIATIONS))],
        "desired_speed_mps": round(max(0.8, min(1.8, speed)), 2),
        "default_wait_seconds": wait,
        "training_summary": summary,
        "domain_knowledge": _shorten(record.domain_knowledge, 500),
        "reference_routine": _shorten(record.train_routine_list[-1] if record.train_routine_list else "", 500),
        "identification_method": "llmob_heuristic_self_consistency",
    }


def _indoor_preferences_from_evidence(evidence):
    preferences = []
    if any(word in evidence for word in ["restaurant", "cafe", "coffee", "food", "bar"]):
        preferences.append("restaurant")
    if any(word in evidence for word in ["shop", "store", "mall", "market"]):
        preferences.append("shops")
    if any(word in evidence for word in ["office", "work", "company"]):
        preferences.append("office elevator")
        preferences.append("service corridor")
    if any(word in evidence for word in ["station", "train", "subway", "bus"]):
        preferences.append("train corridor")
        preferences.append("direct exit")
    if "parking" in evidence:
        preferences.append("parking connection")
    if any(word in evidence for word in ["hotel", "museum", "park", "cinema", "theater"]):
        preferences.append("visible landmarks")
    return preferences


def _build_attribute_sentence(role, top_activities, top_categories):
    details = top_categories[:3] or top_activities[:3]
    if details:
        return f"You behave like a {role} whose historical routines often involve {', '.join(details)}."
    return f"You behave like a {role} with limited historical activity evidence."


def _build_motivation(role, top_activities, top_categories, fallback):
    details = top_categories[:2] or top_activities[:2]
    if details:
        return f"Use historical preference for {', '.join(details)} to choose a plausible indoor stop, then leave."
    return fallback


def _shorten(value, max_length):
    text = str(value or "")
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def _stable_index(value, modulo):
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


class LLMobProfileSampler:
    """Stable profile sampler built from identified LLMob person profiles."""

    def __init__(self, profiles, seed=2026):
        if not profiles:
            raise ValueError("LLMobProfileSampler requires at least one profile")
        self.profiles = profiles
        self.seed = seed

    def __call__(self, agent_id, spawn_name, spawn_order, seed=2026):
        rng = random.Random(f"{self.seed}-{seed}-{agent_id}-{spawn_name}-{spawn_order}")
        base = dict(rng.choice(self.profiles))
        base["profile_id"] = f"{base['profile_id']}_{agent_id}"
        base["assigned_agent_id"] = agent_id
        base["spawn"] = spawn_name
        base["spawn_order"] = spawn_order
        base["personal_variation"] = rng.choice(PERSONAL_VARIATIONS)
        base["desired_speed_mps"] = round(
            max(0.8, min(1.8, float(base.get("desired_speed_mps", 1.3)) + rng.uniform(-0.08, 0.08))),
            2,
        )
        return base


def build_llmob_profile_sampler(data_root, dataset="2019", max_persons=0, seed=2026, cache_output=None):
    records = load_llmob_records(data_root, dataset=dataset, max_persons=max_persons)
    profiles = [identify_profile(record) for record in records]
    if cache_output:
        path = pathlib.Path(cache_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"profiles": profiles}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return LLMobProfileSampler(profiles, seed=seed), profiles


def load_atc_regions(regions_path):
    if not regions_path:
        return []
    path = pathlib.Path(regions_path)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    regions = []
    for item in payload.get("regions", []):
        points = item.get("points_world") or item.get("points_px") or []
        if len(points) < 3:
            continue
        regions.append({
            "name": item.get("name", f"region_{len(regions) + 1}"),
            "description": item.get("description", ""),
            "points": [(float(x), float(y)) for x, y in points],
        })
    return regions


def iter_atc_csv_files(path):
    root = pathlib.Path(path)
    if root.is_file():
        return [root]
    return sorted(root.rglob("*.csv"), key=lambda item: item.as_posix().lower())


def load_atc_tracks(raw_path, regions_path=None, max_persons=0, max_rows=0, min_points=300):
    """Stream ATC raw/processed CSV files and aggregate per-person trajectories.

    Raw ATC rows are expected to have:
    time, person_id, x_mm, y_mm, z_mm, speed_mm_s, move_rad, heading_rad.

    Processed ATC rows produced by ATC-map/process_atc_csvs.py are also
    accepted and detected by their header.
    """
    regions = load_atc_regions(regions_path)
    tracks = {}
    completed_ids = []
    completed_set = set()
    total_rows = 0
    for csv_path in iter_atc_csv_files(raw_path):
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            first = next(reader, None)
            if first is None:
                continue
            processed = first and first[0] == "timestamp"
            rows = reader if processed else _chain_first(first, reader)
            for row in rows:
                parsed = _parse_atc_row(row, processed)
                if parsed is None:
                    continue
                time_s, person_id, x_m, y_m, speed_m_s, row_region = parsed
                region = row_region or find_region_name(x_m, y_m, regions)
                track = tracks.get(person_id)
                if track is None:
                    track = ATCPersonTrack(person_id=person_id, source=csv_path.name)
                    tracks[person_id] = track
                track.add(time_s, x_m, y_m, speed_m_s, region=region)
                total_rows += 1
                if track.count >= min_points and person_id not in completed_set:
                    completed_ids.append(person_id)
                    completed_set.add(person_id)
                    if max_persons and len(completed_ids) >= max_persons:
                        return [tracks[completed_id] for completed_id in completed_ids]
                if max_rows and total_rows >= max_rows:
                    if completed_ids:
                        return [tracks[completed_id] for completed_id in completed_ids]
                    return [track for track in tracks.values() if track.count >= min_points]
    return [track for track in tracks.values() if track.count >= min_points]


def _chain_first(first, reader):
    yield first
    yield from reader


def _parse_atc_row(row, processed):
    try:
        if processed:
            time_s = _parse_time_to_epoch_seconds(row[0])
            person_id = str(row[1])
            x_m = float(row[2])
            y_m = float(row[3])
            speed_m_s = float(row[5])
            region = row[9] if len(row) > 9 else ""
            return time_s, person_id, x_m, y_m, speed_m_s, region
        time_s = float(row[0])
        person_id = str(row[1])
        x_m = float(row[2]) / 1000.0
        y_m = float(row[3]) / 1000.0
        speed_m_s = float(row[5]) / 1000.0
        return time_s, person_id, x_m, y_m, speed_m_s, ""
    except (IndexError, TypeError, ValueError):
        return None


def _parse_time_to_epoch_seconds(value):
    from datetime import datetime

    text = str(value)
    try:
        return float(text)
    except ValueError:
        pass
    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).timestamp()
    except ValueError:
        return 0.0


def find_region_name(x_m, y_m, regions):
    for region in regions:
        if point_in_polygon(x_m, y_m, region["points"]):
            return region["name"]
    return ""


def point_in_polygon(x, y, points):
    inside = False
    count = len(points)
    for index in range(count):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % count]
        if (y1 > y) != (y2 > y):
            cross_x = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x < cross_x:
                inside = not inside
    return inside


def identify_atc_profile(track):
    duration = max(0.0, (track.last_time or 0.0) - (track.first_time or 0.0))
    displacement = _distance(track.first_position, track.last_position)
    mean_speed = track.speed_sum / max(track.count, 1)
    slow_ratio = track.slow_count / max(track.count, 1)
    tortuosity = track.path_length_m / max(displacement, 1e-6)
    region_visits = track.region_counter or Counter()
    visited_regions = [name for name, _ in region_visits.most_common(8)]

    role = "visitor"
    if duration <= 90 and displacement >= 8 and mean_speed >= 0.9 and slow_ratio < 0.2 and tortuosity < 2.2:
        role = "commuter"
    elif slow_ratio >= 0.25 or duration >= 180 or len(visited_regions) >= 3:
        role = "shopper"
    elif mean_speed >= 0.75 and duration >= 120 and tortuosity >= 2.0:
        role = "staff"

    defaults = ROLE_DEFAULTS[role]
    preferences = list(defaults["preferences"])
    preferences.extend(visited_regions[:4])
    preferences = list(dict.fromkeys(preferences))[:8]
    subtype_options = SUBTYPE_BY_ROLE[role]

    wait = defaults["default_wait_seconds"]
    if role == "commuter":
        wait = int(max(0, min(10, slow_ratio * 30)))
    elif role == "shopper":
        wait = int(max(25, min(140, 35 + slow_ratio * 220 + len(visited_regions) * 6)))
    elif role == "staff":
        wait = int(max(10, min(80, 20 + slow_ratio * 120)))
    else:
        wait = int(max(10, min(90, 20 + slow_ratio * 140)))

    speed = max(0.8, min(1.8, mean_speed if mean_speed > 0 else defaults["desired_speed_mps"]))
    if role == "commuter":
        speed = max(speed, 1.25)

    summary = {
        "source": track.source,
        "points": track.count,
        "duration_seconds": round(duration, 2),
        "path_length_m": round(track.path_length_m, 2),
        "displacement_m": round(displacement, 2),
        "mean_speed_m_s": round(mean_speed, 3),
        "slow_ratio": round(slow_ratio, 3),
        "tortuosity": round(tortuosity, 3),
        "top_regions": region_visits.most_common(8),
    }
    return {
        "profile_id": f"atc_{track.person_id}",
        "source": "atc",
        "atc_person_id": track.person_id,
        "role": role,
        "subtype": subtype_options[_stable_index(track.person_id, len(subtype_options))],
        "attribute": _build_atc_attribute(role, summary, visited_regions),
        "motivation": _build_atc_motivation(role, visited_regions, defaults["motivation"]),
        "preferences": preferences,
        "personal_variation": PERSONAL_VARIATIONS[_stable_index(track.person_id, len(PERSONAL_VARIATIONS))],
        "desired_speed_mps": round(speed, 2),
        "default_wait_seconds": wait,
        "training_summary": summary,
        "reference_routine": _format_atc_reference(track, visited_regions),
        "identification_method": "atc_trajectory_behavior_heuristic",
    }


def _build_atc_attribute(role, summary, visited_regions):
    region_text = ", ".join(visited_regions[:3]) if visited_regions else "unlabeled indoor areas"
    return (
        f"You behave like a {role} inferred from ATC indoor tracking: "
        f"duration {summary['duration_seconds']}s, mean speed {summary['mean_speed_m_s']}m/s, "
        f"slow ratio {summary['slow_ratio']}, regions {region_text}."
    )


def _build_atc_motivation(role, visited_regions, fallback):
    if visited_regions:
        return f"Follow an indoor movement pattern associated with {', '.join(visited_regions[:3])}, then leave."
    return fallback


def _format_atc_reference(track, visited_regions):
    start = track.first_position or (0.0, 0.0)
    end = track.last_position or (0.0, 0.0)
    via = " -> ".join(visited_regions[:5]) if visited_regions else "unlabeled path"
    return (
        f"ATC trajectory {track.person_id}: "
        f"start=({start[0]:.2f},{start[1]:.2f}), via={via}, "
        f"end=({end[0]:.2f},{end[1]:.2f})"
    )


def _distance(a, b):
    if a is None or b is None:
        return 0.0
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_atc_profile_sampler(
    raw_path,
    regions_path=None,
    max_persons=200,
    max_rows=0,
    min_points=300,
    seed=2026,
    cache_output=None,
):
    tracks = load_atc_tracks(
        raw_path,
        regions_path=regions_path,
        max_persons=max_persons,
        max_rows=max_rows,
        min_points=min_points,
    )
    profiles = [identify_atc_profile(track) for track in tracks]
    if cache_output:
        path = pathlib.Path(cache_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"profiles": profiles}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return LLMobProfileSampler(profiles, seed=seed), profiles
