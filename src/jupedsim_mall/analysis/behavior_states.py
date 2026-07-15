"""Sliding-window behavioral state discovery with held-out stability checks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from jupedsim_mall.analysis.run_metrics import (
    _artifact_path,
    _region_at,
    extract_run_metrics,
    load_events,
    load_regions,
    load_trajectories,
)
from jupedsim_mall.experiments.verify_runs import discover_manifests
from jupedsim_mall.project import PROJECT_ROOT

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "analysis" / "behavior_states.json"


def _angle_change(ax: float, ay: float, bx: float, by: float) -> float:
    norm_a, norm_b = math.hypot(ax, ay), math.hypot(bx, by)
    if norm_a <= 1e-9 or norm_b <= 1e-9:
        return 0.0
    cosine = max(-1.0, min(1.0, (ax * bx + ay * by) / (norm_a * norm_b)))
    return math.acos(cosine)


def extract_window_features(manifest_path: pathlib.Path, config: dict) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    trajectory_path = _artifact_path(manifest_path, artifacts["trajectory"])
    events = load_events(_artifact_path(manifest_path, artifacts["events"]))
    _, fps, trajectories = load_trajectories(trajectory_path)
    run_metrics = extract_run_metrics(manifest_path)
    agent_metadata = {row["agent_id"]: row for row in run_metrics["agent_rows"]}
    sim_to_agent = {
        int(event["simulation_agent_id"]): str(event["agent_id"])
        for event in events
        if event.get("simulation_agent_id") is not None and event.get("agent_id")
    }
    regions = load_regions(PROJECT_ROOT / "data" / "map" / "localization_grid_regions.json")
    region_names = sorted([name for name, _ in regions] + ["unmapped"])
    region_codes = {name: index for index, name in enumerate(region_names)}
    window_frames = max(2, int(float(config["window_seconds"]) * fps))
    stride_frames = max(1, int(float(config["stride_seconds"]) * fps))
    rows = []
    for sim_id, points in trajectories.items():
        agent_id = sim_to_agent.get(sim_id, f"simulation_agent_{sim_id}")
        metadata = agent_metadata.get(agent_id, {})
        if len(points) < 2:
            continue
        first_frame, last_frame = points[0][0], points[-1][0]
        start = first_frame
        while start <= last_frame:
            window = [point for point in points if start <= point[0] < start + window_frames]
            if len(window) < 2:
                start += stride_frames
                continue
            speeds: list[float] = []
            turning: list[float] = []
            slow_time = stop_time = total_time = 0.0
            vectors: list[tuple[float, float]] = []
            for left, right in zip(window, window[1:]):
                elapsed = max(0.0, (right[0] - left[0]) / fps)
                dx, dy = right[1] - left[1], right[2] - left[2]
                speed = math.hypot(dx, dy) / elapsed if elapsed > 0 else 0.0
                speeds.append(speed)
                vectors.append((dx, dy))
                total_time += elapsed
                slow_time += elapsed if speed < 0.2 else 0.0
                stop_time += elapsed if speed < 0.05 else 0.0
            for vector_left, vector_right in zip(vectors, vectors[1:]):
                turning.append(_angle_change(*vector_left, *vector_right))
            dominant_region = Counter(
                _region_at(x, y, regions) for _, x, y in window
            ).most_common(1)[0][0]
            rows.append({
                "run_id": manifest["run_id"],
                "scenario_name": manifest["scenario_name"],
                "seed": manifest["seed"],
                "agent_id": agent_id,
                "window_start_seconds": round(start / fps, 6),
                "window_end_seconds": round((start + window_frames) / fps, 6),
                "role": metadata.get("role", "unknown"),
                "profile_source": metadata.get("profile_source", "unknown"),
                "terminal_state": metadata.get("terminal_state", "unknown"),
                "rerouted": bool(metadata.get("rerouted")),
                "mean_speed_mps": float(np.mean(speeds)),
                "speed_std_mps": float(np.std(speeds)),
                "slow_ratio": slow_time / total_time if total_time else 0.0,
                "turning_rad": float(np.mean(turning)) if turning else 0.0,
                "local_density_per_m2": metadata.get("mean_local_density_per_m2") or 0.0,
                "stop_ratio": stop_time / total_time if total_time else 0.0,
                "region": dominant_region,
                "region_code": region_codes[dominant_region],
            })
            start += stride_frames
    return rows


def fit_states(rows: list[dict], config: dict) -> tuple[dict, list[dict]]:
    feature_names = config["features"]
    if len(rows) < 3:
        raise ValueError("at least three behavior windows are required")
    run_ids = sorted({row["run_id"] for row in rows})
    holdout_count = int(round(len(run_ids) * float(config["holdout_run_fraction"]))) if len(run_ids) > 1 else 0
    holdout_ids = set(run_ids[-holdout_count:]) if holdout_count else set()
    train_rows = [row for row in rows if row["run_id"] not in holdout_ids]
    holdout_rows = [row for row in rows if row["run_id"] in holdout_ids]
    train = np.asarray([[float(row[name]) for name in feature_names] for row in train_rows])
    scaler = StandardScaler().fit(train)
    scaled = scaler.transform(train)
    unique_feature_rows = len(np.unique(np.round(scaled, decimals=10), axis=0))
    candidates = [
        int(value) for value in config["candidate_clusters"]
        if 2 <= int(value) < len(train_rows) and int(value) <= unique_feature_rows
    ]
    if not candidates:
        raise ValueError("not enough training windows for the configured cluster candidates")
    scores: dict[int, float] = {}
    models: dict[int, KMeans] = {}
    for clusters in candidates:
        model = KMeans(
            n_clusters=clusters,
            random_state=int(config["random_seed"]),
            n_init=int(config["n_init"]),
        ).fit(scaled)
        if len(set(model.labels_)) < 2:
            score = -1.0
        else:
            score = float(silhouette_score(scaled, model.labels_))
        scores[clusters] = score
        models[clusters] = model
    selected = max(scores, key=lambda clusters: (scores[clusters], -clusters))
    model = models[selected]
    for row, label in zip(train_rows, model.labels_):
        row["state_id"] = int(label)
        row["split"] = "train"
    holdout_distance = None
    if holdout_rows:
        holdout = scaler.transform(
            np.asarray([[float(row[name]) for name in feature_names] for row in holdout_rows])
        )
        labels = model.predict(holdout)
        distances = model.transform(holdout)
        holdout_distance = float(np.mean([distances[index, label] for index, label in enumerate(labels)]))
        for row, label in zip(holdout_rows, labels):
            row["state_id"] = int(label)
            row["split"] = "holdout"
    centers_original = scaler.inverse_transform(model.cluster_centers_)
    center_rows: list[dict[str, Any]] = [
        {"state_id": index, **{
            name: round(float(value), 6)
            for name, value in zip(feature_names, center)
        }}
        for index, center in enumerate(centers_original)
    ]
    state_names: dict[int, str] = {}
    median_speed = float(np.median([center["mean_speed_mps"] for center in center_rows]))
    for center in center_rows:
        traits = []
        traits.append("higher_speed" if center["mean_speed_mps"] >= median_speed else "lower_speed")
        if center.get("slow_ratio", 0) > 0.4:
            traits.append("slow")
        if center.get("turning_rad", 0) > 0.4:
            traits.append("turning")
        if center.get("local_density_per_m2", 0) >= 1.5:
            traits.append("dense")
        state_names[center["state_id"]] = "_".join(traits)
        center["data_driven_name"] = state_names[center["state_id"]]
    for row in rows:
        row["state_name"] = state_names[row["state_id"]]
    transitions: Counter[tuple[int, int]] = Counter()
    dwell: dict[int, float] = defaultdict(float)
    for agent_rows in _group(rows, ("run_id", "agent_id")).values():
        ordered = sorted(agent_rows, key=lambda row: row["window_start_seconds"])
        for row in ordered:
            dwell[row["state_id"]] += float(config["stride_seconds"])
        for left, right in zip(ordered, ordered[1:]):
            transitions[(left["state_id"], right["state_id"])] += 1
    associations = {}
    for field in ("role", "profile_source", "terminal_state", "rerouted"):
        associations[field] = {
            str(value): dict(Counter(row["state_name"] for row in rows if row.get(field) == value))
            for value in sorted({row.get(field) for row in rows}, key=str)
        }
    model_payload = {
        "schema_version": "1.0",
        "features": feature_names,
        "random_seed": config["random_seed"],
        "n_init": config["n_init"],
        "candidate_silhouette_scores": {str(key): round(value, 6) for key, value in scores.items()},
        "selected_clusters": selected,
        "scaler": {
            "mean": [round(float(value), 8) for value in scaler.mean_],
            "scale": [round(float(value), 8) for value in scaler.scale_],
        },
        "state_centers": center_rows,
        "state_occupancy": dict(Counter(row["state_name"] for row in rows)),
        "state_dwell_seconds": {state_names[key]: round(value, 6) for key, value in dwell.items()},
        "transition_matrix": [
            {"from_state": state_names[source], "to_state": state_names[target], "count": count}
            for (source, target), count in sorted(transitions.items())
        ],
        "associations": associations,
        "holdout": {
            "run_ids": sorted(holdout_ids),
            "windows": len(holdout_rows),
            "mean_distance_to_assigned_center": round(holdout_distance, 6) if holdout_distance is not None else None,
        },
        "interpretation": "State names are generated from relative center features and are supplementary, not primary outcomes.",
    }
    return model_payload, rows


def _group(rows: list[dict], fields: tuple[str, ...]) -> dict[tuple, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in fields)].append(row)
    return grouped


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Run directories or parent runs directories.")
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=pathlib.Path, default=PROJECT_ROOT / "outputs" / "behavior_states")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifests = discover_manifests(args.paths)
    rows = [row for manifest in manifests for row in extract_window_features(manifest, config)]
    model, rows = fit_states(rows, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "behavior_state_model.json").write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(args.output_dir / "window_states.csv", rows)
    _write_csv(args.output_dir / "state_transitions.csv", model["transition_matrix"])
    print(f"Behavior windows: {len(rows)}, selected states: {model['selected_clusters']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
