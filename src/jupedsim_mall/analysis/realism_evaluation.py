"""Distributional realism evaluation against a frozen ATC reference."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from collections.abc import Sequence

from jupedsim_mall.analysis.experiment_summary import summarize_sqlite
from jupedsim_mall.analysis.metrics import (
    bootstrap_jsd_ci,
    js_divergence_from_counts,
    ks_from_counts,
    wasserstein_from_counts,
)
from jupedsim_mall.project import PROJECT_ROOT

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "analysis" / "realism.json"


def load_reference(path: pathlib.Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    distributions = data.get("distributions", {})
    if not isinstance(distributions, dict):
        raise ValueError(f"Invalid reference distributions: {path}")
    return distributions


def simulation_distributions(path: pathlib.Path) -> tuple[dict, dict]:
    if path.name == "metrics.json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        distributions = payload.get("distributions", {})
        return {
            "duration_seconds": distributions.get("travel_time_seconds", {}),
            "path_length_m": distributions.get("path_length_m", {}),
            "mean_speed_mps": distributions.get("mean_speed_mps", {}),
            "region_visits": distributions.get("region_visits", {}),
            "slow_ratio": distributions.get("slow_ratio", {}),
        }, {
            "run_id": payload.get("run_id"),
            "scenario": payload.get("scenario_name", path.parent.name),
            "seed": payload.get("seed"),
        }
    summary = summarize_sqlite(path)
    return {
        "duration_seconds": summary.get("travel_seconds_distribution", {}),
        "path_length_m": summary.get("path_length_distribution", {}),
        "mean_speed_mps": summary.get("observed_speed_distribution", {}),
        "region_visits": summary.get("region_visit_distribution", {}),
        "slow_ratio": summary.get("slow_ratio_distribution", {}),
    }, {"run_id": None, "scenario": path.stem, "seed": None}


def weighted_score(field_results: dict, weights: dict[str, float]) -> tuple[float | None, float | None]:
    used = [
        (weights[field], result["jsd"])
        for field, result in field_results.items()
        if result.get("jsd") is not None and weights.get(field, 0) > 0
    ]
    if not used:
        return None, None
    difference = sum(weight * value for weight, value in used) / sum(weight for weight, _ in used)
    return round(difference, 6), round(1.0 / (1.0 + difference), 6)


def evaluate_one(reference: dict, simulation_path: pathlib.Path, config: dict) -> dict:
    simulated, identity = simulation_distributions(simulation_path)
    bootstrap = config["bootstrap"]
    fields = {}
    weights = {field: float(settings["weight"]) for field, settings in config["fields"].items()}
    for field, settings in config["fields"].items():
        ref_dist, sim_dist = reference.get(field, {}), simulated.get(field, {})
        if not ref_dist or not sim_dist:
            fields[field] = {"jsd": None, "jsd_ci": None, "wasserstein": None, "ks": None}
            continue
        ci = bootstrap_jsd_ci(
            ref_dist,
            sim_dist,
            iterations=int(bootstrap["iterations"]),
            seed=int(bootstrap["seed"]),
            confidence=float(bootstrap["confidence"]),
        )
        fields[field] = {
            "jsd": js_divergence_from_counts(ref_dist, sim_dist),
            "jsd_ci": list(ci) if ci else None,
            "wasserstein": wasserstein_from_counts(ref_dist, sim_dist) if settings.get("continuous") else None,
            "ks": ks_from_counts(ref_dist, sim_dist) if settings.get("continuous") else None,
        }
    weighted_jsd, realism_score = weighted_score(fields, weights)
    sensitivity = []
    for field in weights:
        for factor in config.get("weight_sensitivity", [1.0]):
            varied = dict(weights)
            varied[field] *= float(factor)
            difference, score = weighted_score(fields, varied)
            sensitivity.append({
                "varied_field": field,
                "factor": factor,
                "weighted_jsd": difference,
                "realism_score": score,
            })
    return {
        **identity,
        "path": str(simulation_path),
        "fields": fields,
        "weighted_jsd": weighted_jsd,
        "realism_score": realism_score,
        "weight_sensitivity": sensitivity,
    }


def _flatten(result: dict) -> dict:
    row = {
        "run_id": result.get("run_id"),
        "scenario": result.get("scenario"),
        "seed": result.get("seed"),
        "path": result["path"],
        "weighted_jsd": result["weighted_jsd"],
        "realism_score": result["realism_score"],
    }
    for field, metrics in result["fields"].items():
        row[f"{field}_jsd"] = metrics["jsd"]
        row[f"{field}_jsd_ci_low"] = metrics["jsd_ci"][0] if metrics["jsd_ci"] else None
        row[f"{field}_jsd_ci_high"] = metrics["jsd_ci"][1] if metrics["jsd_ci"] else None
        row[f"{field}_wasserstein"] = metrics["wasserstein"]
        row[f"{field}_ks"] = metrics["ks"]
    return row


def write_csv(results: list[dict], path: pathlib.Path) -> None:
    rows = [_flatten(result) for result in results]
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reference", type=pathlib.Path)
    parser.add_argument("--sim", nargs="+", type=pathlib.Path, required=True)
    parser.add_argument("--output-json", type=pathlib.Path, default=PROJECT_ROOT / "outputs" / "summaries" / "realism_evaluation.json")
    parser.add_argument("--output-csv", type=pathlib.Path, default=PROJECT_ROOT / "outputs" / "summaries" / "realism_evaluation.csv")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    reference_path = args.reference or PROJECT_ROOT / config["reference"]
    reference = load_reference(reference_path)
    results = [evaluate_one(reference, path, config) for path in args.sim if path.exists()]
    if not results:
        raise SystemExit("No simulation metric or SQLite files found.")
    payload = {
        "schema_version": "1.0",
        "reference": str(reference_path),
        "config": config,
        "limitations": config["limitations"],
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(results, args.output_csv)
    print(f"Evaluated {len(results)} simulation artifact(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
