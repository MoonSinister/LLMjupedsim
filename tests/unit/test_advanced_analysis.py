import json

from jupedsim_mall.analysis.metrics import (
    bootstrap_jsd_ci,
    ks_from_counts,
    wasserstein_from_counts,
)
from jupedsim_mall.analysis.realism_evaluation import evaluate_one
from jupedsim_mall.analysis.statistics import analyze, bootstrap_mean_ci, holm_adjust


def test_distribution_distances_and_bootstrap_are_deterministic():
    left = {"0-1": 8, "1-2": 2}
    right = {"0-1": 2, "1-2": 8}
    assert wasserstein_from_counts(left, right) == 0.6
    assert ks_from_counts(left, right) == 0.6
    assert bootstrap_jsd_ci(left, right, iterations=100, seed=7) == bootstrap_jsd_ci(
        left, right, iterations=100, seed=7
    )


def test_realism_reports_all_submetrics_and_weight_sensitivity(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({
        "run_id": "r1",
        "scenario_name": "M1",
        "seed": 1,
        "distributions": {
            "travel_time_seconds": {"0-10": 5},
            "path_length_m": {"0-5": 5},
            "mean_speed_mps": {"1-1.2": 5},
            "region_visits": {"shop": 5},
            "slow_ratio": {"0-0.1": 5},
        },
    }), encoding="utf-8")
    reference = {
        "duration_seconds": {"0-10": 5},
        "path_length_m": {"0-5": 5},
        "mean_speed_mps": {"1-1.2": 5},
        "region_visits": {"shop": 5},
        "slow_ratio": {"0-0.1": 5},
    }
    config = {
        "fields": {
            key: {"weight": 0.2, "continuous": key != "region_visits"}
            for key in reference
        },
        "bootstrap": {"iterations": 20, "seed": 1, "confidence": 0.95},
        "weight_sensitivity": [0.5, 1.5],
    }
    result = evaluate_one(reference, metrics_path, config)
    assert result["weighted_jsd"] == 0
    assert result["realism_score"] == 1
    assert len(result["weight_sensitivity"]) == 10


def test_statistics_use_runs_and_pair_by_seed(tmp_path):
    paths = []
    for cell, values in {"B1": [1.0, 2.0, 3.0], "M1": [2.0, 3.0, 4.0]}.items():
        for seed, value in enumerate(values, 1):
            path = tmp_path / f"{cell}_{seed}.json"
            path.write_text(json.dumps({
                "run_id": f"{cell}-{seed}",
                "scenario_name": cell,
                "seed": seed,
                "matrix": {"cell_id": cell},
                "metrics": {"completion_rate": value},
            }), encoding="utf-8")
            paths.append(path)
    config = {
        "primary_metrics": ["completion_rate"],
        "exploratory_metrics": [],
        "baseline_cell": "B1",
        "comparison_cells": ["M1"],
        "confidence": 0.95,
        "bootstrap_iterations": 100,
        "random_seed": 4,
        "minimum_paired_runs": 5,
    }
    result = analyze(paths, config)
    comparison = result["paired_comparisons"][0]
    assert result["run_count"] == 6
    assert comparison["paired_n"] == 3
    assert comparison["mean_paired_difference"] == 1
    assert comparison["sample_size_warning"] is True
    assert bootstrap_mean_ci([1, 2, 3], 100, 0.95, 5) == bootstrap_mean_ci(
        [1, 2, 3], 100, 0.95, 5
    )


def test_holm_adjustment_is_monotonic():
    rows = [{"p_value": 0.01}, {"p_value": 0.03}, {"p_value": 0.04}]
    holm_adjust(rows)
    adjusted = [row["p_value_holm"] for row in rows]
    assert adjusted == sorted(adjusted)
