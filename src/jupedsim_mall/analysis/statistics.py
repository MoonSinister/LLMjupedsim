"""Across-run paired statistics with deterministic bootstrap and Holm correction."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import random
from collections import defaultdict
from collections.abc import Sequence
from statistics import mean, median, stdev

from jupedsim_mall.project import PROJECT_ROOT

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover - exercised only in minimal installs
    scipy_stats = None

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "analysis" / "statistics.json"


def bootstrap_mean_ci(values: list[float], iterations: int, confidence: float, seed: int) -> tuple[float, float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    estimates = sorted(
        mean(rng.choice(values) for _ in values)
        for _ in range(iterations)
    )
    alpha = (1 - confidence) / 2
    low = estimates[max(0, int(alpha * len(estimates)))]
    high = estimates[min(len(estimates) - 1, int((1 - alpha) * len(estimates)) - 1)]
    return round(low, 6), round(high, 6)


def describe(values: list[float], config: dict) -> dict:
    if not values:
        return {"n": 0, "mean": None, "std": None, "median": None, "q1": None, "q3": None, "iqr": None, "ci_low": None, "ci_high": None}
    ordered = sorted(values)
    q1 = ordered[int(0.25 * (len(ordered) - 1))]
    q3 = ordered[int(0.75 * (len(ordered) - 1))]
    ci = bootstrap_mean_ci(
        values,
        int(config["bootstrap_iterations"]),
        float(config["confidence"]),
        int(config["random_seed"]),
    )
    return {
        "n": len(values),
        "mean": round(mean(values), 6),
        "std": round(stdev(values), 6) if len(values) > 1 else 0.0,
        "median": round(median(values), 6),
        "q1": round(q1, 6),
        "q3": round(q3, 6),
        "iqr": round(q3 - q1, 6),
        "ci_low": ci[0] if ci else None,
        "ci_high": ci[1] if ci else None,
    }


def _paired_permutation(differences: list[float], seed: int, iterations: int = 10000) -> float:
    if not differences:
        return 1.0
    observed = abs(mean(differences))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(iterations):
        statistic = abs(mean(value * rng.choice((-1, 1)) for value in differences))
        extreme += statistic >= observed
    return (extreme + 1) / (iterations + 1)


def paired_test(left: list[float], right: list[float], seed: int) -> dict:
    differences = [b - a for a, b in zip(left, right)]
    n = len(differences)
    if n < 2:
        return {"test": "insufficient", "p_value": None, "effect_size": None, "effect_name": None}
    normal = False
    difference_spread = stdev(differences)
    if scipy_stats is not None and n >= 3 and difference_spread > 1e-12:
        normal = bool(scipy_stats.shapiro(differences).pvalue >= 0.05)
    if normal and scipy_stats is not None:
        result = scipy_stats.ttest_rel(right, left)
        test_name = "paired_t"
        p_value = float(result.pvalue)
        effect = mean(differences) / difference_spread if difference_spread > 0 else 0.0
        effect_name = "cohen_dz"
    else:
        test_name = "paired_wilcoxon" if scipy_stats is not None else "paired_sign_permutation"
        if scipy_stats is not None:
            try:
                p_value = float(scipy_stats.wilcoxon(differences).pvalue)
            except ValueError:
                p_value = 1.0
        else:
            p_value = _paired_permutation(differences, seed)
        positive = sum(value > 0 for value in differences)
        negative = sum(value < 0 for value in differences)
        effect = (positive - negative) / max(positive + negative, 1)
        effect_name = "rank_biserial"
    return {
        "test": test_name,
        "p_value": round(p_value, 8),
        "effect_size": round(effect, 6),
        "effect_name": effect_name,
        "mean_paired_difference": round(mean(differences), 6),
    }


def holm_adjust(rows: list[dict]) -> None:
    indexed = sorted(
        [(index, row["p_value"]) for index, row in enumerate(rows) if row.get("p_value") is not None],
        key=lambda item: item[1],
    )
    running = 0.0
    total = len(indexed)
    for rank, (index, p_value) in enumerate(indexed):
        adjusted = min(1.0, (total - rank) * p_value)
        running = max(running, adjusted)
        rows[index]["p_value_holm"] = round(running, 8)
        rows[index]["significant_holm_0_05"] = running < 0.05


def discover_metric_paths(paths: Sequence[str]) -> list[pathlib.Path]:
    found = []
    for value in paths:
        path = pathlib.Path(value)
        if path.is_file() and path.name == "metrics.json":
            found.append(path)
        elif path.is_dir():
            if (path / "metrics.json").is_file():
                found.append(path / "metrics.json")
            else:
                found.extend(path.rglob("metrics.json"))
    return sorted(set(item.resolve() for item in found))


def analyze(metric_paths: list[pathlib.Path], config: dict, plan_path: pathlib.Path | None = None) -> dict:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in metric_paths]
    all_metrics = [*config["primary_metrics"], *config["exploratory_metrics"]]
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_cell_seed: dict[tuple[str, int, str], float] = {}
    for payload in payloads:
        cell = payload.get("matrix", {}).get("cell_id") or payload["scenario_name"]
        for metric in all_metrics:
            value = payload.get("metrics", {}).get(metric)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                grouped[(cell, metric)].append(float(value))
                by_cell_seed[(cell, int(payload["seed"]), metric)] = float(value)
    aggregates = []
    for (cell, metric), values in sorted(grouped.items()):
        aggregates.append({
            "cell_id": cell,
            "metric": metric,
            "classification": "primary" if metric in config["primary_metrics"] else "exploratory",
            **describe(values, config),
            "sample_size_warning": len(values) < int(config["minimum_paired_runs"]),
        })
    comparisons = []
    baseline = config["baseline_cell"]
    for comparison in config["comparison_cells"]:
        for metric in all_metrics:
            seeds = sorted(
                seed for cell, seed, name in by_cell_seed
                if cell == baseline and name == metric and (comparison, seed, metric) in by_cell_seed
            )
            left = [by_cell_seed[(baseline, seed, metric)] for seed in seeds]
            right = [by_cell_seed[(comparison, seed, metric)] for seed in seeds]
            result = paired_test(left, right, int(config["random_seed"]))
            comparisons.append({
                "baseline": baseline,
                "comparison": comparison,
                "metric": metric,
                "classification": "primary" if metric in config["primary_metrics"] else "exploratory",
                "paired_n": len(seeds),
                "paired_seeds": seeds,
                "sample_size_warning": len(seeds) < int(config["minimum_paired_runs"]),
                **result,
            })
    holm_adjust(comparisons)
    failure_sensitivity = None
    if plan_path and plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        statuses: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for item in plan["runs"]:
            statuses[item["cell_id"]][item["status"]] += 1
        failure_sensitivity = {
            cell: {
                "planned": sum(counts.values()),
                "completed": counts["completed"],
                "failed_or_missing": sum(count for status, count in counts.items() if status != "completed"),
                "complete_case_fraction": round(counts["completed"] / max(sum(counts.values()), 1), 6),
                "interpretation": "Report complete-case estimates with planned-vs-completed counts; do not impute favorable outcomes.",
            }
            for cell, counts in statuses.items()
        }
    return {
        "schema_version": "1.0",
        "config": config,
        "metric_files": [str(path) for path in metric_paths],
        "run_count": len(payloads),
        "aggregates": aggregates,
        "paired_comparisons": comparisons,
        "failure_sensitivity": failure_sensitivity,
    }


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row if not isinstance(row.get(key), (list, dict))})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def _write_markdown(path: pathlib.Path, result: dict) -> None:
    lines = [
        "# Statistical Analysis",
        "",
        f"Independent run count: {result['run_count']}. Statistical unit: run/seed.",
        "",
        "## Aggregates",
        "",
        "| Cell | Metric | Class | n | Mean | SD | 95% CI | Warning |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in result["aggregates"]:
        lines.append(
            f"| {row['cell_id']} | {row['metric']} | {row['classification']} | {row['n']} | "
            f"{row['mean']} | {row['std']} | [{row['ci_low']}, {row['ci_high']}] | "
            f"{'low sample size' if row['sample_size_warning'] else ''} |"
        )
    lines.extend(["", "## Paired Comparisons", "", "| Baseline | Comparison | Metric | n | Test | Effect | Holm p |", "| --- | --- | --- | ---: | --- | ---: | ---: |"])
    for row in result["paired_comparisons"]:
        lines.append(
            f"| {row['baseline']} | {row['comparison']} | {row['metric']} | {row['paired_n']} | "
            f"{row['test']} | {row['effect_size']} ({row['effect_name']}) | {row.get('p_value_holm')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="metrics.json files or directories containing them.")
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--plan", type=pathlib.Path)
    parser.add_argument("--output-dir", type=pathlib.Path, default=PROJECT_ROOT / "outputs" / "statistics")
    args = parser.parse_args(argv)
    paths = discover_metric_paths(args.paths)
    if not paths:
        raise SystemExit("No metrics.json files found.")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = analyze(paths, config, args.plan)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "statistics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(args.output_dir / "aggregates.csv", result["aggregates"])
    _write_csv(args.output_dir / "paired_comparisons.csv", result["paired_comparisons"])
    _write_markdown(args.output_dir / "statistics.md", result)
    print(f"Analyzed {result['run_count']} independent run(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
