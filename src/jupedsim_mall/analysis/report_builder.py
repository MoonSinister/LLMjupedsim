"""Generate paper figures, tables, and a report from standardized analysis tables."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import shutil
from collections import Counter, defaultdict
from collections.abc import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jupedsim_mall.project import PROJECT_ROOT

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "analysis" / "report.json"


def read_csv(path: pathlib.Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def configure_style(style: dict) -> None:
    plt.rcParams.update({
        "font.family": style["font_family"],
        "figure.figsize": (style["figure_width_in"], style["figure_height_in"]),
        "figure.dpi": style["dpi"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
    })


def save_figure(fig, stem: pathlib.Path, style: dict, rows: list[dict], spec: dict) -> dict:
    stem.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for extension in style["formats"]:
        path = stem.with_suffix("." + extension)
        fig.savefig(path, dpi=style["dpi"], bbox_inches="tight")
        files.append(str(path))
    plt.close(fig)
    write_csv(stem.with_name(stem.name + "_source.csv"), rows)
    stem.with_name(stem.name + "_plot.json").write_text(
        json.dumps({"style": style, **spec}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"figure": stem.name, "files": files, "source_rows": len(rows)}


def primary_figures(aggregates: list[dict], config: dict, output: pathlib.Path) -> list[dict]:
    results = []
    style, colors = config["style"], config["style"]["colors"]
    for metric, label in config["primary_metrics"].items():
        rows = [row for row in aggregates if row.get("metric") == metric and row.get("mean") not in ("", None)]
        if not rows:
            continue
        values = [float(row["mean"]) for row in rows]
        lower = [value - float(row["ci_low"]) for value, row in zip(values, rows)]
        upper = [float(row["ci_high"]) - value for value, row in zip(values, rows)]
        fig, ax = plt.subplots()
        ax.bar([row["cell_id"] for row in rows], values, color=[colors[i % len(colors)] for i in range(len(rows))])
        ax.errorbar([row["cell_id"] for row in rows], values, yerr=[lower, upper], fmt="none", color="#222222", capsize=4)
        ax.set_title(label + " by method")
        ax.set_ylabel(label)
        for index, row in enumerate(rows):
            ax.text(index, values[index], "n=" + str(row["n"]), ha="center", va="bottom", fontsize=8)
        results.append(save_figure(fig, output / ("primary_" + metric), style, rows, {"kind": "bar_ci", "sample_unit": "run/seed"}))
    return results


def distribution_figures(metric_paths: list[pathlib.Path], config: dict, output: pathlib.Path) -> list[dict]:
    grouped = {"exit_distribution": defaultdict(Counter), "region_visits": defaultdict(Counter)}
    for path in metric_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cell = payload.get("matrix", {}).get("cell_id") or payload["scenario_name"]
        grouped["exit_distribution"][cell].update(payload.get("distributions", {}).get("exit", {}))
        grouped["region_visits"][cell].update(payload.get("distributions", {}).get("region_visits", {}))
    results = []
    for figure_name, values in grouped.items():
        categories = sorted({category for counts in values.values() for category in counts})
        cells = sorted(values)
        if not categories or not cells:
            continue
        bottom = np.zeros(len(cells))
        rows = []
        fig, ax = plt.subplots()
        for index, category in enumerate(categories):
            counts = np.asarray([values[cell].get(category, 0) for cell in cells])
            ax.bar(cells, counts, bottom=bottom, label=category, color=config["style"]["colors"][index % len(config["style"]["colors"])])
            bottom += counts
            rows.extend({"cell_id": cell, "category": category, "count": int(count)} for cell, count in zip(cells, counts))
        ax.set_title(figure_name.replace("_", " ").title())
        ax.set_ylabel("Count")
        ax.legend(fontsize=7, ncol=2)
        results.append(save_figure(fig, output / figure_name, config["style"], rows, {"kind": "stacked_bar"}))
    return results


def realism_figure(path: pathlib.Path | None, config: dict, output: pathlib.Path) -> list[dict]:
    if path is None or not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for result in payload.get("results", []):
        for field, metrics in result["fields"].items():
            if metrics["jsd"] is not None:
                rows.append({"scenario": result["scenario"], "field": field, "jsd": metrics["jsd"], "run_id": result.get("run_id")})
    if not rows:
        return []
    scenarios = sorted({row["scenario"] for row in rows})
    fields = config["realism_metrics"]
    width = 0.8 / len(scenarios)
    x = np.arange(len(fields))
    fig, ax = plt.subplots()
    for index, scenario in enumerate(scenarios):
        values = [
            np.mean([float(row["jsd"]) for row in rows if row["scenario"] == scenario and row["field"] == field])
            if any(row["scenario"] == scenario and row["field"] == field for row in rows) else np.nan
            for field in fields
        ]
        ax.bar(x + index * width, values, width, label=scenario, color=config["style"]["colors"][index % len(config["style"]["colors"])])
    ax.set_xticks(x + width * (len(scenarios) - 1) / 2, fields, rotation=20)
    ax.set_ylabel("Jensen-Shannon divergence")
    ax.set_title("ATC realism submetrics")
    ax.legend(fontsize=8)
    return [save_figure(fig, output / "realism_submetrics", config["style"], rows, {"kind": "grouped_bar"})]


def trajectory_figures(metric_paths: list[pathlib.Path], config: dict, output: pathlib.Path) -> list[dict]:
    rows = [row for metric in metric_paths for row in read_csv(metric.with_name("trajectory_points.csv"))]
    if not rows:
        return []
    fig, ax = plt.subplots()
    image = ax.hist2d([float(row["x_m"]) for row in rows], [float(row["y_m"]) for row in rows], bins=50, cmap="viridis")
    fig.colorbar(image[3], ax=ax, label="Observed point count")
    ax.set(xlabel="x (m)", ylabel="y (m)", title="Trajectory occupancy density")
    density = save_figure(fig, output / "trajectory_density", config["style"], [{"run_id": row["run_id"], "x_m": row["x_m"], "y_m": row["y_m"]} for row in rows], {"kind": "hist2d", "bins": 50})
    selected = sorted({(row["run_id"], row["agent_id"]) for row in rows})[:8]
    source = []
    fig, ax = plt.subplots()
    for index, key in enumerate(selected):
        points = sorted([row for row in rows if (row["run_id"], row["agent_id"]) == key], key=lambda row: float(row["time_seconds"]))
        ax.plot([float(row["x_m"]) for row in points], [float(row["y_m"]) for row in points], label=key[1], color=config["style"]["colors"][index % len(config["style"]["colors"])])
        source.extend(points)
    ax.set(xlabel="x (m)", ylabel="y (m)", title="Representative trajectories")
    ax.legend(fontsize=7)
    traces = save_figure(fig, output / "representative_trajectories", config["style"], source, {"kind": "line", "selection": "first eight stable run-agent identifiers"})
    density_by_agent = {}
    for metric in metric_paths:
        for row in read_csv(metric.with_name("agent_metrics.csv")):
            value = row.get("mean_local_density_per_m2")
            if value not in ("", None):
                density_by_agent[(row["run_id"], row["agent_id"])] = float(value)
    weighted = [
        {**row, "density_weight": density_by_agent.get((row["run_id"], row["agent_id"]), 0.0)}
        for row in rows
    ]
    fig, ax = plt.subplots()
    heat = ax.hist2d(
        [float(row["x_m"]) for row in weighted],
        [float(row["y_m"]) for row in weighted],
        bins=50,
        weights=[float(row["density_weight"]) for row in weighted],
        cmap="magma",
    )
    fig.colorbar(heat[3], ax=ax, label="Accumulated local density")
    ax.set(xlabel="x (m)", ylabel="y (m)", title="Congestion exposure heatmap")
    congestion = save_figure(fig, output / "congestion_heatmap", config["style"], weighted, {"kind": "weighted_hist2d", "bins": 50, "weight": "local_density_per_m2"})
    return [density, congestion, traces]


def sensitivity_figures(metric_paths: list[pathlib.Path], config: dict, output: pathlib.Path) -> list[dict]:
    rows = []
    for path in metric_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        factors = payload.get("matrix", {}).get("factors", {})
        if "parameter" not in factors or "level" not in factors:
            continue
        for metric in config["primary_metrics"]:
            value = payload.get("metrics", {}).get(metric)
            if isinstance(value, (int, float)):
                rows.append({
                    "parameter": factors["parameter"],
                    "level": factors["level"],
                    "metric": metric,
                    "value": value,
                    "seed": payload["seed"],
                })
    figures = []
    for parameter in sorted({row["parameter"] for row in rows}):
        parameter_rows = [row for row in rows if row["parameter"] == parameter]
        fig, ax = plt.subplots()
        for index, metric in enumerate(sorted({row["metric"] for row in parameter_rows})):
            metric_rows = [row for row in parameter_rows if row["metric"] == metric]
            grouped = defaultdict(list)
            for row in metric_rows:
                grouped[str(row["level"])].append(float(row["value"]))
            levels = sorted(grouped, key=lambda value: float(value) if value.replace(".", "", 1).isdigit() else value)
            means = [float(np.mean(grouped[level])) for level in levels]
            errors = [float(np.std(grouped[level])) for level in levels]
            ax.errorbar(levels, means, yerr=errors, marker="o", label=metric, color=config["style"]["colors"][index % len(config["style"]["colors"])])
        ax.set(title="Sensitivity: " + parameter, xlabel=parameter, ylabel="Metric value")
        ax.legend(fontsize=7)
        figures.append(save_figure(fig, output / ("sensitivity_" + parameter), config["style"], parameter_rows, {"kind": "line_ci", "parameter": parameter}))
    return figures


def state_figures(state_dir: pathlib.Path | None, config: dict, output: pathlib.Path) -> list[dict]:
    if state_dir is None:
        return []
    transitions = read_csv(state_dir / "state_transitions.csv")
    model_path = state_dir / "behavior_state_model.json"
    if not transitions or not model_path.is_file():
        return []
    states = sorted({row["from_state"] for row in transitions} | {row["to_state"] for row in transitions})
    lookup = {state: index for index, state in enumerate(states)}
    matrix = np.zeros((len(states), len(states)))
    for row in transitions:
        matrix[lookup[row["from_state"]], lookup[row["to_state"]]] += float(row["count"])
    fig, ax = plt.subplots()
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Transition count")
    ax.set_xticks(range(len(states)), states, rotation=30)
    ax.set_yticks(range(len(states)), states)
    ax.set_title("Behavior state transitions")
    transition_figure = save_figure(fig, output / "state_transitions", config["style"], transitions, {"kind": "heatmap", "states": states})
    model = json.loads(model_path.read_text(encoding="utf-8"))
    dwell = [{"state": state, "seconds": seconds} for state, seconds in model["state_dwell_seconds"].items()]
    fig, ax = plt.subplots()
    ax.bar([row["state"] for row in dwell], [row["seconds"] for row in dwell], color=config["style"]["colors"][:len(dwell)])
    ax.set(title="Behavior state dwell", ylabel="Window-derived dwell (s)")
    dwell_figure = save_figure(fig, output / "state_dwell", config["style"], dwell, {"kind": "bar"})
    return [transition_figure, dwell_figure]


def export_tables(statistics_dir: pathlib.Path, output: pathlib.Path) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for name in ("aggregates", "paired_comparisons"):
        source = statistics_dir / (name + ".csv")
        rows = read_csv(source)
        if not rows:
            continue
        shutil.copy2(source, output / source.name)
        fields = list(rows[0])
        markdown = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
        markdown.extend("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |" for row in rows)
        (output / (name + ".md")).write_text("\n".join(markdown) + "\n", encoding="utf-8")
        latex = ["\\begin{tabular}{" + "l" * len(fields) + "}", "\\hline", " & ".join(fields) + " \\\\", "\\hline"]
        latex.extend(" & ".join(str(row.get(field, "")).replace("_", "\\_") for field in fields) + " \\\\" for row in rows)
        latex.extend(["\\hline", "\\end{tabular}"])
        (output / (name + ".tex")).write_text("\n".join(latex) + "\n", encoding="utf-8")
        files.extend(str(output / (name + extension)) for extension in (".csv", ".md", ".tex"))
    return files


def build_report(config_path: pathlib.Path, metric_paths: list[pathlib.Path], statistics_dir: pathlib.Path, realism_path: pathlib.Path | None, state_dir: pathlib.Path | None, matrix_status: pathlib.Path | None, output_dir: pathlib.Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    configure_style(config["style"])
    figure_dir = output_dir / "figures"
    figures = [
        *primary_figures(read_csv(statistics_dir / "aggregates.csv"), config, figure_dir),
        *distribution_figures(metric_paths, config, figure_dir),
        *realism_figure(realism_path, config, figure_dir),
        *trajectory_figures(metric_paths, config, figure_dir),
        *sensitivity_figures(metric_paths, config, figure_dir),
        *state_figures(state_dir, config, figure_dir),
    ]
    tables = export_tables(statistics_dir, output_dir / "tables")
    matrix = json.loads(matrix_status.read_text(encoding="utf-8")) if matrix_status and matrix_status.is_file() else {}
    lines = [
        "# " + config["title"], "",
        "Generated from standardized analysis tables; figures are not manually edited.", "",
        "## Run Status", "",
        "- Matrix counts: " + str(matrix.get("counts", "not supplied")),
        "- Planned sample size satisfied: " + str(matrix.get("sample_size_satisfied", "unknown")),
        "- Metric runs included: " + str(len(metric_paths)), "",
        "## Figures", "",
    ]
    for figure in figures:
        png = next((path for path in figure["files"] if path.endswith(".png")), figure["files"][0])
        relative = pathlib.Path(png).relative_to(output_dir).as_posix()
        lines.extend(["### " + figure["figure"], "", "![" + figure["figure"] + "](" + relative + ")", "", "Source rows: " + str(figure["source_rows"]) + ".", ""])
    lines.extend(["## Tables", ""])
    lines.extend("- " + str(pathlib.Path(path).relative_to(output_dir)) for path in tables)
    lines.extend([
        "", "## Exceptions And Limitations", "",
        "- ATC and simulated pedestrians are compared as distributions, not one-to-one trajectories.",
        "- LLM outputs may vary across model versions; prompt and response hashes are retained.",
        "- Results generalize only to the validated map, population assumptions, and parameter ranges.",
        "- Failed and missing runs remain visible and are not silently discarded.",
    ])
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "experiment_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    provenance = {
        "schema_version": "1.0", "config": str(config_path),
        "metric_files": [str(path) for path in metric_paths],
        "statistics_dir": str(statistics_dir), "realism": str(realism_path) if realism_path else None,
        "state_dir": str(state_dir) if state_dir else None,
        "matrix_status": str(matrix_status) if matrix_status else None,
        "figures": figures, "tables": tables, "report": str(report_path),
    }
    (output_dir / "report_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    return provenance


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="+", type=pathlib.Path)
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--statistics-dir", type=pathlib.Path, required=True)
    parser.add_argument("--realism", type=pathlib.Path)
    parser.add_argument("--states", type=pathlib.Path)
    parser.add_argument("--matrix-status", type=pathlib.Path)
    parser.add_argument("--output-dir", type=pathlib.Path, default=PROJECT_ROOT / "outputs" / "report")
    args = parser.parse_args(argv)
    metric_paths = [path / "metrics.json" if path.is_dir() else path for path in args.metrics if (path / "metrics.json").is_file() or path.is_file()]
    result = build_report(args.config, metric_paths, args.statistics_dir, args.realism, args.states, args.matrix_status, args.output_dir)
    print("Report: " + result["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
