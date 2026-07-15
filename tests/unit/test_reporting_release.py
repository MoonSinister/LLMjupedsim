import csv
import json

import pytest

from jupedsim_mall.analysis.report_builder import build_report
from jupedsim_mall.experiments.experiment_control import finalize, preflight
from jupedsim_mall.release_builder import build, verify
from jupedsim_mall.pipeline import main as pipeline_main


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_report_and_release_are_rebuildable_and_hash_verified(tmp_path):
    statistics = tmp_path / "statistics"
    _write_csv(statistics / "aggregates.csv", [{
        "cell_id": "B1", "metric": "completion_rate", "n": 2,
        "mean": 0.8, "std": 0.1, "median": 0.8, "q1": 0.75,
        "q3": 0.85, "ci_low": 0.7, "ci_high": 0.9,
    }])
    metric_dir = tmp_path / "run"
    metric_dir.mkdir()
    metrics = metric_dir / "metrics.json"
    metrics.write_text(json.dumps({
        "scenario_name": "smoke", "seed": 1,
        "matrix": {"cell_id": "B1"},
        "distributions": {"exit": {"exit_1": 2}, "region_visits": {"region_1": 3}},
        "metrics": {"completion_rate": 0.8},
    }), encoding="utf-8")
    _write_csv(metric_dir / "trajectory_points.csv", [
        {"run_id": "r1", "agent_id": "a1", "time_seconds": 0, "x_m": 0, "y_m": 0},
        {"run_id": "r1", "agent_id": "a1", "time_seconds": 1, "x_m": 1, "y_m": 1},
    ])
    _write_csv(metric_dir / "agent_metrics.csv", [{
        "run_id": "r1", "agent_id": "a1", "mean_local_density_per_m2": 0.5,
    }])
    config = tmp_path / "report.json"
    config.write_text(json.dumps({
        "title": "Test report",
        "style": {"font_family": "DejaVu Sans", "figure_width_in": 4,
                  "figure_height_in": 3, "dpi": 40, "formats": ["png"],
                  "colors": ["#276FBF", "#D1495B", "#2A9D8F"]},
        "primary_metrics": {"completion_rate": "Completion rate"},
        "realism_metrics": ["duration_seconds"],
    }), encoding="utf-8")
    report_dir = tmp_path / "report"
    result = build_report(config, [metrics], statistics, None, None, None, report_dir)
    assert (report_dir / "experiment_report.md").is_file()
    assert result["figures"]
    assert all((report_dir / "figures" / (item["figure"] + "_source.csv")).is_file() for item in result["figures"])

    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"runs": [{
        "plan_item_id": "p1", "cell_id": "B1", "seed": 1,
        "status": "pending", "attempts": [],
    }]}), encoding="utf-8")
    release = tmp_path / "release"
    build(release, plan, report_dir)
    assert verify(release)
    assert any((release / "summaries").iterdir())
    (release / "limitations.md").write_text("changed", encoding="utf-8")
    assert not verify(release)


def test_formal_controls_fail_closed(tmp_path):
    matrix = tmp_path / "matrix.json"
    matrix.write_text("{}", encoding="utf-8")
    quality = tmp_path / "quality.json"
    quality.write_text(json.dumps({"passed": True, "mode": "quick"}), encoding="utf-8")
    result = preflight(matrix, quality, False)
    assert not result["passed"]
    assert any(check["name"] == "full_quality_gate" and not check["passed"] for check in result["checks"])

    snapshot = tmp_path / "FORMAL_SNAPSHOT.json"
    snapshot.write_text(json.dumps({"status": "frozen_for_execution"}), encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "matrix_fingerprint": "abc", "runs": [{"status": "pending", "attempts": []}],
    }), encoding="utf-8")
    report = tmp_path / "report.md"
    report.write_text("draft", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot finalize"):
        finalize(snapshot, plan, report)


def test_pipeline_refuses_incomplete_matrix(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "matrix_name": "test", "matrix_fingerprint": "abc",
        "runs": [{"plan_item_id": "p1", "status": "pending", "attempts": []}],
    }), encoding="utf-8")
    reference = tmp_path / "reference.json"
    reference.write_text("{}", encoding="utf-8")
    output = tmp_path / "pipeline"
    assert pipeline_main([str(plan), "--reference", str(reference), "--output-dir", str(output)]) == 1
    status = json.loads((output / "pipeline_status.json").read_text(encoding="utf-8"))
    assert status["passed"] is False
