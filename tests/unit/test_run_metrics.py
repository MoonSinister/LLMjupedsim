import json
import sqlite3

import pytest

from jupedsim_mall.analysis.run_metrics import (
    detect_trajectory_schema,
    extract_run_metrics,
)


def build_run(tmp_path):
    trajectory = tmp_path / "trajectory.sqlite"
    with sqlite3.connect(trajectory) as connection:
        connection.execute(
            "CREATE TABLE trajectory_data "
            "(frame INTEGER, id INTEGER, pos_x REAL, pos_y REAL)"
        )
        connection.execute("CREATE TABLE metadata (key TEXT, value TEXT)")
        connection.execute("INSERT INTO metadata VALUES ('fps', '2')")
        connection.executemany(
            "INSERT INTO trajectory_data VALUES (?, ?, ?, ?)",
            [
                (0, 1, 0, 0), (2, 1, 1, 0), (4, 1, 2, 0),
                (0, 2, 0, 0), (2, 2, 0, 0),
            ],
        )
    plans = {
        "metadata": {"route_policy": "random"},
        "agents": [
            {"agent_id": "a1", "role": "visitor", "final_exit": "east", "profile": {"source": "mall"}},
            {"agent_id": "a2", "role": "staff", "final_exit": "west", "profile": {"source": "mall"}},
        ],
    }
    (tmp_path / "plans.json").write_text(json.dumps(plans), encoding="utf-8")
    events = [
        {"event_type": "agent_spawned", "agent_id": "a1", "simulation_agent_id": 1, "simulation_time": 0},
        {"event_type": "agent_spawned", "agent_id": "a2", "simulation_agent_id": 2, "simulation_time": 0},
        {"event_type": "agent_rerouted", "agent_id": "a1", "simulation_agent_id": 1, "simulation_time": 1},
        {"event_type": "agent_completed", "agent_id": "a1", "simulation_agent_id": 1, "simulation_time": 2},
        {"event_type": "agent_ttl_removed", "agent_id": "a2", "simulation_agent_id": 2, "simulation_time": 1},
    ]
    (tmp_path / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "run.log").write_text("ok", encoding="utf-8")
    (tmp_path / "resolved_scenario.json").write_text("{}", encoding="utf-8")
    artifacts = {
        "trajectory": str(trajectory),
        "plans": str(tmp_path / "plans.json"),
        "events": str(tmp_path / "events.jsonl"),
        "log": str(tmp_path / "run.log"),
        "resolved_scenario": str(tmp_path / "resolved_scenario.json"),
    }
    manifest = {
        "schema_version": "1.0",
        "run_id": "run-test",
        "scenario_name": "fixture",
        "seed": 1,
        "status": "completed",
        "returncode": 0,
        "artifacts": artifacts,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_extract_metrics_uses_events_and_agent_trajectories(tmp_path):
    result = extract_run_metrics(build_run(tmp_path), regions_path=tmp_path / "missing.json")
    assert result["spawned_agents"] == 2
    assert result["completed_agents"] == 1
    assert result["ttl_removed_agents"] == 1
    assert result["metrics"]["completion_rate"] == 0.5
    assert result["metrics"]["mean_path_length_m"] == 1.0
    rows = {row["agent_id"]: row for row in result["agent_rows"]}
    assert rows["a1"]["path_length_m"] == 2.0
    assert rows["a1"]["mean_speed_mps"] == 1.0
    assert rows["a1"]["reroute_extra_path_m"] == 0.0
    assert rows["a2"]["slow_ratio"] == 1.0


def test_schema_detection_reports_missing_trajectory_table(tmp_path):
    path = tmp_path / "empty.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE metadata (key TEXT, value TEXT)")
        with pytest.raises(ValueError, match="no supported trajectory table"):
            detect_trajectory_schema(connection)


def test_empty_trajectory_is_explicit_missing_data(tmp_path):
    manifest = build_run(tmp_path)
    with sqlite3.connect(tmp_path / "trajectory.sqlite") as connection:
        connection.execute("DELETE FROM trajectory_data")
    result = extract_run_metrics(manifest, regions_path=tmp_path / "missing.json")
    assert result["spawned_agents"] == 2
    assert result["metrics"]["mean_path_length_m"] is None
    assert all("path_length_m" not in row for row in result["agent_rows"])
