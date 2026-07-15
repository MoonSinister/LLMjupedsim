import json
from types import SimpleNamespace

import pytest

from jupedsim_mall.simulation.event_recorder import EventRecorder
from jupedsim_mall.simulation.movement import MovementConfig


def test_event_recorder_writes_versioned_jsonl(tmp_path):
    output = tmp_path / "events.jsonl"
    with EventRecorder(output, run_id="run-1", simulation_dt=0.02) as recorder:
        recorder.record("agent_spawned", iteration=25, agent_id="a1")
    event = json.loads(output.read_text(encoding="utf-8"))
    assert event["schema_version"] == "1.0"
    assert event["run_id"] == "run-1"
    assert event["simulation_time"] == 0.5


def test_movement_config_rejects_non_positive_parameters():
    args = SimpleNamespace(
        movement_model="cfsv3", agent_radius=0, agent_time_gap=0.7,
        neighbor_repulsion_strength=8, neighbor_repulsion_range=0.12,
        geometry_repulsion_strength=4, geometry_repulsion_range=0.03,
        cfsv3_range_x_scale=20, cfsv3_range_y_scale=8, cfsv3_theta_max=1.35,
        cfsv3_agent_buffer=0, avm_wall_buffer_distance=0.08,
        avm_anticipation_time=1, avm_reaction_time=0.3,
    )
    with pytest.raises(ValueError, match="agent_radius"):
        MovementConfig.from_args(args)
