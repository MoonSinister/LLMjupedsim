import pytest

from jupedsim_mall.experiments.sim_schema import (
    AgentProfile,
    RunManifest,
    activity_from_mapping,
    migrate_payload,
    profile_from_mapping,
    route_from_mapping,
    scenario_from_mapping,
)


def test_profile_normalization_clamps_values_and_preserves_metadata():
    profile = profile_from_mapping(
        {
            "profile_id": "p1",
            "role": "UNKNOWN",
            "desired_speed_mps": 9,
            "default_wait_seconds": -4,
            "custom": "value",
        }
    )

    assert isinstance(profile, AgentProfile)
    assert profile.role == "visitor"
    assert profile.desired_speed_mps == 1.8
    assert profile.default_wait_seconds == 0
    assert profile.metadata == {"custom": "value"}


def test_activity_requires_region():
    assert activity_from_mapping({"action": "wait"}) is None


def test_route_normalization_uses_stable_nested_contracts():
    route = route_from_mapping(
        {
            "agent_id": "a1",
            "exit_idx": 2,
            "activities": [{"region": "region_1", "wait_seconds": 12}],
            "planner": "llm_semantic",
        }
    )

    assert route.exit_idx == 2
    assert route.activities[0].region == "region_1"
    assert route.planner == "llm_semantic"
    assert route.to_json()["schema_version"] == "1.0"


def test_scenario_normalizes_cli_values_to_strings():
    scenario = scenario_from_mapping(
        {"name": "demo", "description": "demo", "args": ["-n", 5], "env": {"A": 1}}
    )

    assert scenario.args == ["-n", "5"]
    assert scenario.env == {"A": "1"}


def test_manifest_rejects_unknown_status():
    with pytest.raises(ValueError, match="invalid run status"):
        RunManifest(run_id="r1", scenario_name="demo", seed=1, status="mystery")


def test_legacy_payload_migration_adds_current_version():
    migrated = migrate_payload("scenario", {"name": "demo", "description": "demo", "args": []})
    assert migrated["schema_version"] == "1.0"
    assert migrated["enabled"] is True


def test_unknown_migration_kind_is_rejected():
    with pytest.raises(ValueError, match="unsupported payload kind"):
        migrate_payload("mystery", {})
