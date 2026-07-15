import json
import urllib.error

from shapely.geometry import Polygon

from jupedsim_mall.planning.route_planner import (
    BaselineRoutePlanner,
    LLMPlannerConfig,
    LLMExitOnlyRoutePlanner,
    LLMSemanticRoutePlanner,
    OpenAICompatibleClient,
    ReplayRoutePlanner,
    extract_json_object,
    validate_route,
    validate_llm_response,
)


def square(x: float, y: float) -> Polygon:
    return Polygon([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)])


def spawn_plans(count: int = 2):
    return [{
        "label": "entrance_0",
        "polygon": square(0, 0),
        "candidate_exit_indices": [0, 1],
        "agent_queue": [
            {
                "agent_id": f"agent_{index + 1:04d}",
                "spawn_order": index + 1,
                "scheduled_iteration": index,
                "profile": {
                    "role": "visitor",
                    "subtype": "explorer",
                    "desired_speed_mps": 1.2,
                    "default_wait_seconds": 5,
                },
                "route": None,
            }
            for index in range(count)
        ],
    }]


EXITS = [(square(10, 0), "exit_0"), (square(20, 0), "exit_1")]
REGIONS = [{"name": "region_1", "description": "shop", "position": (5.0, 0.0), "area": 2.0}]


def test_baseline_planners_share_contract_and_nearest_is_geometric():
    plans = spawn_plans()
    result = BaselineRoutePlanner("nearest", seed=2026).plan(plans, EXITS, REGIONS)

    assert result.policy == "baseline_nearest"
    assert result.planned_agents == 2
    assert {agent["route"]["exit_idx"] for agent in plans[0]["agent_queue"]} == {0}
    assert all(agent["route"]["planner"] == "baseline_nearest" for agent in plans[0]["agent_queue"])


def test_route_validation_rejects_unknown_region_and_exit():
    errors = validate_route(
        {"exit_idx": 9, "regions": ["missing"], "activities": [], "desired_speed_mps": 1.2},
        {0, 1},
        2,
        {"region_1"},
    )
    assert "invalid exit_idx" in errors
    assert "unknown region: missing" in errors


def test_replay_planner_applies_saved_routes(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps({
            "schema_version": "1.0",
            "metadata": {"route_policy": "baseline_random"},
            "agents": [
                {"agent_id": "agent_0001", "exit_idx": 0, "regions": [], "activities": []},
                {"agent_id": "agent_0002", "exit_idx": 1, "regions": [], "activities": []},
            ],
        }),
        encoding="utf-8",
    )
    plans = spawn_plans()
    result = ReplayRoutePlanner(path).plan(plans, EXITS, REGIONS)

    assert result.planned_agents == 2
    assert result.metadata["source_route_policy"] == "baseline_random"
    assert [agent["route"]["exit_idx"] for agent in plans[0]["agent_queue"]] == [0, 1]


def test_llm_planner_validates_fake_response_without_network():
    class FakeClient:
        def complete(self, messages):
            return json.dumps({
                "plans": [
                    {
                        "agent_id": "agent_0001",
                        "final_exit": "exit_1",
                        "regions": ["region_1"],
                        "intent": "visit a shop",
                        "desired_speed_mps": 1.2,
                    }
                ]
            }), {"cache_hit": True}

    planner = LLMSemanticRoutePlanner(LLMPlannerConfig("http://unused", "mock"))
    planner.client = FakeClient()
    plans = spawn_plans()
    result = planner.plan(plans, EXITS, REGIONS)

    assert result.planned_agents == 1
    assert result.fallback_agents == 1
    assert plans[0]["agent_queue"][0]["route"]["regions"] == ["region_1"]
    assert plans[0]["agent_queue"][1]["route"]["fallback_reason"]


def test_exit_only_planner_forces_zero_semantic_regions():
    planner = LLMExitOnlyRoutePlanner(
        LLMPlannerConfig("http://unused", "mock", max_regions_per_agent=3)
    )
    assert planner.policy == "llm_exit_only"
    assert planner.config.max_regions_per_agent == 0


def test_llm_client_reads_precomputed_cache_and_records_call(tmp_path):
    config = LLMPlannerConfig(
        "http://unused",
        "mock",
        cache_dir=str(tmp_path / "cache"),
        calls_output=str(tmp_path / "calls.jsonl"),
    )
    client = OpenAICompatibleClient(config)
    messages = [{"role": "user", "content": "test"}]
    key = client._request_key(messages)
    cache_path = tmp_path / "cache" / f"{key}.json"
    cache_path.parent.mkdir()
    cache_path.write_text(json.dumps({"content": "{\"plans\": []}"}), encoding="utf-8")

    content, record = client.complete(messages)
    assert extract_json_object(content) == {"plans": []}
    assert record["cache_hit"] is True
    assert json.loads((tmp_path / "calls.jsonl").read_text(encoding="utf-8"))["request_key"] == key


def test_llm_response_schema_rejects_missing_exit():
    try:
        validate_llm_response({"plans": [{"agent_id": "agent_0001"}]})
    except ValueError as exc:
        assert "schema" in str(exc)
    else:
        raise AssertionError("invalid response should be rejected")


def test_llm_timeout_is_recorded_without_remote_network():
    class TimeoutClient:
        def complete(self, _messages):
            raise urllib.error.URLError("mock timeout")

    planner = LLMSemanticRoutePlanner(LLMPlannerConfig("http://unused", "mock"))
    planner.client = TimeoutClient()
    plans = spawn_plans()
    result = planner.plan(plans, EXITS, REGIONS)
    assert result.failed_batches == 1
    assert result.planned_agents == 0


def test_partial_response_rejects_invalid_region_without_rejecting_valid_exit():
    class PartialClient:
        def complete(self, _messages):
            return json.dumps({
                "plans": [
                    {
                        "agent_id": "agent_0001",
                        "final_exit": "exit_0",
                        "regions": ["missing_region"],
                    },
                    {
                        "agent_id": "agent_0002",
                        "final_exit": "exit_1",
                        "regions": ["region_1"],
                    },
                ]
            }), {"cache_hit": False}

    planner = LLMSemanticRoutePlanner(LLMPlannerConfig("http://unused", "mock"))
    planner.client = PartialClient()
    plans = spawn_plans()
    result = planner.plan(plans, EXITS, REGIONS)
    assert result.validation_errors == 1
    assert result.planned_agents == 2
    assert result.fallback_agents == 0
    assert plans[0]["agent_queue"][0]["route"]["regions"] == []
