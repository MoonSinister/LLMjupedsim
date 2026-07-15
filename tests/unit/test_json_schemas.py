import json

from jsonschema import Draft202012Validator

from jupedsim_mall.experiments.sim_schema import AgentProfile, RunManifest, RunMetrics
from jupedsim_mall.project import SCHEMA_DIR


def load_schema(name: str) -> dict:
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


def test_all_json_schemas_are_valid_draft_2020_12():
    for path in SCHEMA_DIR.glob("*.schema.json"):
        Draft202012Validator.check_schema(json.loads(path.read_text(encoding="utf-8")))


def test_profile_contract_validates_against_schema():
    Draft202012Validator(load_schema("profile.schema.json")).validate(AgentProfile(profile_id="p1").to_json())


def test_manifest_contract_validates_against_schema():
    manifest = RunManifest(run_id="r1", scenario_name="smoke", seed=2026)
    Draft202012Validator(load_schema("manifest.schema.json")).validate(manifest.to_json())


def test_metrics_contract_validates_against_schema():
    metrics = RunMetrics(run_id="r1", scenario_name="smoke", seed=2026)
    Draft202012Validator(load_schema("metrics.schema.json")).validate(metrics.to_json())
