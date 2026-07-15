from jupedsim_mall.profiles.providers import (
    MallProfileProvider,
    build_profile_provider,
    partition_overlap,
    source_fingerprint,
)
from jupedsim_mall.profiles.llmob_training_adapter import atc_partition


def test_mall_provider_is_deterministic_and_versioned():
    provider = MallProfileProvider()
    first = provider.sample("agent_0001", "entrance_0", 1, seed=2026)
    second = provider.sample("agent_0001", "entrance_0", 1, seed=2026)

    assert first == second
    assert first["schema_version"] == "1.0"
    assert first["source"] == "mall"


def test_missing_external_source_falls_back_with_reason(tmp_path):
    result = build_profile_provider(
        "llmob",
        data_path=str(tmp_path / "missing"),
        cache_output=str(tmp_path / "profiles.json"),
    )

    assert result.report.requested_source == "llmob"
    assert result.report.active_source == "mall"
    assert result.report.fallback_reason
    assert result.provider.source == "mall"


def test_file_source_fingerprint_changes_with_content(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("one", encoding="utf-8")
    first = source_fingerprint(source)
    source.write_text("two", encoding="utf-8")
    assert source_fingerprint(source) != first


def test_partition_overlap_reports_data_leakage():
    assert partition_overlap({"p1", "p2"}, {"p2", "p3"}) == {"p2"}


def test_atc_person_partition_is_stable_and_disjoint():
    assignments = {person_id: atc_partition(person_id) for person_id in map(str, range(100))}
    assert assignments == {person_id: atc_partition(person_id) for person_id in assignments}
    partitions = {
        name: {person_id for person_id, assigned in assignments.items() if assigned == name}
        for name in ("train", "tuning", "evaluation")
    }
    assert set.union(*partitions.values()) == set(assignments)
    assert not partition_overlap(partitions["train"], partitions["evaluation"])


def test_profile_report_exposes_quality_fields(tmp_path):
    result = build_profile_provider(
        "atc",
        data_path=str(tmp_path / "missing"),
        allow_fallback=True,
    )
    report = result.report.to_json()
    assert report["input_count"] == 0
    assert report["output_count"] == 0
    assert report["missing_rate"] == 0.0
    assert report["anomaly_rate"] == 0.0
    assert report["distributions"] == {}
