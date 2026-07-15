from jupedsim_mall.project import PROJECT_ROOT, SCENARIO_DIR, SCHEMA_DIR


def test_project_paths_resolve_to_repository_root():
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert SCENARIO_DIR == PROJECT_ROOT / "configs" / "scenarios"
    assert (SCHEMA_DIR / "scenario.schema.json").is_file()

