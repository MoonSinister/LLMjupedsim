"""Environment diagnostics for reproducible experiment runs."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from jupedsim_mall.project import PROJECT_ROOT


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str
    required: bool = True


def _package_check(
    import_name: str,
    distribution_name: str | None = None,
    *,
    required: bool = True,
) -> CheckResult:
    distribution_name = distribution_name or import_name
    if importlib.util.find_spec(import_name) is None:
        status = "error" if required else "warning"
        return CheckResult(import_name, status, "not installed", required=required)
    try:
        version = importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        version = "version unknown"
    return CheckResult(import_name, "ok", version, required=required)


def _file_check(relative_path: str) -> CheckResult:
    path = PROJECT_ROOT / relative_path
    if path.is_file():
        return CheckResult(relative_path, "ok", str(path))
    return CheckResult(relative_path, "error", f"missing: {path}")


def _optional_path_check(env_name: str) -> CheckResult:
    value = os.environ.get(env_name, "").strip()
    if not value:
        return CheckResult(env_name, "warning", "not configured", required=False)
    path = Path(value).expanduser()
    if path.exists():
        return CheckResult(env_name, "ok", str(path), required=False)
    return CheckResult(env_name, "warning", f"configured path does not exist: {path}", required=False)


def _llm_check(check_network: bool) -> CheckResult:
    base_url = os.environ.get("LOCAL_LLM_BASE_URL", "").rstrip("/")
    if not base_url:
        return CheckResult("llm_endpoint", "warning", "LOCAL_LLM_BASE_URL is not configured", required=False)
    if not check_network:
        return CheckResult("llm_endpoint", "warning", f"configured but not probed: {base_url}", required=False)
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=3) as response:
            return CheckResult("llm_endpoint", "ok", f"HTTP {response.status}: {base_url}", required=False)
    except (OSError, urllib.error.URLError) as exc:
        return CheckResult("llm_endpoint", "warning", f"unreachable: {exc}", required=False)


def collect_checks(check_network: bool = False) -> list[CheckResult]:
    return [
        CheckResult("python", "ok", f"{platform.python_version()} ({sys.executable})"),
        CheckResult("project_root", "ok" if PROJECT_ROOT.is_dir() else "error", str(PROJECT_ROOT)),
        _package_check("jupedsim"),
        _package_check("shapely"),
        _package_check("numpy"),
        _package_check("yaml", "PyYAML"),
        _package_check("jsonschema"),
        _package_check("jupedsim_visualizer", required=False),
        _file_check("data/map/geometry.wkt"),
        _file_check("data/map/stages.json"),
        _file_check("data/map/localization_grid_regions.json"),
        _file_check("configs/scenarios/smoke_baseline.json"),
        _optional_path_check("ATC_RAW_PATH"),
        _optional_path_check("LLMOB_DATA_ROOT"),
        _llm_check(check_network),
    ]


def run_doctor(*, check_network: bool = False, json_output: bool = False) -> int:
    checks = collect_checks(check_network=check_network)
    if json_output:
        print(json.dumps({"checks": [asdict(item) for item in checks]}, indent=2, ensure_ascii=False))
    else:
        for item in checks:
            print(f"[{item.status.upper():7}] {item.name}: {item.detail}")
    return 1 if any(item.required and item.status == "error" for item in checks) else 0
