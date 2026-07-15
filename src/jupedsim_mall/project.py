"""Project-level paths shared by package modules."""

from __future__ import annotations

import os
from pathlib import Path


def _project_root() -> Path:
    override = os.environ.get("JUPEDSIM_MALL_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _project_root()
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SCENARIO_DIR = CONFIG_DIR / "scenarios"
MATRIX_DIR = CONFIG_DIR / "matrices"
SCHEMA_DIR = CONFIG_DIR / "schemas"
