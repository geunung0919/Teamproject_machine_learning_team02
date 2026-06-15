from __future__ import annotations

import os
from pathlib import Path

# Base workspace directories
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
HANDOFF_BASE = ROOT / "handoff_for_chatgpt"

# Dataset build & patch directories
CLEAN_BUILD_DIR = DATA_DIR / "v5_clean_dataset_build_v1"
CLEAN_PATCH_DIR = DATA_DIR / "v5_clean_dataset_patch_v1"

# Stage experiment outputs
POLICY_COMP_DIR = DATA_DIR / "v5_direct_multihorizon_policy_comparison_v1"
COHORT_TRAIN_DIR = DATA_DIR / "v5_r4_r5_r6_cohort_pressure_model_training_v1"
RECURSIVE_FORECAST_DIR = DATA_DIR / "v5_recursive_and_multioutput_forecasting_r3_r6_v1"
WEB_PACKAGE_DIR = DATA_DIR / "v5_web_scenario_package_v1"

# Public assets folder
PUBLIC_ASSETS_DIR = ROOT / "public" / "data"

def rel(path: Path) -> str:
    """Return relative path from project root for reporting."""
    try:
        return str(path.relative_to(ROOT)).replace("/", "\\")
    except ValueError:
        return str(path)

def ensure_all_dirs() -> None:
    """Ensure standard output folders exist."""
    for d in [DATA_DIR, RAW_DIR, MODELS_DIR, REPORTS_DIR, HANDOFF_BASE, PUBLIC_ASSETS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
