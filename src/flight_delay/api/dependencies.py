# -*- coding: utf-8 -*-
"""[STEP 7] FastAPI Shared Artifact & Report Loaders."""
from pathlib import Path
from typing import Optional

from flight_delay.common.constants import (
    ARTIFACT_MANIFESTS_DIR,
    DATA_MANIFESTS_DIR,
    EXPERIMENTS_DIR,
    FILE_RULES,
    REPORTS_DIR,
    ROOT_DIR,
)
from flight_delay.common.utils import read_json
from flight_delay.inference.model_loader import ModelBundle, load_production_artifacts


def get_bundle() -> ModelBundle:
    return load_production_artifacts()


def safe_read_json(path: Path, default: Optional[dict] = None) -> dict:
    if path.exists():
        return read_json(path)
    return default or {}


def load_consolidated_reports() -> dict:
    return {
        "evaluation": safe_read_json(REPORTS_DIR / "experiments" / "evaluation_report.json"),
        "classification": safe_read_json(REPORTS_DIR / "experiments" / "classification_results.json"),
        "regression": safe_read_json(REPORTS_DIR / "experiments" / "regression_results.json"),
        "unsupervised": safe_read_json(REPORTS_DIR / "experiments" / "unsupervised_results.json"),
        "drift": safe_read_json(REPORTS_DIR / "monitoring" / "drift_report.json"),
        "eda": safe_read_json(REPORTS_DIR / "eda" / "eda_report.json"),
        "registry": safe_read_json(ARTIFACT_MANIFESTS_DIR / "model_registry.json"),
        "dataset_manifest": safe_read_json(DATA_MANIFESTS_DIR / "dataset_manifest.json"),
        "lock_manifest": safe_read_json(ROOT_DIR / "models" / "LOCK_pipeline_manifest.json"),
        "rules_exists": FILE_RULES.exists(),
        "experiments_dir": str(EXPERIMENTS_DIR),
    }
