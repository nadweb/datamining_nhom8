# -*- coding: utf-8 -*-
"""[STEP 6] End-to-End Bronze -> Silver -> Gold -> Model Data Lineage Builder."""
from datetime import datetime

from flight_delay.common.constants import ARTIFACT_MANIFESTS_DIR, DATA_MANIFESTS_DIR
from flight_delay.common.utils import read_json, write_json


def build_data_lineage_snapshot() -> dict:
    bronze = read_json(DATA_MANIFESTS_DIR / "bronze_manifest.json") if (DATA_MANIFESTS_DIR / "bronze_manifest.json").exists() else {}
    dataset = read_json(DATA_MANIFESTS_DIR / "dataset_manifest.json") if (DATA_MANIFESTS_DIR / "dataset_manifest.json").exists() else {}
    registry = read_json(ARTIFACT_MANIFESTS_DIR / "model_registry.json") if (ARTIFACT_MANIFESTS_DIR / "model_registry.json").exists() else {}
    lineage = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "bronze_source_files": bronze.get("source_csv_count", 0),
        "bronze_raw_rows": bronze.get("total_raw_mixed_rows", 0),
        "dataset_version": dataset.get("dataset_version", "unknown"),
        "prediction_horizon_hours": dataset.get("prediction_horizon_hours", 3),
        "splits": dataset.get("splits", {}),
        "registered_models": registry.get("models", {}),
    }
    write_json(ARTIFACT_MANIFESTS_DIR / "data_lineage.json", lineage)
    return lineage
