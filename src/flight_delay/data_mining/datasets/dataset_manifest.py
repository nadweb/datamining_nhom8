# -*- coding: utf-8 -*-
"""[STEP 3] Schema Lineage & Feature Count Explanation Manifest (12 -> 24 -> Gold -> X)."""
from datetime import datetime

from flight_delay.common.constants import (
    CLEANED_FIELDS,
    DATA_MANIFESTS_DIR,
    FIELDS,
    FORBIDDEN_POST_FLIGHT_COLUMNS,
)
from flight_delay.common.utils import write_json
from flight_delay.data_mining.feature_engineering.registry import FEATURE_REGISTRY


def build_dataset_manifest(
    dataset_version: str,
    horizon_hours: int,
    silver_rows: int,
    gold_cols_count: int,
    preprocessor_meta: dict,
    split_summary: dict,
    bronze_audit: dict = None,
) -> dict:
    """Document exact schema transformations so 12/24/Gold/X column counts are 100% transparent."""
    encoded_x_count = len(preprocessor_meta.get("feature_names", []))
    cat_cols = preprocessor_meta.get("categorical_columns", [])
    num_cols = preprocessor_meta.get("numeric_continuous_columns", [])
    bin_cols = preprocessor_meta.get("binary_columns", [])
    cat_ohe_count = len(preprocessor_meta.get("encoded_cat_names", []))

    manifest = {
        "dataset_version": dataset_version,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prediction_horizon_hours": horizon_hours,
        "schema_lineage": [
            {
                "stage": "01_bronze_raw",
                "columns": bronze_audit.get("total_unique_raw_columns", 58) if bronze_audit else 58,
                "rows": bronze_audit.get("total_raw_mixed_rows", 0) if bronze_audit else 0,
                "explanation": "Heterogeneous FIDS HTML + Aviationstack JSON + Open-Meteo raw caches.",
            },
            {
                "stage": "02_bronze_standardized",
                "columns": len(FIELDS),
                "rows": bronze_audit.get("standardized_rows", 0) if bronze_audit else 0,
                "explanation": "Unified 12 canonical FIELDS mapping across FIDS and Aviationstack.",
            },
            {
                "stage": "03_silver_cleaned",
                "columns": len(CLEANED_FIELDS),
                "rows": silver_rows,
                "explanation": "3-tier SHA-256 deduplicated legs, Modulo 1440 delay_min, ICAO labels {on_time, delayed, cancelled}, Open-Meteo weather.",
            },
            {
                "stage": "04_gold_analytical",
                "columns": gold_cols_count,
                "rows": silver_rows,
                "explanation": (
                    "24 Silver cols + domain engineered features (temporal, weather hazard, congestion, interactions, causal historical rates) "
                    "+ prediction_time metadata & targets. Auxiliary/post-flight columns are preserved for EDA/Regression ONLY."
                ),
            },
            {
                "stage": "05_modeling_matrix_X",
                "columns": encoded_x_count,
                "rows": split_summary["train_rows"] + split_summary["val_rows"] + split_summary["test_rows"],
                "explanation": (
                    f"Exact model matrix X = {cat_ohe_count} One-Hot cols (from {len(cat_cols)} categorical inputs) "
                    f"+ {len(num_cols)} StandardScaled continuous/historical inputs + {len(bin_cols)} binary flags "
                    f"= {encoded_x_count} columns. Excludes {len(FORBIDDEN_POST_FLIGHT_COLUMNS)} post-flight/target columns & raw identifiers."
                ),
            },
        ],
        "feature_counts_breakdown": {
            "categorical_input_cols": len(cat_cols),
            "categorical_one_hot_cols": cat_ohe_count,
            "numeric_scaled_cols": len(num_cols),
            "binary_passthrough_cols": len(bin_cols),
            "total_pre_flight_input_cols": len(cat_cols) + len(num_cols) + len(bin_cols),
            "total_encoded_matrix_X_cols": encoded_x_count,
            "excluded_post_flight_columns": FORBIDDEN_POST_FLIGHT_COLUMNS,
        },
        "splits": split_summary,
        "feature_registry": FEATURE_REGISTRY.to_manifest(),
    }
    DATA_MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(DATA_MANIFESTS_DIR / "dataset_manifest.json", manifest)
    return manifest
