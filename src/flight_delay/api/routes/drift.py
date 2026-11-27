# -*- coding: utf-8 -*-
"""[STEP 7] Data Drift (PSI + KS), Concept Drift, Monitoring & Lineage Manifest Routes."""
from fastapi import APIRouter

from flight_delay.api.dependencies import safe_read_json
from flight_delay.common.constants import ARTIFACT_MANIFESTS_DIR, DATA_MANIFESTS_DIR, REPORTS_DIR, ROOT_DIR

router = APIRouter(prefix="/api", tags=["Monitoring & Manifest"])


@router.get("/drift")
def api_drift():
    drift = safe_read_json(REPORTS_DIR / "monitoring" / "drift_report.json")
    dd = drift.get("data_drift", {})
    cd = drift.get("concept_drift", {})
    fdrift = drift.get("feature_drift", {})
    ref_r = cd.get("reference_rates_pct", {})
    cur_r = cd.get("current_rates_pct", {})

    # Flatten labels cleanly for dashboard charts
    label_table = {
        "on_time": {
            "train_pct": ref_r.get("on_time_pct", 0.0),
            "test_pct": cur_r.get("on_time_pct", 0.0),
            "diff_pct_points": round(cur_r.get("on_time_pct", 0.0) - ref_r.get("on_time_pct", 0.0), 2),
        },
        "delayed": {
            "train_pct": ref_r.get("delayed_pct", 0.0),
            "test_pct": cur_r.get("delayed_pct", 0.0),
            "diff_pct_points": round(cur_r.get("delayed_pct", 0.0) - ref_r.get("delayed_pct", 0.0), 2),
        },
        "cancelled": {
            "train_pct": ref_r.get("cancelled_pct", 0.0),
            "test_pct": cur_r.get("cancelled_pct", 0.0),
            "diff_pct_points": round(cur_r.get("cancelled_pct", 0.0) - ref_r.get("cancelled_pct", 0.0), 2),
        },
        "disrupted": {
            "train_pct": ref_r.get("disrupted_pct", 0.0),
            "test_pct": cur_r.get("disrupted_pct", 0.0),
            "diff_pct_points": round(cur_r.get("disrupted_pct", 0.0) - ref_r.get("disrupted_pct", 0.0), 2),
        },
    }
    return {
        **drift,
        "train_rows": dd.get("ref_rows", 0),
        "test_rows": dd.get("cur_rows", 0),
        "feature_psi": fdrift,
        "drifted_features": [k for k, v in fdrift.items() if float(v.get("psi", 0)) >= 0.10],
        "high_drift_features": [k for k, v in fdrift.items() if float(v.get("psi", 0)) >= 0.25],
        "label_drift": label_table,
    }


@router.get("/manifest")
def api_manifest():
    lock = safe_read_json(ROOT_DIR / "models" / "LOCK_pipeline_manifest.json")
    ds = safe_read_json(DATA_MANIFESTS_DIR / "dataset_manifest.json")
    reg = safe_read_json(ARTIFACT_MANIFESTS_DIR / "model_registry.json")
    return {
        **lock,
        "dataset_manifest": ds,
        "model_registry": reg,
        "pipeline_lifecycle": {
            "phase_1_ingestion": {"status": "LOCKED"},
            "phase_2_data_engineering": {"status": "LOCKED"},
            "phase_3_processing": {"status": "APPROVED & LOCKED"},
        },
    }
