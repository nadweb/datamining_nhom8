# -*- coding: utf-8 -*-
"""[STEP 6] Model Quality Gates, Leakage Audit, Approval & Production Promotion."""
import shutil
from datetime import datetime

from flight_delay.common.constants import FORBIDDEN_POST_FLIGHT_COLUMNS, PROCESSED_DIR, ROOT_DIR
from flight_delay.common.hashing import file_checksum_record
from flight_delay.common.utils import read_json, write_json
from flight_delay.governance.artifact_manager import verify_bundle_checksums, write_checksums
from flight_delay.governance.data_lineage import build_data_lineage_snapshot
from flight_delay.governance.model_card import render_model_card
from flight_delay.governance.model_registry import load_registry, transition_model_status


def validate_and_approve_model(
    model_name: str,
    version: str,
    reviewer: str = "automated-governance-gate",
    promote_to_production: bool = True,
) -> dict:
    """Enforce leakage, checksum, data quality & evaluation gates before Approved/Production promotion."""
    reg = load_registry()
    entry = reg.get("models", {}).get(model_name, {}).get(version)
    if not entry:
        raise KeyError(f"Unregistered model {model_name}:{version}")

    bundle_dir = ROOT_DIR / entry["bundle_dir"]
    if not verify_bundle_checksums(bundle_dir):
        raise RuntimeError(f"Bundle checksum verification failed for {bundle_dir}")

    meta = read_json(bundle_dir / "metadata.json")
    fschema = read_json(bundle_dir / "feature_schema.json")
    ev = read_json(bundle_dir / "evaluation.json")

    feat_names = fschema.get("feature_names", [])
    bad = sorted(set(feat_names) & set(FORBIDDEN_POST_FLIGHT_COLUMNS))
    if bad:
        raise RuntimeError(f"Approval blocked: forbidden post-flight columns detected: {bad}")

    # Task-specific performance sanity check
    if "classification" in ev and ev["classification"]:
        bal_acc = float(ev["classification"].get("balanced_accuracy", 0.0))
        if bal_acc <= 0.34:
            raise RuntimeError(f"Approval blocked: classification balanced_accuracy={bal_acc:.4f} too low")
        perm = ev.get("permutation_test") or {}
        if perm and not perm.get("statistically_significant", True):
            raise RuntimeError("Approval blocked: permutation test not statistically significant")
    elif "regression" in ev and ev["regression"]:
        mae = float(ev["regression"].get("MAE_minutes", 999.0))
        if mae <= 0 or mae > 500:
            raise RuntimeError(f"Approval blocked: implausible MAE={mae}")

    transition_model_status(model_name, version, "Validated", note=f"Quality gates passed ({reviewer})")
    new_state = "Production" if promote_to_production else "Approved"
    updated_entry = transition_model_status(model_name, version, new_state, note=f"Approved by {reviewer}")

    meta["status"] = new_state
    meta["approved"] = True
    meta["approved_by"] = reviewer
    meta["approved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_json(bundle_dir / "metadata.json", meta)
    (bundle_dir / "model_card.md").write_text(
        render_model_card(meta, ev, feature_count=len(feat_names)), encoding="utf-8"
    )
    write_checksums(bundle_dir)

    # Sync legacy compatibility copies in models/ for dashboard & benchmark consumers
    legacy_models = ROOT_DIR / "models"
    legacy_models.mkdir(parents=True, exist_ok=True)
    if new_state == "Production":
        if "classifier" in model_name:
            shutil.copy2(bundle_dir / "model.joblib", legacy_models / "best_classifier.joblib")
        elif "regressor" in model_name:
            shutil.copy2(bundle_dir / "model.joblib", legacy_models / "best_regressor.joblib")

    # Rebuild overall LOCK_pipeline_manifest.json
    refresh_lock_manifest(dataset_version=meta.get("dataset_version"))
    build_data_lineage_snapshot()
    return updated_entry


def refresh_lock_manifest(dataset_version: str = None) -> dict:
    """Recompute SHA-256 checksums across all active serving & preprocessing artifacts."""
    legacy_models = ROOT_DIR / "models"
    legacy_models.mkdir(parents=True, exist_ok=True)
    if not dataset_version:
        meta_path = PROCESSED_DIR / "preprocessing_metadata.json"
        if meta_path.exists():
            dataset_version = read_json(meta_path).get("dataset_version")
    checksums = {}
    for key, p in [
        ("encoder", PROCESSED_DIR / "encoder.joblib"),
        ("scaler", PROCESSED_DIR / "scaler.joblib"),
        ("preprocessing_metadata", PROCESSED_DIR / "preprocessing_metadata.json"),
        ("causal_history_store", PROCESSED_DIR / "causal_history_store.json"),
        ("classifier_model", legacy_models / "best_classifier.joblib"),
        ("regressor_model", legacy_models / "best_regressor.joblib"),
        ("kmeans_model", legacy_models / "kmeans_model.joblib"),
        ("isolation_forest", legacy_models / "isolation_forest.joblib"),
    ]:
        if p.exists():
            checksums[key] = file_checksum_record(p, ROOT_DIR)

    lock_payload = {
        "lock_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_version": dataset_version,
        "production_registry": load_registry().get("production", {}),
        "artifact_checksums_sha256": checksums,
    }
    write_json(legacy_models / "LOCK_pipeline_manifest.json", lock_payload)
    return lock_payload
