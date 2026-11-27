# -*- coding: utf-8 -*-
"""[STEP 7] Fail-Closed Registry & SHA-256 Verified Model Bundle Loader."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib

from flight_delay.common.constants import MONITORING_CFG, PROCESSED_DIR, ROOT_DIR
from flight_delay.common.exceptions import ApprovalError, ArtifactError, SchemaValidationError
from flight_delay.common.utils import read_json
from flight_delay.data_mining.encoding.preprocessing import FlightPreprocessor
from flight_delay.data_mining.feature_engineering.historical import CausalHistoryStore
from flight_delay.governance.artifact_manager import verify_bundle_checksums
from flight_delay.governance.model_registry import get_production_entry

PRIMARY_CLASSIFIER_NAME = "flight_delay_classifier_3class"
PRIMARY_REGRESSOR_NAME = "flight_delay_regressor_mae"

_CACHED_BUNDLE: Optional["ModelBundle"] = None


@dataclass
class ModelBundle:
    cls_model: Any
    reg_model: Any
    preprocessor: FlightPreprocessor
    history_store: CausalHistoryStore
    feature_schema: dict
    label_mapping: dict
    cls_meta: dict
    reg_meta: dict
    bundle_dir: Path


def _verify_and_load_bundle(model_name: str) -> tuple[Any, dict, dict, dict, Path]:
    entry = get_production_entry(model_name)
    if not entry:
        raise ApprovalError(f"No Production model registered for {model_name!r}. Run training & approval first.")
    require_approved = bool(MONITORING_CFG.get("serving", {}).get("require_approved_model", True))
    if require_approved and (not entry.get("approved", False) or entry.get("status") not in ("Approved", "Production")):
        raise ApprovalError(f"Model {model_name}:{entry.get('version')} is not Approved/Production.")

    bundle_dir = ROOT_DIR / entry["bundle_dir"]
    if not bundle_dir.exists():
        raise ArtifactError(f"Missing artifact bundle directory: {bundle_dir}")
    if not verify_bundle_checksums(bundle_dir):
        raise ArtifactError(f"Checksum mismatch in model bundle: {bundle_dir}")

    meta = read_json(bundle_dir / "metadata.json")
    if require_approved and not meta.get("approved", False):
        raise ApprovalError(f"Bundle metadata for {model_name} has approved=False.")

    fschema = read_json(bundle_dir / "feature_schema.json")
    lmap = read_json(bundle_dir / "label_mapping.json")
    model = joblib.load(bundle_dir / "model.joblib")
    return model, meta, fschema, lmap, bundle_dir


def load_production_artifacts(force_reload: bool = False) -> ModelBundle:
    """Load, checksum-verify, approval-verify, and parity-check all production inference artifacts."""
    global _CACHED_BUNDLE
    if _CACHED_BUNDLE is not None and not force_reload:
        return _CACHED_BUNDLE

    cls_model, cls_meta, fschema, lmap, cls_dir = _verify_and_load_bundle(PRIMARY_CLASSIFIER_NAME)
    reg_model, reg_meta, _, _, _ = _verify_and_load_bundle(PRIMARY_REGRESSOR_NAME)

    try:
        preprocessor = FlightPreprocessor.load(PROCESSED_DIR)
    except Exception as exc:
        raise ArtifactError(f"Failed loading FlightPreprocessor from {PROCESSED_DIR}: {exc}") from exc

    expected_features = list(fschema.get("feature_names", []))
    if list(preprocessor.feature_names) != expected_features:
        raise SchemaValidationError(
            f"Training-Serving schema mismatch: bundle expects {len(expected_features)} features "
            f"but preprocessor has {len(preprocessor.feature_names)}."
        )

    hist_path = PROCESSED_DIR / "causal_history_store.json"
    if not hist_path.exists():
        raise ArtifactError(f"Missing causal history store at {hist_path}")
    history_store = CausalHistoryStore.from_dict(read_json(hist_path))

    _CACHED_BUNDLE = ModelBundle(
        cls_model=cls_model,
        reg_model=reg_model,
        preprocessor=preprocessor,
        history_store=history_store,
        feature_schema=fschema,
        label_mapping=lmap,
        cls_meta=cls_meta,
        reg_meta=reg_meta,
        bundle_dir=cls_dir,
    )
    return _CACHED_BUNDLE


def clear_model_cache() -> None:
    global _CACHED_BUNDLE
    _CACHED_BUNDLE = None
