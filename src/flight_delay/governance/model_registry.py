# -*- coding: utf-8 -*-
"""[STEP 6] Model Registry Lifecycle (Candidate -> Validated -> Approved -> Production -> Archived)."""
from datetime import datetime
from typing import Optional

from flight_delay.common.constants import ARTIFACT_MANIFESTS_DIR
from flight_delay.common.utils import read_json, write_json

REGISTRY_FILE = ARTIFACT_MANIFESTS_DIR / "model_registry.json"
VALID_STATES = ("Candidate", "Validated", "Approved", "Production", "Archived")


def load_registry() -> dict:
    if REGISTRY_FILE.exists():
        return read_json(REGISTRY_FILE)
    return {"updated_at": None, "production": {}, "models": {}}


def save_registry(registry: dict) -> dict:
    registry["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ARTIFACT_MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(REGISTRY_FILE, registry)
    return registry


def register_candidate_model(
    model_name: str,
    version: str,
    algorithm: str,
    task: str,
    bundle_dir: str,
    dataset_version: str,
    metrics: dict,
) -> dict:
    reg = load_registry()
    model_entries = reg["models"].setdefault(model_name, {})
    model_entries[version] = {
        "model_name": model_name,
        "version": version,
        "algorithm": algorithm,
        "task": task,
        "status": "Candidate",
        "approved": False,
        "dataset_version": dataset_version,
        "bundle_dir": bundle_dir,
        "metrics": metrics,
        "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_registry(reg)
    return model_entries[version]


def transition_model_status(model_name: str, version: str, new_status: str, note: str = "") -> dict:
    if new_status not in VALID_STATES:
        raise ValueError(f"Invalid registry status: {new_status!r}")
    reg = load_registry()
    entries = reg.get("models", {}).get(model_name)
    if not entries or version not in entries:
        raise KeyError(f"Model {model_name}:{version} not found in registry.")
    if new_status == "Production":
        # Archive previous production version of same model_name
        for v, info in entries.items():
            if v != version and info.get("status") == "Production":
                info["status"] = "Archived"
        reg["production"][model_name] = {
            "version": version,
            "algorithm": entries[version]["algorithm"],
            "bundle_dir": entries[version]["bundle_dir"],
            "promoted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    entries[version]["status"] = new_status
    entries[version]["approved"] = new_status in ("Approved", "Production")
    if note:
        entries[version]["transition_note"] = note
    save_registry(reg)
    return entries[version]


def get_production_entry(model_name: str) -> Optional[dict]:
    reg = load_registry()
    prod = reg.get("production", {}).get(model_name)
    if not prod:
        return None
    v = prod.get("version")
    return reg.get("models", {}).get(model_name, {}).get(v)
