# -*- coding: utf-8 -*-
"""[STEP 6] Versioned Model Bundle & SHA-256 Checksum Manager."""
import importlib
from datetime import datetime
from pathlib import Path

import joblib

from flight_delay.common.constants import MODELS_DIR, ROOT_DIR
from flight_delay.common.hashing import sha256_file
from flight_delay.common.utils import write_json
from flight_delay.governance.model_card import render_model_card
from flight_delay.governance.model_registry import register_candidate_model


def write_checksums(bundle_dir: Path) -> dict:
    checksums = {}
    for p in sorted(bundle_dir.iterdir()):
        if p.is_file() and p.name != "checksums.sha256":
            checksums[p.name] = sha256_file(p)
    lines = [f"{digest}  {name}" for name, digest in sorted(checksums.items())]
    (bundle_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checksums


def verify_bundle_checksums(bundle_dir: Path) -> bool:
    csum_file = Path(bundle_dir) / "checksums.sha256"
    if not csum_file.exists():
        return False
    for line in csum_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(None, 1)
        target = Path(bundle_dir) / name.strip()
        if not target.exists() or sha256_file(target) != expected.strip():
            return False
    return True


def _requirements_lock_text() -> str:
    pkgs = ["pandas", "numpy", "scikit-learn", "xgboost", "catboost", "scipy", "joblib", "pydantic", "fastapi", "pyyaml", "shap"]
    lines = []
    for pkg in pkgs:
        mod_name = "yaml" if pkg == "pyyaml" else "sklearn" if pkg == "scikit-learn" else pkg
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "installed")
        except Exception:
            ver = "missing"
        lines.append(f"{pkg}=={ver}")
    return "\n".join(lines) + "\n"


def package_model_bundle(
    model,
    model_name: str,
    version: str,
    algorithm: str,
    task: str,
    dataset_version: str,
    feature_schema: dict,
    label_mapping: dict,
    evaluation: dict,
    prediction_horizon_hours: int,
    training_cutoff: str,
    random_seed: int = 42,
) -> Path:
    """Create complete governance bundle under artifacts/models/<model_name>_<version>/."""
    bundle_dir = MODELS_DIR / f"{model_name}_{version}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, bundle_dir / "model.joblib")
    metadata = {
        "model_name": model_name,
        "version": version,
        "algorithm": algorithm,
        "task": task,
        "status": "Candidate",
        "approved": False,
        "dataset_version": dataset_version,
        "feature_schema_version": str(feature_schema.get("schema_version", "2.0")),
        "prediction_horizon_hours": int(prediction_horizon_hours),
        "training_cutoff": training_cutoff,
        "random_seed": int(random_seed),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(bundle_dir / "metadata.json", metadata)
    write_json(bundle_dir / "feature_schema.json", feature_schema)
    write_json(bundle_dir / "label_mapping.json", label_mapping)
    write_json(bundle_dir / "evaluation.json", evaluation)
    (bundle_dir / "requirements.lock").write_text(_requirements_lock_text(), encoding="utf-8")

    n_feats = len(feature_schema.get("feature_names", []))
    card_md = render_model_card(metadata, evaluation, feature_count=n_feats)
    (bundle_dir / "model_card.md").write_text(card_md, encoding="utf-8")

    write_checksums(bundle_dir)

    rel_bundle = str(bundle_dir.relative_to(ROOT_DIR)).replace("\\", "/")
    summary_metrics = evaluation.get("classification") or evaluation.get("regression") or evaluation
    register_candidate_model(
        model_name=model_name,
        version=version,
        algorithm=algorithm,
        task=task,
        bundle_dir=rel_bundle,
        dataset_version=dataset_version,
        metrics=summary_metrics,
    )
    return bundle_dir
