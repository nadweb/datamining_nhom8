# -*- coding: utf-8 -*-
"""[MODELING & GOVERNANCE] Validate Approved Bundles, SHA-256 Checksums, Metrics & Unsupervised Rules."""
import pandas as pd

from flight_delay.common.constants import (
    ARTIFACT_MANIFESTS_DIR,
    FILE_RULES,
    MODELS_DIR,
    REPORTS_DIR,
    ROOT_DIR,
)
from flight_delay.common.hashing import sha256_file
from flight_delay.common.utils import read_json
from flight_delay.governance.artifact_manager import verify_bundle_checksums


def test_registry_approvals_and_3class_classification_artifacts():
    reg_man = read_json(ARTIFACT_MANIFESTS_DIR / "model_registry.json")
    prod = reg_man["production"]
    assert "flight_delay_classifier_3class" in prod
    assert "flight_delay_regressor_mae" in prod

    cls_rep = read_json(REPORTS_DIR / "experiments" / "classification_results.json")
    mc = cls_rep["multiclass_3"]
    assert mc["best_model"] in ("HistGradientBoosting", "XGBoost", "CatBoost")
    best_m = mc["test_comparison"][mc["best_model"]]
    assert best_m["roc_auc_ovr"] >= 0.70
    assert "delayed_recall" in best_m and "cancelled_recall" in best_m
    assert cls_rep["binary_disrupted"]["test_comparison"]["CatBoost"]["balanced_accuracy"] >= 0.60


def test_all_production_bundles_pass_sha256_verification():
    reg_man = read_json(ARTIFACT_MANIFESTS_DIR / "model_registry.json")
    for model_name, entry in reg_man["production"].items():
        bdir = ROOT_DIR / entry["bundle_dir"]
        assert bdir.exists(), f"Missing bundle dir {bdir}"
        assert verify_bundle_checksums(bdir), f"SHA-256 checksum failed for {model_name}"
        meta = read_json(bdir / "metadata.json")
        assert meta["approved"] is True
        assert (bdir / "model_card.md").exists()
        assert (bdir / "requirements.lock").exists()

    lock_path = ROOT_DIR / "models" / "LOCK_pipeline_manifest.json"
    assert lock_path.exists()
    lock = read_json(lock_path)
    for key, rec in lock["artifact_checksums_sha256"].items():
        p = ROOT_DIR / rec["path"]
        assert p.exists(), f"Missing locked artifact {key}: {p}"
        assert sha256_file(p) == rec["sha256"], f"Lock SHA-256 mismatch on {key}"
    assert MODELS_DIR.exists()


def test_regression_excluded_cancelled_and_achieved_sub_median_mae():
    reg_rep = read_json(REPORTS_DIR / "experiments" / "regression_results.json")
    assert reg_rep["excluded_counts"]["test"] >= 1
    best = reg_rep["best_model"]
    assert reg_rep["test_comparison"][best]["MAE_minutes"] <= reg_rep["test_comparison"]["MedianBaseline"]["MAE_minutes"]


def test_unsupervised_stability_and_holdout_rules():
    uns = read_json(REPORTS_DIR / "experiments" / "unsupervised_results.json")
    assert uns["kmeans"]["k_selected"] == 4
    assert uns["kmeans"]["seed_stability"]["ari_mean"] >= 0.50
    assert uns["pca"]["cumulative_variance_pct"] >= 40.0
    assert FILE_RULES.exists()
    rules = pd.read_csv(FILE_RULES)
    assert len(rules) >= 15
    assert int(rules["holdout_validated"].sum()) >= 10
