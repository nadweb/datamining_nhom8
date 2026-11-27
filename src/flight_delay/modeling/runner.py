# -*- coding: utf-8 -*-
"""[STEP 4 & 5] Unified Training, Evaluation, Governance Registration & Unsupervised Runner."""
import pandas as pd

from flight_delay.common.constants import FILE_GOLD, REPORTS_DIR
from flight_delay.common.utils import write_json
from flight_delay.evaluation.evaluation_report import save_evaluation_report
from flight_delay.governance.approval import refresh_lock_manifest
from flight_delay.modeling.classification.trainer import run_classification_suite
from flight_delay.modeling.regression.trainer import run_regression_suite
from flight_delay.modeling.unsupervised.anomaly_detection import run_anomaly_detection
from flight_delay.modeling.unsupervised.association_rules import mine_association_rules
from flight_delay.modeling.unsupervised.clustering import run_kmeans_clustering
from flight_delay.modeling.unsupervised.dimensionality_reduction import run_pca_analysis


def run_unsupervised_suite() -> dict:
    gold_df = pd.read_csv(FILE_GOLD, low_memory=False)
    km_raw = run_kmeans_clustering(gold_df)
    labels = km_raw.pop("_labels")
    X_scaled = km_raw.pop("_X_scaled")

    pca_res = run_pca_analysis(X_scaled, km_raw["features_used"], labels, gold_df)
    iso_res = run_anomaly_detection(gold_df)
    rules_res = mine_association_rules(gold_df)

    payload = {
        "kmeans": km_raw,
        "pca": pca_res,
        "isolation_forest": iso_res,
        "association_rules": rules_res,
    }
    write_json(REPORTS_DIR / "experiments" / "unsupervised_results.json", payload)
    refresh_lock_manifest()
    return payload


def run_all_modeling_and_evaluation(include_horizon_ablation: bool = True) -> dict:
    """Execute Regression -> Classification -> Unsupervised -> Consolidated Evaluation Report."""
    reg_res = run_regression_suite()
    cls_res = run_classification_suite(
        reg_test_preds=reg_res.get("full_test_predictions"),
        include_horizon_ablation=include_horizon_ablation,
    )
    unsup_res = run_unsupervised_suite()

    mc = cls_res["multiclass_3"]
    best_cls_name = mc["best_model"]
    best_cls_metrics = mc["test_comparison"][best_cls_name]
    eval_summary = save_evaluation_report({
        "dataset_version": cls_res.get("dataset_version"),
        "primary_horizon_hours": cls_res.get("primary_horizon_hours"),
        "classification_3class": {
            "best_model": best_cls_name,
            "test_metrics": best_cls_metrics,
            "temporal_cv": mc["temporal_cv"].get(best_cls_name, {}),
            "calibration": mc["calibration"],
            "statistical_comparisons": mc["statistical_comparisons"],
            "permutation_test": mc["permutation_test"],
            "subgroups": mc["subgroups"],
        },
        "classification_binary_ablation": {
            "best_model": cls_res["binary_disrupted"]["best_model"],
            "test_metrics": cls_res["binary_disrupted"]["test_comparison"][cls_res["binary_disrupted"]["best_model"]],
        },
        "horizon_ablation": cls_res.get("horizon_ablation", {}),
        "regression_mae": {
            "best_model": reg_res["best_model"],
            "test_metrics": reg_res["test_comparison"][reg_res["best_model"]],
            "subgroups": reg_res["subgroups"],
        },
        "unsupervised_summary": {
            "kmeans_k": unsup_res["kmeans"]["k_selected"],
            "kmeans_silhouette": unsup_res["kmeans"]["full_silhouette"],
            "kmeans_seed_ari_mean": unsup_res["kmeans"]["seed_stability"]["ari_mean"],
            "pca_cumulative_variance_pct": unsup_res["pca"]["cumulative_variance_pct"],
            "anomaly_rate_pct": unsup_res["isolation_forest"]["anomaly_rate_pct"],
            "association_rules_total": unsup_res["association_rules"]["total_non_redundant_rules"],
        },
    })
    return {
        "regression": reg_res,
        "classification": cls_res,
        "unsupervised": unsup_res,
        "evaluation_report": eval_summary,
    }
