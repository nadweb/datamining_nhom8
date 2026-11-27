# -*- coding: utf-8 -*-
"""[STEP 4] Tabular Supervised Classification Trainer (3-Class Primary + Binary & Horizon Ablations)."""
import logging
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from flight_delay.common.constants import (
    ABLATION_HORIZONS,
    EXPERIMENTS_CFG,
    EXPERIMENTS_DIR,
    FILE_GOLD,
    LABEL_MAP_3CLASS,
    LABEL_MAP_BINARY,
    PRIMARY_HORIZON_HOURS,
    PROCESSED_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
)
from flight_delay.common.utils import read_json, write_json
from flight_delay.data_mining.datasets.dataset_builder import build_analytical_and_modeling_datasets
from flight_delay.data_mining.encoding.dataset_split import expanding_temporal_folds
from flight_delay.data_mining.feature_engineering.registry import FEATURE_REGISTRY, assert_no_leakage
from flight_delay.evaluation.classification_metrics import compute_classification_metrics
from flight_delay.evaluation.confusion_analysis import analyze_confusion_pairs
from flight_delay.evaluation.error_analysis import analyze_prediction_errors
from flight_delay.evaluation.statistical_comparison import (
    mcnemar_test,
    paired_bootstrap_ci,
    permutation_significance_test,
)
from flight_delay.evaluation.subgroup_evaluation import evaluate_subgroup_performance
from flight_delay.evaluation.temporal_backtesting import run_temporal_fold_backtest
from flight_delay.governance.approval import validate_and_approve_model
from flight_delay.governance.artifact_manager import package_model_bundle
from flight_delay.governance.experiment_tracker import log_experiment_run
from flight_delay.modeling.baselines.heuristic_baseline import HeuristicBaselineClassifier
from flight_delay.modeling.baselines.majority_classifier import build_majority_classifier
from flight_delay.modeling.calibration.probability_calibration import compute_probability_calibration
from flight_delay.modeling.classification.catboost_model import build_catboost_classifier
from flight_delay.modeling.classification.hist_gradient_boosting import build_hgb_classifier
from flight_delay.modeling.classification.xgboost_model import build_xgboost_classifier
from flight_delay.modeling.explainability.feature_importance import extract_model_feature_importances
from flight_delay.modeling.explainability.permutation import compute_feature_permutation_importance
from flight_delay.modeling.explainability.shap_analysis import compute_shap_summary

log = logging.getLogger(__name__)


def _predict_probs_safe(est, X: pd.DataFrame, n_classes: int) -> np.ndarray:
    """Return float64 row-normalized (N, n_classes) probabilities even for partial DummyClassifier."""
    raw = np.asarray(est.predict_proba(X), dtype=np.float64)
    if raw.shape[1] == n_classes:
        row_sums = np.where(raw.sum(axis=1, keepdims=True) > 0, raw.sum(axis=1, keepdims=True), 1.0)
        return raw / row_sums
    classes = getattr(est, "classes_", np.arange(raw.shape[1]))
    out = np.full((len(X), n_classes), 1e-6, dtype=np.float64)
    for col_i, cls_val in enumerate(classes):
        idx = int(cls_val)
        if 0 <= idx < n_classes:
            out[:, idx] = raw[:, col_i]
    return out / out.sum(axis=1, keepdims=True)


def _predict_labels_from_probs(probs: np.ndarray) -> np.ndarray:
    return np.argmax(probs, axis=1).astype(int)


def _train_single_classification_task(
    task_name: str,
    target_col: str,
    class_names: List[str],
    label_mapping: Dict[str, int],
    model_name: str,
    version: str,
    tr_df: pd.DataFrame,
    va_df: pd.DataFrame,
    te_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    feature_cols: List[str],
    prep_meta: dict,
    reg_test_preds: Optional[np.ndarray] = None,
    approve_to_production: bool = True,
) -> dict:
    assert_no_leakage(feature_cols)
    n_classes = len(class_names)

    X_tr, y_tr = tr_df[feature_cols], tr_df[target_col].astype(int)
    X_va, y_va = va_df[feature_cols], va_df[target_col].astype(int)
    X_te, y_te = te_df[feature_cols], te_df[target_col].astype(int)

    X_tr_va = pd.concat([X_tr, X_va], ignore_index=True)
    y_tr_va = pd.concat([y_tr, y_va], ignore_index=True)
    cv_folds_count = int(EXPERIMENTS_CFG["split"].get("temporal_cv_folds", 3))
    folds = expanding_temporal_folds(X_tr_va, n_folds=cv_folds_count)

    def _builders() -> Dict[str, callable]:
        return {
            "MajorityClassifier": lambda: build_majority_classifier(),
            "HeuristicBaseline": lambda: HeuristicBaselineClassifier(n_classes=n_classes),
            "HistGradientBoosting": lambda: build_hgb_classifier(),
            "XGBoost": lambda: build_xgboost_classifier(n_classes=n_classes),
            "CatBoost": lambda: build_catboost_classifier(n_classes=n_classes),
        }

    builders = _builders()
    fitted_models = {}
    val_results = {}
    test_results = {}
    cv_results = {}
    test_preds_map = {}
    test_probs_map = {}

    for name, fn in builders.items():
        est = fn()
        t0 = time.perf_counter()
        if name == "XGBoost":
            sw = compute_sample_weight("balanced", y_tr)
            est.fit(X_tr, y_tr, sample_weight=sw)
        else:
            est.fit(X_tr, y_tr)
        fit_ms = (time.perf_counter() - t0) * 1000.0
        fitted_models[name] = est

        # Validation evaluation (model selection occurs strictly here)
        va_probs = _predict_probs_safe(est, X_va, n_classes)
        va_pred = _predict_labels_from_probs(va_probs)
        val_metrics = compute_classification_metrics(y_va, va_pred, va_probs, class_names)
        val_metrics["fit_ms"] = round(fit_ms, 2)
        val_results[name] = val_metrics

        # Single Holdout Test evaluation
        t1 = time.perf_counter()
        te_probs = _predict_probs_safe(est, X_te, n_classes)
        te_pred = _predict_labels_from_probs(te_probs)
        infer_ms = (time.perf_counter() - t1) * 1000.0
        te_metrics = compute_classification_metrics(y_te, te_pred, te_probs, class_names, infer_ms=infer_ms)
        te_metrics["fit_ms"] = round(fit_ms, 2)
        test_results[name] = te_metrics
        test_preds_map[name] = te_pred
        test_probs_map[name] = te_probs

        # Expanding Temporal Fold Backtesting inside Train+Val for ML algorithms
        if name in ("HistGradientBoosting", "XGBoost", "CatBoost") and folds:
            cv_res = run_temporal_fold_backtest(
                X_tr_va,
                y_tr_va,
                fn,
                folds,
                sample_weights_fn=(lambda ys: compute_sample_weight("balanced", ys)),
            )
            cv_results[name] = cv_res
            te_metrics["cv_f1_macro_mean"] = cv_res["cv_f1_macro_mean"]
            te_metrics["cv_f1_macro_std"] = cv_res["cv_f1_macro_std"]
            te_metrics["cv_bal_acc_mean"] = cv_res["cv_bal_acc_mean"]

        log_experiment_run(
            category="classification",
            run_name=f"{task_name}_{name}",
            params={"task": task_name, "algorithm": name, "horizon_hours": prep_meta.get("horizon_hours", PRIMARY_HORIZON_HOURS)},
            metrics={
                "val_f1_macro": val_metrics["f1_macro"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "test_f1_macro": te_metrics["f1_macro"],
                "test_balanced_accuracy": te_metrics["balanced_accuracy"],
            },
        )

    ml_candidates = ["HistGradientBoosting", "XGBoost", "CatBoost"]
    best_name = max(
        ml_candidates,
        key=lambda n: (val_results[n]["f1_macro"], val_results[n]["balanced_accuracy"]),
    )
    best_model = fitted_models[best_name]
    best_test = test_results[best_name]
    best_probs = test_probs_map[best_name]
    best_preds = test_preds_map[best_name]

    # Statistical Comparisons & Explainability on Holdout Test
    other_ml = [n for n in ml_candidates if n != best_name]
    runner_up_ml = max(other_ml, key=lambda n: val_results[n]["f1_macro"])
    stat_tests = {
        "best_vs_heuristic": {
            "bootstrap_f1_macro": paired_bootstrap_ci(y_te.values, test_preds_map["HeuristicBaseline"], best_preds, metric="f1_macro", seed=RANDOM_STATE),
            "mcnemar": mcnemar_test(y_te.values, test_preds_map["HeuristicBaseline"], best_preds),
        },
        f"best_vs_{runner_up_ml}": {
            "bootstrap_f1_macro": paired_bootstrap_ci(y_te.values, test_preds_map[runner_up_ml], best_preds, metric="f1_macro", seed=RANDOM_STATE),
            "mcnemar": mcnemar_test(y_te.values, test_preds_map[runner_up_ml], best_preds),
        },
    }
    perm_sig = permutation_significance_test(y_te.values, best_preds, n_permutations=200, seed=RANDOM_STATE)
    calibration = compute_probability_calibration(y_te.values, best_probs, n_bins=5)
    confusion = analyze_confusion_pairs(best_test["confusion_matrix"], class_names)
    feat_imp = extract_model_feature_importances(best_model, feature_cols, FEATURE_REGISTRY.categorical_columns)
    perm_imp = compute_feature_permutation_importance(best_model, X_te, y_te, feature_cols, seed=RANDOM_STATE)
    shap_summary = compute_shap_summary(best_model, X_te, feature_cols, class_names, max_samples=400, seed=RANDOM_STATE)

    te_gold = gold_df[gold_df["split_role"] == "test"].reset_index(drop=True)
    subgroups = evaluate_subgroup_performance(
        te_gold,
        y_te.values,
        best_preds,
        y_reg_pred=reg_test_preds if (reg_test_preds is not None and len(reg_test_preds) == len(te_gold)) else None,
        labels=list(range(n_classes)),
    )
    error_examples = analyze_prediction_errors(te_gold, y_te.values, best_preds, best_probs, class_names, top_k=10)

    feature_schema = {
        "schema_version": FEATURE_REGISTRY.schema_version,
        "feature_names": feature_cols,
        "categorical_columns": FEATURE_REGISTRY.categorical_columns,
        "numeric_columns": FEATURE_REGISTRY.numeric_columns,
        "binary_columns": FEATURE_REGISTRY.binary_columns,
        "total_features": len(feature_cols),
    }
    eval_payload = {
        "task": task_name,
        "best_model": best_name,
        "selection_split": "val",
        "selection_metric": EXPERIMENTS_CFG["classification"]["selection_metric"],
        "classification": best_test,
        "validation_metrics": val_results[best_name],
        "temporal_cv": cv_results.get(best_name, {}),
        "calibration": calibration,
        "confusion_analysis": confusion,
        "statistical_comparisons": stat_tests,
        "permutation_test": perm_sig,
        "feature_importance": feat_imp,
        "permutation_importance": perm_imp,
        "shap_summary": shap_summary,
        "subgroups": subgroups,
        "error_analysis": error_examples,
    }

    package_model_bundle(
        model=best_model,
        model_name=model_name,
        version=version,
        algorithm=best_name,
        task=task_name,
        dataset_version=str(prep_meta.get("dataset_version", "unknown")),
        feature_schema=feature_schema,
        label_mapping={"class_to_id": label_mapping, "class_names": class_names},
        evaluation=eval_payload,
        prediction_horizon_hours=int(prep_meta.get("horizon_hours", PRIMARY_HORIZON_HOURS)),
        training_cutoff=str(prep_meta.get("cutoff_date", "")),
        random_seed=RANDOM_STATE,
    )
    reg_entry = validate_and_approve_model(
        model_name=model_name,
        version=version,
        reviewer="automated-governance-gate",
        promote_to_production=approve_to_production,
    )

    return {
        "task": task_name,
        "class_names": class_names,
        "best_model": best_name,
        "registry_entry": reg_entry,
        "validation_comparison": val_results,
        "test_comparison": test_results,
        "temporal_cv": cv_results,
        "calibration": calibration,
        "confusion_analysis": confusion,
        "statistical_comparisons": stat_tests,
        "permutation_test": perm_sig,
        "feature_importance": feat_imp,
        "permutation_importance": perm_imp,
        "shap_summary": shap_summary,
        "subgroups": subgroups,
        "error_analysis": error_examples,
    }


def run_horizon_ablation(horizons: Optional[List[int]] = None) -> dict:
    """Compare T-1h, T-3h, T-6h forecast horizons using XGBoost & CatBoost on canonical Silver split."""
    hs = sorted(set(horizons or ([PRIMARY_HORIZON_HOURS] + ABLATION_HORIZONS)))
    class_names = ["on_time", "delayed", "cancelled"]
    rows = []
    for h in hs:
        out_dir = EXPERIMENTS_DIR / "horizon_ablations" / f"ablation_h{h}"
        res = build_analytical_and_modeling_datasets(horizon_hours=h, output_dir=out_dir)
        tr_df = pd.read_csv(out_dir / "train.csv")
        va_df = pd.read_csv(out_dir / "val.csv")
        te_df = pd.read_csv(out_dir / "test.csv")
        feat_cols = [c for c in res["metadata"]["feature_names"]]

        X_tr, y_tr = tr_df[feat_cols], tr_df["target"].astype(int)
        X_va, y_va = va_df[feat_cols], va_df["target"].astype(int)
        X_te, y_te = te_df[feat_cols], te_df["target"].astype(int)

        best_va_f1 = -1.0
        chosen_name = ""
        chosen_metrics = {}
        for alg_name, est in [("XGBoost", build_xgboost_classifier(3)), ("CatBoost", build_catboost_classifier(3))]:
            if alg_name == "XGBoost":
                est.fit(X_tr, y_tr, sample_weight=compute_sample_weight("balanced", y_tr))
            else:
                est.fit(X_tr, y_tr)
            va_p = _predict_probs_safe(est, X_va, 3)
            va_m = compute_classification_metrics(y_va, _predict_labels_from_probs(va_p), va_p, class_names)
            if va_m["f1_macro"] > best_va_f1:
                best_va_f1 = va_m["f1_macro"]
                chosen_name = alg_name
                te_p = _predict_probs_safe(est, X_te, 3)
                chosen_metrics = compute_classification_metrics(y_te, _predict_labels_from_probs(te_p), te_p, class_names)

        rows.append({
            "horizon_hours": int(h),
            "is_primary": bool(h == PRIMARY_HORIZON_HOURS),
            "best_algorithm": chosen_name,
            "val_f1_macro": round(float(best_va_f1), 4),
            "test_f1_macro": chosen_metrics.get("f1_macro"),
            "test_balanced_accuracy": chosen_metrics.get("balanced_accuracy"),
            "test_delayed_recall": chosen_metrics.get("delayed_recall"),
            "test_cancelled_recall": chosen_metrics.get("cancelled_recall"),
            "test_log_loss": chosen_metrics.get("log_loss"),
            "test_brier_disrupted": chosen_metrics.get("brier_disrupted"),
        })

    payload = {
        "horizons": rows,
        "note": (
            "Causal lookback sensitivity across T-1h, T-3h, and T-6h. "
            "Historical airport/airline rates only aggregate completed flights before scheduled_dt - H hours."
        ),
    }
    write_json(REPORTS_DIR / "experiments" / "horizon_ablation_results.json", payload)
    return payload


def run_classification_suite(
    reg_test_preds: Optional[np.ndarray] = None,
    include_horizon_ablation: bool = True,
) -> dict:
    """Execute Primary 3-class ICAO Classification + Binary Ablation + Horizon Ablation."""
    tr_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    va_df = pd.read_csv(PROCESSED_DIR / "val.csv")
    te_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    gold_df = pd.read_csv(FILE_GOLD, low_memory=False)
    prep_meta = read_json(PROCESSED_DIR / "preprocessing_metadata.json")
    feature_cols = list(prep_meta["feature_names"])

    primary_3class = _train_single_classification_task(
        task_name="multiclass_3",
        target_col="target",
        class_names=["on_time", "delayed", "cancelled"],
        label_mapping=LABEL_MAP_3CLASS,
        model_name="flight_delay_classifier_3class",
        version="v2.0.0",
        tr_df=tr_df,
        va_df=va_df,
        te_df=te_df,
        gold_df=gold_df,
        feature_cols=feature_cols,
        prep_meta=prep_meta,
        reg_test_preds=reg_test_preds,
        approve_to_production=True,
    )

    binary_ablation = _train_single_classification_task(
        task_name="binary_disrupted",
        target_col="target_binary",
        class_names=["on_time", "disrupted"],
        label_mapping=LABEL_MAP_BINARY,
        model_name="flight_delay_classifier_binary",
        version="v2.0.0",
        tr_df=tr_df,
        va_df=va_df,
        te_df=te_df,
        gold_df=gold_df,
        feature_cols=feature_cols,
        prep_meta=prep_meta,
        reg_test_preds=reg_test_preds,
        approve_to_production=True,
    )

    horizon_res = run_horizon_ablation() if include_horizon_ablation else {"horizons": []}

    consolidated = {
        "primary_task": "multiclass_3",
        "primary_horizon_hours": PRIMARY_HORIZON_HOURS,
        "dataset_version": prep_meta.get("dataset_version"),
        "cutoff_date": prep_meta.get("cutoff_date"),
        "val_cutoff_date": prep_meta.get("val_cutoff_date"),
        "split_sizes": {
            "train": int(len(tr_df)),
            "val": int(len(va_df)),
            "test": int(len(te_df)),
        },
        "multiclass_3": primary_3class,
        "binary_disrupted": binary_ablation,
        "horizon_ablation": horizon_res,
    }
    write_json(REPORTS_DIR / "experiments" / "classification_results.json", consolidated)
    return consolidated
