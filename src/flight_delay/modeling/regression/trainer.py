# -*- coding: utf-8 -*-
"""[STEP 4] Tabular L1/MAE Delay Duration Regression Trainer (Completed Flights Only)."""
import logging
import time
from typing import Dict

import numpy as np
import pandas as pd

from flight_delay.common.constants import (
    EXPERIMENTS_CFG,
    FILE_GOLD,
    PRIMARY_HORIZON_HOURS,
    PROCESSED_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
)
from flight_delay.common.utils import read_json, write_json
from flight_delay.data_mining.feature_engineering.registry import FEATURE_REGISTRY, assert_no_leakage
from flight_delay.evaluation.regression_metrics import (
    compute_regression_metrics,
    compute_regression_subgroups,
)
from flight_delay.evaluation.statistical_comparison import paired_bootstrap_ci
from flight_delay.governance.approval import validate_and_approve_model
from flight_delay.governance.artifact_manager import package_model_bundle
from flight_delay.governance.experiment_tracker import log_experiment_run
from flight_delay.modeling.explainability.feature_importance import extract_model_feature_importances
from flight_delay.modeling.regression.delay_regressor import build_regression_candidates

log = logging.getLogger(__name__)


def _filter_completed(df_mat: pd.DataFrame, gold_split: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Keep strictly on_time & delayed rows with numeric finite delay_min (exclude cancelled)."""
    valid_status = gold_split["status_clean"].isin(["on_time", "delayed"]).values
    delay_vals = pd.to_numeric(df_mat["delay_min"], errors="coerce")
    mask = valid_status & delay_vals.notna().values
    return (
        df_mat.loc[mask].reset_index(drop=True),
        gold_split.loc[mask].reset_index(drop=True),
        delay_vals.loc[mask].astype(float).reset_index(drop=True),
    )


def run_regression_suite() -> dict:
    """Train & evaluate L1/MAE regressors on completed flights, approve winning bundle."""
    tr_raw = pd.read_csv(PROCESSED_DIR / "train.csv")
    va_raw = pd.read_csv(PROCESSED_DIR / "val.csv")
    te_raw = pd.read_csv(PROCESSED_DIR / "test.csv")
    gold_df = pd.read_csv(FILE_GOLD, low_memory=False)
    prep_meta = read_json(PROCESSED_DIR / "preprocessing_metadata.json")
    feature_cols = list(prep_meta["feature_names"])
    assert_no_leakage(feature_cols)

    gold_tr = gold_df[gold_df["split_role"] == "train"].reset_index(drop=True)
    gold_va = gold_df[gold_df["split_role"] == "val"].reset_index(drop=True)
    gold_te = gold_df[gold_df["split_role"] == "test"].reset_index(drop=True)

    tr_mat, _, y_tr = _filter_completed(tr_raw, gold_tr)
    va_mat, _, y_va = _filter_completed(va_raw, gold_va)
    te_mat, te_gold_clean, y_te = _filter_completed(te_raw, gold_te)

    X_tr = tr_mat[feature_cols]
    X_va = va_mat[feature_cols]
    X_te = te_mat[feature_cols]

    candidates = build_regression_candidates()
    fitted = {}
    val_metrics: Dict[str, dict] = {}
    test_metrics: Dict[str, dict] = {}
    test_preds: Dict[str, np.ndarray] = {}

    for name, est in candidates.items():
        t0 = time.perf_counter()
        est.fit(X_tr, y_tr)
        fit_ms = (time.perf_counter() - t0) * 1000.0
        fitted[name] = est

        va_p = est.predict(X_va)
        val_m = compute_regression_metrics(y_va.values, va_p)
        val_m["fit_ms"] = round(fit_ms, 2)
        val_metrics[name] = val_m

        t1 = time.perf_counter()
        te_p = est.predict(X_te)
        infer_ms = (time.perf_counter() - t1) * 1000.0
        te_m = compute_regression_metrics(y_te.values, te_p)
        te_m["fit_ms"] = round(fit_ms, 2)
        te_m["infer_ms"] = round(infer_ms, 2)
        test_metrics[name] = te_m
        test_preds[name] = te_p

        log_experiment_run(
            category="regression",
            run_name=f"delay_min_{name}",
            params={"algorithm": name, "loss": "MAE"},
            metrics={"val_mae": val_m["MAE_minutes"], "test_mae": te_m["MAE_minutes"], "test_medae": te_m["MedAE_minutes"]},
        )

    ml_names = [n for n in candidates if n != "MedianBaseline"]
    best_name = min(ml_names, key=lambda n: (val_metrics[n]["MAE_minutes"], val_metrics[n]["MedAE_minutes"]))
    best_model = fitted[best_name]
    best_te_pred = test_preds[best_name]

    # Map predictions aligned back to full Holdout Test index for subgroup cross-table
    full_test_reg_pred = best_model.predict(te_raw[feature_cols])

    subgroups = compute_regression_subgroups(te_gold_clean, y_te.values, best_te_pred)
    stat_comp = paired_bootstrap_ci(y_te.values, test_preds["MedianBaseline"], best_te_pred, metric="mae", seed=RANDOM_STATE)
    feat_imp = extract_model_feature_importances(best_model, feature_cols, FEATURE_REGISTRY.categorical_columns)

    feature_schema = {
        "schema_version": FEATURE_REGISTRY.schema_version,
        "feature_names": feature_cols,
        "categorical_columns": FEATURE_REGISTRY.categorical_columns,
        "numeric_columns": FEATURE_REGISTRY.numeric_columns,
        "binary_columns": FEATURE_REGISTRY.binary_columns,
        "total_features": len(feature_cols),
    }
    eval_payload = {
        "task": "delay_regression_mae",
        "best_model": best_name,
        "exclusion_policy": "Strictly excluded cancelled, unknown, and diverted flights (never set cancelled delay_min=0).",
        "regression": test_metrics[best_name],
        "validation": val_metrics[best_name],
        "subgroups": subgroups,
        "bootstrap_vs_median": stat_comp,
        "feature_importance": feat_imp,
    }
    package_model_bundle(
        model=best_model,
        model_name="flight_delay_regressor_mae",
        version="v2.0.0",
        algorithm=best_name,
        task="delay_regression_mae",
        dataset_version=str(prep_meta.get("dataset_version", "unknown")),
        feature_schema=feature_schema,
        label_mapping={"target": "delay_min (minutes)", "excluded_statuses": list(EXPERIMENTS_CFG["regression"]["exclude_statuses"])},
        evaluation=eval_payload,
        prediction_horizon_hours=int(prep_meta.get("horizon_hours", PRIMARY_HORIZON_HOURS)),
        training_cutoff=str(prep_meta.get("cutoff_date", "")),
        random_seed=RANDOM_STATE,
    )
    reg_entry = validate_and_approve_model(
        model_name="flight_delay_regressor_mae",
        version="v2.0.0",
        reviewer="automated-governance-gate",
        promote_to_production=True,
    )

    payload = {
        "best_model": best_name,
        "registry_entry": reg_entry,
        "completed_counts": {"train": int(len(tr_mat)), "val": int(len(va_mat)), "test": int(len(te_mat))},
        "excluded_counts": {
            "train": int(len(tr_raw) - len(tr_mat)),
            "val": int(len(va_raw) - len(va_mat)),
            "test": int(len(te_raw) - len(te_mat)),
        },
        "validation_comparison": val_metrics,
        "test_comparison": test_metrics,
        "subgroups": subgroups,
        "bootstrap_vs_median": stat_comp,
        "feature_importance": feat_imp,
        "full_test_predictions": [round(float(x), 2) for x in full_test_reg_pred],
    }
    write_json(REPORTS_DIR / "experiments" / "regression_results.json", payload)
    return payload
