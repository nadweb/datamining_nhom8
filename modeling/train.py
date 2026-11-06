# -*- coding: utf-8 -*-
"""Supervised Machine Learning & Data Splitting Module (Part 5A - Member 5)
Handles:
1. Chronological (Date-based) Train/Test Splitting:
   - Sort by flight_date and scheduled_time
   - 80% Train, 20% Test (Cutoff date: 2026-10-30)
   - Zero lookahead bias, strict chronological ordering
   - Generates data/processed/train.csv and data/processed/test.csv
2. Multi-class Status Classification (on_time, delayed, cancelled):
   - Baseline Logistic Regression (class_weight='balanced')
   - Balanced Random Forest Classifier
   - HistGradientBoosting Classifier
3. Delay Minutes Regression (Predict continuous delay magnitude for late flights):
   - HistGradientBoosting Regressor / Random Forest Regressor
4. Comprehensive Model Evaluation:
   - Accuracy, Precision (macro/weighted), Recall, F1-macro, ROC-AUC (ovr)
   - Confusion matrix & Per-class classification report
   - Top Feature importance extraction
5. Persistence of models and metrics to models/ directory
"""
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, mean_absolute_error, mean_squared_error, precision_score,
    r2_score, recall_score, roc_auc_score
)

from config import BASE_DIR
from modeling.encoding import (
    CATEGORICAL_COLS, FILE_METADATA, FILE_PROCESSED, FILE_TEST, FILE_TRAIN,
    INV_LABEL_MAP, LABEL_MAP, NUMERIC_CONTINUOUS_COLS, PASSTHROUGH_BINARY_COLS,
    PROCESSED_DIR, FlightDataPreprocessor
)

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FILE_CLASSIFIER = MODELS_DIR / "best_classifier.joblib"
FILE_REGRESSOR = MODELS_DIR / "best_regressor.joblib"
FILE_METRICS = MODELS_DIR / "supervised_metrics.json"


def split_by_date(
    df: pd.DataFrame, train_ratio: float = 0.80
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Perform chronological train-test split without shuffling (anti-lookahead)."""
    df_sorted = df.sort_values(["flight_date", "scheduled_time"]).reset_index(drop=True)
    split_idx = int(len(df_sorted) * train_ratio)
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    cutoff_date = str(train_df["flight_date"].max())
    return train_df, test_df, cutoff_date


def prepare_chronological_train_test(
    source_df: pd.DataFrame = None, train_ratio: float = 0.80
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Execute chronological train-test split and apply Member 4 preprocessor."""
    if source_df is None:
        features_path = PROCESSED_DIR / "flights_features.csv"
        if not features_path.exists():
            from features.features import run_feature_pipeline
            source_df = run_feature_pipeline()
        else:
            source_df = pd.read_csv(features_path)

    clean_df = source_df[source_df["status_clean"].isin(LABEL_MAP.keys())].copy()
    train_raw, test_raw, cutoff_date = split_by_date(clean_df, train_ratio=train_ratio)
    print(f"\n[Member 5 - Chronological Split] Cutoff date: {cutoff_date}")
    print(f"  Train set: {len(train_raw):,} flights ({train_ratio*100:.0f}%) | Test set: {len(test_raw):,} flights ({(1-train_ratio)*100:.0f}%)")

    y_train = train_raw["status_clean"].map(LABEL_MAP).astype(int)
    y_test = test_raw["status_clean"].map(LABEL_MAP).astype(int)
    delay_train = train_raw["delay_min"]
    delay_test = test_raw["delay_min"]

    # Fit Member 4 preprocessor on train only, transform both
    preprocessor = FlightDataPreprocessor()
    preprocessor.fit(train_raw)
    X_train_mat = preprocessor.transform(train_raw)
    X_test_mat = preprocessor.transform(test_raw)

    X_train = preprocessor.to_dataframe(X_train_mat)
    X_test = preprocessor.to_dataframe(X_test_mat)

    preprocessor.save(PROCESSED_DIR)

    # Save metadata
    meta = {
        "cutoff_date": cutoff_date,
        "train_rows": len(train_raw),
        "test_rows": len(test_raw),
        "total_features": len(preprocessor.feature_names),
        "feature_names": preprocessor.feature_names,
        "encoded_cat_names": preprocessor.encoded_cat_names,
        "categorical_columns": CATEGORICAL_COLS,
        "numeric_continuous_columns": NUMERIC_CONTINUOUS_COLS,
        "binary_columns": PASSTHROUGH_BINARY_COLS,
        "label_mapping": LABEL_MAP,
    }
    with open(FILE_METADATA, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Export split CSVs
    train_export = X_train.copy()
    train_export["target"] = y_train.values
    train_export["delay_min"] = delay_train.values
    train_export.to_csv(FILE_TRAIN, index=False)

    test_export = X_test.copy()
    test_export["target"] = y_test.values
    test_export["delay_min"] = delay_test.values
    test_export.to_csv(FILE_TEST, index=False)

    print(f"  Exported: {FILE_TRAIN.name} ({train_export.shape[0]}x{train_export.shape[1]})")
    print(f"  Exported: {FILE_TEST.name} ({test_export.shape[0]}x{test_export.shape[1]})")

    return X_train, X_test, y_train, y_test, delay_train, delay_test


def load_train_test_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Load train and test datasets, triggering chronological split if missing."""
    if not FILE_TRAIN.exists() or not FILE_TEST.exists():
        print("Train/test datasets not found. Executing Member 5 chronological split...")
        return prepare_chronological_train_test()

    train_df = pd.read_csv(FILE_TRAIN)
    test_df = pd.read_csv(FILE_TEST)

    feature_cols = [c for c in train_df.columns if c not in ["target", "delay_min"]]

    X_train = train_df[feature_cols]
    y_train = train_df["target"].astype(int)
    delay_train = train_df["delay_min"]

    X_test = test_df[feature_cols]
    y_test = test_df["target"].astype(int)
    delay_test = test_df["delay_min"]

    return X_train, X_test, y_train, y_test, delay_train, delay_test


def train_classification_models(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> Tuple[object, Dict[str, dict]]:
    """Train multiple classifiers, compare on test set, and select the best model."""
    print("\n" + "=" * 65)
    print("      TRAINING SUPERVISED CLASSIFICATION MODELS (3-CLASS)")
    print("=" * 65)

    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=150, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=150, class_weight="balanced", random_state=42
        ),
    }

    results = {}
    best_model = None
    best_f1 = -1.0
    best_name = ""

    for name, model in candidates.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        p_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        r_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # ROC-AUC multi-class (one-vs-rest)
        try:
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr") if y_prob is not None else 0.0
        except Exception:
            auc = 0.0

        cm = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(
            y_test, y_pred, target_names=["on_time", "delayed", "cancelled"], output_dict=True, zero_division=0
        )

        metrics = {
            "accuracy": round(float(acc), 4),
            "precision_macro": round(float(p_macro), 4),
            "recall_macro": round(float(r_macro), 4),
            "f1_macro": round(float(f1_mac), 4),
            "f1_weighted": round(float(f1_weighted), 4),
            "roc_auc_ovr": round(float(auc), 4),
            "confusion_matrix": cm,
            "classification_report": report,
        }
        results[name] = metrics

        print(f"  Accuracy       : {acc*100:.2f}%")
        print(f"  F1-Score (Macro): {f1_mac:.4f} | F1-Score (Weighted): {f1_weighted:.4f}")
        print(f"  Recall (Macro) : {r_macro:.4f} | Precision (Macro) : {p_macro:.4f}")
        print(f"  ROC-AUC (ovr)  : {auc:.4f}")

        if f1_mac > best_f1:
            best_f1 = f1_mac
            best_model = model
            best_name = name

    print(f"\n>>> Selected Best Classifier: {best_name} (F1-Macro = {best_f1:.4f})")
    results["best_classifier_name"] = best_name
    return best_model, results


def train_delay_regression_model(
    X_train: pd.DataFrame, X_test: pd.DataFrame, delay_train: pd.Series, delay_test: pd.Series
) -> Tuple[object, dict]:
    """Train regression model to estimate continuous delay minutes."""
    print("\n" + "=" * 65)
    print("      TRAINING SUPERVISED DELAY MINUTES REGRESSION MODEL")
    print("=" * 65)

    # Filter out missing delays (cancelled flights have no delay_min)
    train_mask = delay_train.notna()
    test_mask = delay_test.notna()

    X_tr_reg = X_train[train_mask]
    y_tr_reg = delay_train[train_mask]

    X_te_reg = X_test[test_mask]
    y_te_reg = delay_test[test_mask]

    regressor = HistGradientBoostingRegressor(max_iter=150, random_state=42)
    regressor.fit(X_tr_reg, y_tr_reg)

    y_pred_reg = regressor.predict(X_te_reg)

    mae = mean_absolute_error(y_te_reg, y_pred_reg)
    rmse = np.sqrt(mean_squared_error(y_te_reg, y_pred_reg))
    r2 = r2_score(y_te_reg, y_pred_reg)

    reg_metrics = {
        "MAE_minutes": round(float(mae), 2),
        "RMSE_minutes": round(float(rmse), 2),
        "R2_Score": round(float(r2), 4),
    }

    print(f"  MAE (Mean Absolute Error): {mae:.2f} minutes")
    print(f"  RMSE                    : {rmse:.2f} minutes")
    print(f"  R2 Score                : {r2:.4f}")

    return regressor, reg_metrics


def extract_feature_importances(model, feature_names) -> list:
    """Extract top predictive features if available on model."""
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
        sorted_pairs = sorted(zip(feature_names, imps), key=lambda x: x[1], reverse=True)
        return [{"feature": f, "importance": round(float(val), 4)} for f, val in sorted_pairs[:20]]
    return []


def run_supervised_training():
    """Execute complete supervised training pipeline."""
    X_train, X_test, y_train, y_test, delay_train, delay_test = load_train_test_data()

    best_clf, clf_results = train_classification_models(X_train, X_test, y_train, y_test)
    best_reg, reg_metrics = train_delay_regression_model(X_train, X_test, delay_train, delay_test)

    feature_importances = extract_feature_importances(best_clf, list(X_train.columns))
    if feature_importances:
        print("\nTop 10 Most Important Features:")
        for rank, item in enumerate(feature_importances[:10], 1):
            print(f"  {rank:2d}. {item['feature']:<30}: {item['importance']*100:.2f}%")

    # Persist models and metrics
    joblib.dump(best_clf, FILE_CLASSIFIER)
    joblib.dump(best_reg, FILE_REGRESSOR)

    all_metrics = {
        "classification": clf_results,
        "regression": reg_metrics,
        "feature_importances": feature_importances,
    }
    with open(FILE_METRICS, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n" + "=" * 65)
    print(f"Saved best classifier to: {FILE_CLASSIFIER}")
    print(f"Saved best regressor to : {FILE_REGRESSOR}")
    print(f"Saved metrics summary to: {FILE_METRICS}")
    print("=" * 65)


if __name__ == "__main__":
    run_supervised_training()

