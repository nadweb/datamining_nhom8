# -*- coding: utf-8 -*-
"""[STEP 5] Imbalanced Classification Metrics (Macro-F1, BalAcc, Per-Class Recall, PR-AUC, LogLoss)."""
from typing import List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
    infer_ms: float = 0.0,
) -> dict:
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    labels = list(range(len(class_names)))
    n_cls = len(class_names)

    rep = classification_report(y_t, y_p, labels=labels, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_t, y_p, labels=labels).tolist()

    # PR-AUC per class
    pr_auc_per_class = {}
    for k, name in enumerate(class_names):
        yb = (y_t == k).astype(int)
        if yb.sum() > 0 and probs.shape[1] > k:
            pr_auc_per_class[name] = round(float(average_precision_score(yb, probs[:, k])), 4)
        else:
            pr_auc_per_class[name] = None

    valid_prs = [v for v in pr_auc_per_class.values() if v is not None]
    macro_pr_auc = round(float(np.mean(valid_prs)), 4) if valid_prs else None

    if n_cls == 2:
        try:
            roc = round(float(roc_auc_score(y_t, probs[:, 1])), 4)
        except Exception:
            roc = None
        disr_probs = probs[:, 1]
    else:
        try:
            roc = round(float(roc_auc_score(y_t, probs, multi_class="ovr", average="macro")), 4)
        except Exception:
            roc = None
        disr_probs = probs[:, 1] + probs[:, 2]

    brier_disrupted = round(float(brier_score_loss((y_t >= 1).astype(int), np.clip(disr_probs, 0.0, 1.0))), 4)
    ll = round(float(log_loss(y_t, probs, labels=labels)), 4)

    out = {
        "accuracy": round(float(accuracy_score(y_t, y_p)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_t, y_p)), 4),
        "precision_macro": round(float(precision_score(y_t, y_p, labels=labels, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_t, y_p, labels=labels, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_t, y_p, labels=labels, average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_t, y_p, labels=labels, average="weighted", zero_division=0)), 4),
        "roc_auc": roc,
        "roc_auc_ovr": roc,
        "pr_auc_macro": macro_pr_auc,
        "pr_auc_per_class": pr_auc_per_class,
        "log_loss": ll,
        "brier_disrupted": brier_disrupted,
        "confusion_matrix": cm,
        "classification_report": rep,
        "infer_ms": round(float(infer_ms), 2),
        "test_infer_ms": round(float(infer_ms), 2),
    }
    # Explicit per-class recalls (Section 4.1 requirement: report DELAYED & CANCELLED recall separately)
    for name in class_names:
        c_rep = rep.get(name, {})
        out[f"{name}_precision"] = round(float(c_rep.get("precision", 0.0)), 4)
        out[f"{name}_recall"] = round(float(c_rep.get("recall", 0.0)), 4)
        out[f"{name}_f1"] = round(float(c_rep.get("f1-score", 0.0)), 4)
        out[f"{name}_support"] = int(c_rep.get("support", 0))
    if "delayed" in pr_auc_per_class:
        out["average_precision_delayed"] = pr_auc_per_class["delayed"]
    if "disrupted" in pr_auc_per_class:
        out["average_precision_disrupted"] = pr_auc_per_class["disrupted"]
    return out
