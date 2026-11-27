# -*- coding: utf-8 -*-
"""[STEP 5] Expanding Window Temporal Fold Backtesting."""
from typing import Callable, List

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score


def run_temporal_fold_backtest(
    X_train_val: pd.DataFrame,
    y_train_val: pd.Series,
    build_model_fn: Callable[[], object],
    folds: List[tuple],
    sample_weights_fn: Callable[[pd.Series], np.ndarray] = None,
) -> dict:
    """Evaluate stability across expanding chronological folds inside pre-holdout window."""
    fold_results = []
    for idx, (tr_idx, va_idx) in enumerate(folds, start=1):
        X_tr, y_tr = X_train_val.loc[tr_idx], y_train_val.loc[tr_idx]
        X_va, y_va = X_train_val.loc[va_idx], y_train_val.loc[va_idx]
        est = build_model_fn()
        if sample_weights_fn is not None and "XGB" in type(est).__name__:
            est.fit(X_tr, y_tr, sample_weight=sample_weights_fn(y_tr))
        else:
            est.fit(X_tr, y_tr)
        preds = est.predict(X_va)
        fold_results.append({
            "fold": idx,
            "train_size": int(len(X_tr)),
            "val_size": int(len(X_va)),
            "val_f1_macro": round(float(f1_score(y_va, preds, average="macro", zero_division=0)), 4),
            "val_balanced_accuracy": round(float(balanced_accuracy_score(y_va, preds)), 4),
        })

    f1s = [r["val_f1_macro"] for r in fold_results]
    bals = [r["val_balanced_accuracy"] for r in fold_results]
    return {
        "folds": fold_results,
        "cv_f1_macro_mean": round(float(np.mean(f1s)), 4) if f1s else 0.0,
        "cv_f1_macro_std": round(float(np.std(f1s)), 4) if f1s else 0.0,
        "cv_bal_acc_mean": round(float(np.mean(bals)), 4) if bals else 0.0,
    }
