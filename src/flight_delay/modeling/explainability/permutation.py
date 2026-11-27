# -*- coding: utf-8 -*-
"""[STEP 4] Permutation Importance & Prediction Significance Testing."""
from typing import List

import pandas as pd
from sklearn.inspection import permutation_importance


def compute_feature_permutation_importance(model, X: pd.DataFrame, y: pd.Series, feature_names: List[str], seed: int = 42, n_repeats: int = 5) -> list:
    res = permutation_importance(model, X, y, scoring="balanced_accuracy", n_repeats=n_repeats, random_state=seed, n_jobs=-1)
    rows = []
    for col, mean_imp, std_imp in zip(feature_names, res.importances_mean, res.importances_std):
        rows.append({
            "feature": col,
            "mean_drop_bal_acc": round(float(mean_imp), 5),
            "std_drop_bal_acc": round(float(std_imp), 5),
        })
    return sorted(rows, key=lambda r: r["mean_drop_bal_acc"], reverse=True)[:15]
