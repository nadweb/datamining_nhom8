# -*- coding: utf-8 -*-
"""[STEP 5] Confusion Matrix & Mistake Pair Analysis."""
from typing import List

import numpy as np


def analyze_confusion_pairs(cm: list, class_names: List[str]) -> dict:
    mat = np.asarray(cm, dtype=float)
    total = float(mat.sum()) or 1.0
    pairs = []
    for i, actual in enumerate(class_names):
        row_sum = float(mat[i].sum()) or 1.0
        for j, pred in enumerate(class_names):
            if i != j and mat[i, j] > 0:
                pairs.append({
                    "actual": actual,
                    "predicted": pred,
                    "count": int(mat[i, j]),
                    "pct_of_actual_class": round(float(mat[i, j] / row_sum * 100), 2),
                    "pct_of_total": round(float(mat[i, j] / total * 100), 2),
                })
    pairs.sort(key=lambda x: x["count"], reverse=True)
    return {"top_confusions": pairs}
