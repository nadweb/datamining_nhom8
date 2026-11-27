# -*- coding: utf-8 -*-
"""[STEP 4] SHAP TreeExplainer Summary Values for Tabular GBDT Interpretability."""
from typing import List

import numpy as np
import pandas as pd
import shap


def compute_shap_summary(
    model,
    X_sample: pd.DataFrame,
    feature_names: List[str],
    class_names: List[str],
    max_samples: int = 400,
    seed: int = 42,
) -> dict:
    """Compute mean(|SHAP|) global and per-class attributions via TreeExplainer."""
    if len(X_sample) > max_samples:
        X_use = X_sample.sample(n=max_samples, random_state=seed)
    else:
        X_use = X_sample

    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_use)
    except Exception as exc:
        return {"available": False, "reason": str(exc), "global_top_features": []}

    # Handle shap >= 0.45 3D ndarray (N, F, C) or legacy list of (N, F) arrays
    if isinstance(sv, list):
        arrs = [np.asarray(a, dtype=float) for a in sv]
        stacked = np.stack(arrs, axis=-1)
    else:
        arr = np.asarray(sv, dtype=float)
        if arr.ndim == 2:
            stacked = arr[:, :, np.newaxis]
        else:
            stacked = arr

    global_mean = np.mean(np.abs(stacked), axis=(0, 2))
    order = np.argsort(global_mean)[::-1][:15]
    top = []
    for idx in order:
        item = {"feature": feature_names[idx], "mean_abs_shap": round(float(global_mean[idx]), 5)}
        for c_i, c_name in enumerate(class_names):
            if c_i < stacked.shape[2]:
                item[f"mean_abs_shap_{c_name}"] = round(float(np.mean(np.abs(stacked[:, idx, c_i]))), 5)
        top.append(item)
    return {"available": True, "sample_size": int(len(X_use)), "global_top_features": top}
