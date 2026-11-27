# -*- coding: utf-8 -*-
"""[STEP 4] Aggregated Tree Gain / Split Feature Importances."""
from typing import Dict, List

import numpy as np


def aggregate_one_hot_feature_name(col: str, categorical_prefixes: List[str]) -> str:
    for p in categorical_prefixes:
        if col.startswith(f"{p}_"):
            return p
    return col


def extract_model_feature_importances(
    model,
    feature_names: List[str],
    categorical_cols: List[str],
    top_n: int = 15,
) -> dict:
    raw_imp = getattr(model, "feature_importances_", None)
    if raw_imp is None:
        return {"raw_top": [], "grouped_top": []}
    imp = np.asarray(raw_imp, dtype=float)
    total = float(imp.sum()) or 1.0
    norm = imp / total * 100.0

    raw_rows = sorted(
        [{"feature": col, "importance_pct": round(float(val), 2)} for col, val in zip(feature_names, norm)],
        key=lambda r: r["importance_pct"],
        reverse=True,
    )
    grouped: Dict[str, float] = {}
    for col, val in zip(feature_names, norm):
        g_name = aggregate_one_hot_feature_name(col, categorical_cols)
        grouped[g_name] = grouped.get(g_name, 0.0) + float(val)
    grouped_rows = sorted(
        [{"feature": k, "importance_pct": round(v, 2)} for k, v in grouped.items()],
        key=lambda r: r["importance_pct"],
        reverse=True,
    )
    return {"raw_top": raw_rows[:top_n], "grouped_top": grouped_rows[:top_n]}
