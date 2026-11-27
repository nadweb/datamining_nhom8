# -*- coding: utf-8 -*-
"""[STEP 5] Qualitative Error Analysis for High-Confidence Misses & Extreme Delay Errors."""
from typing import List

import numpy as np
import pandas as pd


def analyze_prediction_errors(
    gold_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
    top_k: int = 10,
) -> dict:
    df = gold_df.reset_index(drop=True).copy()
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    p_bad = probs[:, 1] if probs.shape[1] == 2 else (probs[:, 1] + probs[:, 2])

    df["_yt"] = [class_names[i] if i < len(class_names) else str(i) for i in yt]
    df["_yp"] = [class_names[i] if i < len(class_names) else str(i) for i in yp]
    df["_p_disr"] = np.round(p_bad, 4)

    cols = [c for c in ("leg_key", "airline_code", "airport", "scheduled_time", "weather_hazard_score", "airport_load", "_yt", "_yp", "_p_disr", "delay_min") if c in df.columns]
    # False alarms on true on_time flights with highest predicted disruption risk
    fp = df[(yt == 0) & (yp != 0)].sort_values("_p_disr", ascending=False)[cols].head(top_k)
    # Missed disruptions (especially cancelled or heavy delayed) with lowest predicted disruption risk
    fn = df[(yt != 0) & (yp == 0)].sort_values("_p_disr", ascending=True)[cols].head(top_k)

    return {
        "total_test_samples": int(len(df)),
        "total_errors": int((yt != yp).sum()),
        "high_confidence_false_positives": fp.fillna("").to_dict("records"),
        "missed_disruptions_false_negatives": fn.fillna("").to_dict("records"),
    }
