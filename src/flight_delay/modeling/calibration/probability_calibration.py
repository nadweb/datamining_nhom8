# -*- coding: utf-8 -*-
"""[STEP 4] Multi-Class / Disruption Probability Reliability & Calibration Analysis."""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


def compute_probability_calibration(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 5,
) -> dict:
    """Compute Brier score for disruption risk, multi-class log loss, and quantile calibration bins."""
    y_arr = np.asarray(y_true, dtype=int)
    p_disrupted = probs[:, 1] if probs.shape[1] == 2 else (probs[:, 1] + probs[:, 2])
    y_disrupted = (y_arr >= 1).astype(int)

    df = pd.DataFrame({"p": p_disrupted, "y": y_disrupted}).sort_values("p", kind="stable")
    df["bin"] = pd.qcut(df["p"].rank(method="first"), q=min(n_bins, max(len(df), 1)), labels=False)
    bins = []
    for b, g in df.groupby("bin"):
        bins.append({
            "bin": int(b) + 1,
            "count": int(len(g)),
            "mean_predicted": round(float(g["p"].mean()), 4),
            "observed_rate": round(float(g["y"].mean()), 4),
        })

    n_cls = probs.shape[1]
    ll = float(log_loss(y_arr, probs, labels=list(range(n_cls))))
    brier = float(brier_score_loss(y_disrupted, np.clip(p_disrupted, 0.0, 1.0)))
    return {
        "brier_disrupted": round(brier, 4),
        "log_loss": round(ll, 4),
        "base_rate_disrupted": round(float(y_disrupted.mean()), 4),
        "bins": bins,
    }
