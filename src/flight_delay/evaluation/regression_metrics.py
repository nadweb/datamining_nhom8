# -*- coding: utf-8 -*-
"""[STEP 5] Regression Metrics (MAE, RMSE, MedAE, R2 + Slice Breakdowns)."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return {
        "MAE_minutes": round(float(mean_absolute_error(yt, yp)), 2),
        "RMSE_minutes": round(float(np.sqrt(mean_squared_error(yt, yp))), 2),
        "MedAE_minutes": round(float(median_absolute_error(yt, yp)), 2),
        "R2_Score": round(float(r2_score(yt, yp)), 4),
    }


def compute_regression_subgroups(gold_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    work = gold_df.copy().reset_index(drop=True)
    work["_yt"] = np.asarray(y_true, dtype=float)
    work["_yp"] = np.asarray(y_pred, dtype=float)
    work["_err"] = np.abs(work["_yt"] - work["_yp"])

    bins = [-1000, -0.01, 14.99, 60.0, 1e9]
    labels = ["early (<0m)", "on_time (0-14m)", "moderate_delay (15-60m)", "severe_delay (>60m)"]
    work["delay_bucket"] = pd.cut(work["_yt"], bins=bins, labels=labels)

    def _agg(col: str) -> list:
        if col not in work.columns:
            return []
        rows = []
        for val, g in work.groupby(col, observed=True):
            if len(g) == 0:
                continue
            rows.append({
                col: str(val),
                "count": int(len(g)),
                "mae_minutes": round(float(g["_err"].mean()), 2),
                "medae_minutes": round(float(g["_err"].median()), 2),
                "mean_actual_delay": round(float(g["_yt"].mean()), 2),
            })
        return sorted(rows, key=lambda r: r["count"], reverse=True)

    return {
        "by_airline": _agg("airline_code"),
        "by_airport": _agg("airport"),
        "by_time_slot": _agg("time_slot"),
        "by_delay_severity": _agg("delay_bucket"),
    }
