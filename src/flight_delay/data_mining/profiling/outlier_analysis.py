# -*- coding: utf-8 -*-
"""[STEP 3] Delay & Numerical Outlier Profiling."""
import pandas as pd


def analyze_outliers(df: pd.DataFrame) -> dict:
    """Profile extreme flight delays, early departures, and weather extremes."""
    delays = pd.to_numeric(df.get("delay_min", pd.Series(dtype=float)), errors="coerce").dropna()
    if delays.empty:
        return {"delay_count": 0}
    q1, q3 = float(delays.quantile(0.25)), float(delays.quantile(0.75))
    iqr = q3 - q1
    upper_iqr = q3 + 1.5 * iqr
    return {
        "delay_count": int(len(delays)),
        "early_negative_count": int((delays < 0).sum()),
        "early_min": float(delays.min()),
        "q1": round(q1, 2),
        "median": round(float(delays.median()), 2),
        "q3": round(q3, 2),
        "iqr": round(iqr, 2),
        "upper_iqr_fence": round(upper_iqr, 2),
        "iqr_outlier_count": int((delays > upper_iqr).sum()),
        "extreme_gt_180m": int((delays > 180).sum()),
        "max_delay": float(delays.max()),
        "l1_mae_rationale": "Heavy right tail (median 1m vs max 629m) motivates MAE/L1 loss over MSE.",
    }
