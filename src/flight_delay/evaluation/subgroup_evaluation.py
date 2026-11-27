# -*- coding: utf-8 -*-
"""[STEP 5] Operational Subgroup Evaluation (Airport, Airline, Peak, DayType, Weather, Route)."""
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def _wilson_ci(k: int, n: int, z: float = 1.96) -> List[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return [round(max(0.0, float(center - half)) * 100, 2), round(min(1.0, float(center + half)) * 100, 2)]


def evaluate_subgroup_performance(
    gold_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_reg_pred: np.ndarray = None,
    labels: List[int] = (0, 1, 2),
) -> dict:
    """Audit classifier & MAE performance across Vietnamese operational subgroups with CIs."""
    df = gold_df.reset_index(drop=True).copy()
    df["_yt"] = np.asarray(y_true, dtype=int)
    df["_yp"] = np.asarray(y_pred, dtype=int)
    if y_reg_pred is not None:
        df["_yreg"] = np.asarray(y_reg_pred, dtype=float)

    df["weather_group"] = np.where(
        df.get("is_heavy_rain", 0).astype(int) == 1,
        "heavy_rain_or_storm",
        np.where((df.get("is_rain", 0).astype(int) == 1) | (df.get("is_high_wind", 0).astype(int) == 1), "rain_or_windy", "clear"),
    )
    df["peak_group"] = np.where(df.get("is_peak_hour", 0).astype(int) == 1, "peak_hour", "off_peak")

    def _slice(col: str, top_keys: list = None) -> Dict[str, dict]:
        if col not in df.columns:
            return {}
        keys = top_keys if top_keys is not None else sorted(df[col].dropna().astype(str).unique().tolist())
        out = {}
        for k in keys:
            sub = df[df[col].astype(str) == str(k)]
            if sub.empty:
                continue
            n = len(sub)
            acc = accuracy_score(sub["_yt"], sub["_yp"])
            bal = balanced_accuracy_score(sub["_yt"], sub["_yp"]) if sub["_yt"].nunique() > 1 else acc
            f1m = f1_score(sub["_yt"], sub["_yp"], average="macro", zero_division=0)
            ci = _wilson_ci(int((sub["_yt"] == sub["_yp"]).sum()), n)
            valid_d = sub[sub["status_clean"] != "cancelled"] if "status_clean" in sub else sub
            mae_val = None
            if "_yreg" in valid_d and "delay_min" in valid_d:
                m = pd.to_numeric(valid_d["delay_min"], errors="coerce").notna()
                if m.any():
                    mae_val = round(float(np.abs(pd.to_numeric(valid_d.loc[m, "delay_min"]) - valid_d.loc[m, "_yreg"]).mean()), 2)
            out[str(k)] = {
                "count": int(n),
                "accuracy": round(float(acc * 100), 2),
                "accuracy_ci_95": ci,
                "balanced_accuracy": round(float(bal * 100), 2),
                "macro_f1": round(float(f1m * 100), 2),
                "mae_model": mae_val,
                "small_sample_warning": bool(n < 30),
            }
        return out

    return {
        "by_airport": _slice("airport", ["SGN", "HAN", "DAD", "CXR", "PQC"]),
        "by_airline": _slice("airline_code"),
        "by_peak_hour": _slice("peak_group"),
        "by_day_type": _slice("day_type"),
        "by_weather_group": _slice("weather_group"),
        "by_route_type": _slice("route_type"),
    }
