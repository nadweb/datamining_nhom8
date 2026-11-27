# -*- coding: utf-8 -*-
"""[STEP 7] Feature Data Drift Monitor (PSI + Kolmogorov-Smirnov + Chi-Square)."""
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from flight_delay.common.constants import FILE_GOLD, MONITORING_CFG
from flight_delay.monitoring.psi import compute_psi

MONITOR_NUM_FEATURES = [
    "airport_load",
    "airport_congestion",
    "precipitation",
    "wind_gusts_10m",
    "visibility",
    "weather_hazard_score",
    "historical_airline_delay_rate",
    "historical_airport_delay_rate",
]
MONITOR_CAT_FEATURES = ["airline_code", "airport", "route_type", "direction"]


def run_data_drift_monitor(ref_df: Optional[pd.DataFrame] = None, cur_df: Optional[pd.DataFrame] = None) -> dict:
    if ref_df is None or cur_df is None:
        gold = pd.read_csv(FILE_GOLD, low_memory=False)
        if "split_role" in gold.columns:
            ref_df = gold[gold["split_role"] != "test"].copy()
            cur_df = gold[gold["split_role"] == "test"].copy()
        else:
            cut = int(len(gold) * 0.8)
            ref_df, cur_df = gold.iloc[:cut].copy(), gold.iloc[cut:].copy()

    thresholds = MONITORING_CFG["psi_thresholds"]
    stable_max = float(thresholds["stable"])
    mod_max = float(thresholds["moderate"])
    alpha = float(MONITORING_CFG.get("ks_test_alpha", 0.05))

    feature_reports = {}
    max_psi = 0.0
    for col in MONITOR_NUM_FEATURES:
        if col not in ref_df.columns or col not in cur_df.columns:
            continue
        r_vals = pd.to_numeric(ref_df[col], errors="coerce").dropna().values
        c_vals = pd.to_numeric(cur_df[col], errors="coerce").dropna().values
        if len(r_vals) == 0 or len(c_vals) == 0:
            continue
        psi_val = compute_psi(r_vals, c_vals, bins=10)
        ks_res = ks_2samp(r_vals, c_vals)
        max_psi = max(max_psi, psi_val)
        status = "CRITICAL" if psi_val >= mod_max else "WARNING" if psi_val >= stable_max else "STABLE"
        feature_reports[col] = {
            "psi": psi_val,
            "ks_statistic": round(float(ks_res.statistic), 4),
            "ks_p_value": round(float(ks_res.pvalue), 6),
            "ks_drift_detected": bool(ks_res.pvalue < alpha and psi_val >= stable_max),
            "status": status,
            "train_mean": round(float(np.mean(r_vals)), 3),
            "test_mean": round(float(np.mean(c_vals)), 3),
            "train_std": round(float(np.std(r_vals)), 3),
            "test_std": round(float(np.std(c_vals)), 3),
        }

    categorical_reports = {}
    for col in MONITOR_CAT_FEATURES:
        if col not in ref_df.columns or col not in cur_df.columns:
            continue
        r_cnt = ref_df[col].astype(str).value_counts()
        c_cnt = cur_df[col].astype(str).value_counts()
        cats = sorted(set(r_cnt.index) | set(c_cnt.index))
        table = [[int(r_cnt.get(k, 0)), int(c_cnt.get(k, 0))] for k in cats]
        try:
            chi2_stat, p_val, _, _ = chi2_contingency(table)
        except Exception:
            chi2_stat, p_val = 0.0, 1.0
        categorical_reports[col] = {
            "chi2_statistic": round(float(chi2_stat), 3),
            "p_value": round(float(p_val), 6),
            "shift_detected": bool(p_val < alpha),
        }

    if max_psi >= mod_max:
        overall = "CRITICAL"
    elif max_psi >= stable_max:
        overall = "WARNING"
    else:
        overall = "STABLE"

    return {
        "overall_status": overall,
        "max_psi": round(max_psi, 4),
        "psi_thresholds": thresholds,
        "ks_alpha": alpha,
        "ref_rows": int(len(ref_df)),
        "cur_rows": int(len(cur_df)),
        "feature_drift": feature_reports,
        "categorical_drift": categorical_reports,
    }
