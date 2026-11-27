# -*- coding: utf-8 -*-
"""[STEP 3] Pearson, Spearman & Cramer's V Correlation Matrices."""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def cramers_v(x: pd.Series, y: pd.Series) -> dict:
    ct = pd.crosstab(x.astype(str), y.astype(str))
    n = ct.sum().sum()
    if n == 0 or min(ct.shape) <= 1:
        return {"v": 0.0, "chi2": 0.0, "p_value": 1.0}
    chi2, p, _, _ = chi2_contingency(ct)
    r, k = ct.shape
    v = np.sqrt(max(chi2 / n, 0.0) / max(min(k - 1, r - 1), 1))
    return {"v": round(float(v), 4), "chi2": round(float(chi2), 2), "p_value": float(p)}


def compute_correlation_suite(df: pd.DataFrame) -> dict:
    work = df.copy()
    work["is_disrupted"] = work["status_clean"].isin(["delayed", "cancelled"]).astype(float)
    work["is_cancelled"] = (work["status_clean"] == "cancelled").astype(float)

    num_cols = [
        c for c in [
            "is_disrupted", "delay_min", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "airport_load", "precipitation", "wind_gusts_10m", "visibility",
            "weather_hazard_score", "load_x_weather", "historical_airline_delay_rate",
            "historical_airport_delay_rate", "is_peak_hour", "is_golden_triangle",
        ] if c in work.columns
    ]
    num_df = work[num_cols].apply(pd.to_numeric, errors="coerce")
    pearson = num_df.corr(method="pearson").fillna(0.0).round(4)
    spearman = num_df.corr(method="spearman").fillna(0.0).round(4)

    cat_candidates = ["airline_code", "carrier_category", "time_slot", "airport", "day_type", "route_type", "direction"]
    cat_assoc = []
    for c in cat_candidates:
        if c in work.columns:
            res_3 = cramers_v(work[c], work["status_clean"])
            res_2 = cramers_v(work[c], work["is_disrupted"])
            cat_assoc.append({
                "feature": c,
                "n_unique": int(work[c].nunique(dropna=True)),
                "cramers_v_3class": res_3["v"],
                "cramers_v_disrupted": res_2["v"],
                "chi2_p_value": res_3["p_value"],
            })
    cat_assoc.sort(key=lambda r: r["cramers_v_disrupted"], reverse=True)

    return {
        "matrix_columns": num_cols,
        "pearson_matrix": pearson.values.tolist(),
        "spearman_matrix": spearman.values.tolist(),
        "target_pearson_ranked": [
            {"feature": col, "r": float(pearson.loc[col, "is_disrupted"])}
            for col in pearson.index if col != "is_disrupted"
        ],
        "categorical_cramers_v": cat_assoc,
    }
