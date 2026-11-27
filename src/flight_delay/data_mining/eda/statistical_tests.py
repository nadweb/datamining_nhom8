# -*- coding: utf-8 -*-
"""[STEP 3] Non-Parametric Statistical Significance Tests for EDA Signals."""
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu


def run_eda_hypothesis_tests(df: pd.DataFrame) -> dict:
    """Verify weather, airport_load, and departure hour differences across outcomes."""
    results = {}
    num_cols = [c for c in ("airport_load", "precipitation", "wind_gusts_10m", "visibility", "weather_hazard_score", "scheduled_hour_num") if c in df.columns]
    disrupted_mask = df["status_clean"].isin(["delayed", "cancelled"])

    for col in num_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        g_ok = s[~disrupted_mask].dropna()
        g_bad = s[disrupted_mask].dropna()
        if len(g_ok) >= 10 and len(g_bad) >= 10:
            stat_u, p_u = mannwhitneyu(g_bad, g_ok, alternative="two-sided")
            groups_3 = [s[df["status_clean"] == lbl].dropna() for lbl in ("on_time", "delayed", "cancelled")]
            groups_3 = [g for g in groups_3 if len(g) >= 5]
            kw_p = float(kruskal(*groups_3).pvalue) if len(groups_3) >= 2 else None
            results[col] = {
                "on_time_median": round(float(g_ok.median()), 4),
                "disrupted_median": round(float(g_bad.median()), 4),
                "mannwhitney_u_p": float(p_u),
                "kruskal_3class_p": kw_p,
                "significant_005": bool(p_u < 0.05),
            }
    return results
