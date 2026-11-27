# -*- coding: utf-8 -*-
"""[STEP 3] Bivariate Operational Delay/Cancellation Breakdowns."""
import pandas as pd


def _group_outcome_stats(df: pd.DataFrame, col: str) -> list:
    if col not in df.columns:
        return []
    rows = []
    for key, g in df.groupby(col, dropna=False):
        n = len(g)
        vc = g["status_clean"].value_counts()
        rows.append({
            col: str(key),
            "total": int(n),
            "on_time_pct": round(float(vc.get("on_time", 0) / n * 100), 2),
            "delayed_pct": round(float(vc.get("delayed", 0) / n * 100), 2),
            "cancelled_pct": round(float(vc.get("cancelled", 0) / n * 100), 2),
            "disrupted_pct": round(float((vc.get("delayed", 0) + vc.get("cancelled", 0)) / n * 100), 2),
            "mean_delay_min": round(float(pd.to_numeric(g["delay_min"], errors="coerce").dropna().mean()), 2)
            if pd.to_numeric(g["delay_min"], errors="coerce").notna().any() else None,
        })
    return sorted(rows, key=lambda r: r["total"], reverse=True)


def compute_bivariate_breakdowns(df: pd.DataFrame) -> dict:
    return {
        "by_airline": _group_outcome_stats(df, "airline_code"),
        "by_airport": _group_outcome_stats(df, "airport"),
        "by_time_slot": _group_outcome_stats(df, "time_slot"),
        "by_carrier_category": _group_outcome_stats(df, "carrier_category"),
        "by_route_type": _group_outcome_stats(df, "route_type"),
        "by_day_type": _group_outcome_stats(df, "day_type"),
    }
