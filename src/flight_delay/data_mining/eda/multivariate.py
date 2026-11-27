# -*- coding: utf-8 -*-
"""[STEP 3] Multivariate Operational Context Interactions."""
import pandas as pd


def compute_multivariate_interactions(df: pd.DataFrame) -> dict:
    out = {}
    if {"carrier_category", "time_slot", "status_clean"}.issubset(df.columns):
        piv = (
            df.assign(_disr=df["status_clean"].isin(["delayed", "cancelled"]).astype(float) * 100)
            .pivot_table(index="carrier_category", columns="time_slot", values="_disr", aggfunc="mean")
            .round(2)
        )
        out["disruption_pct_carrier_x_timeslot"] = {
            str(idx): {str(col): (None if pd.isna(val) else float(val)) for col, val in row.items()}
            for idx, row in piv.iterrows()
        }
    if {"is_heavy_rain", "is_peak_hour", "status_clean"}.issubset(df.columns):
        grp = (
            df.assign(_disr=df["status_clean"].isin(["delayed", "cancelled"]).astype(float) * 100)
            .groupby(["is_peak_hour", "is_heavy_rain"])["_disr"]
            .agg(["count", "mean"])
            .round(2)
            .reset_index()
        )
        out["peak_x_heavy_rain"] = grp.to_dict("records")
    return out
