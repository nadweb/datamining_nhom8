# -*- coding: utf-8 -*-
"""[STEP 7] Production Data Quality Monitor."""
import pandas as pd


def run_data_quality_check(df: pd.DataFrame) -> dict:
    key_cols = ["leg_key", "airline_code", "airport", "scheduled_time", "precipitation", "wind_gusts_10m", "visibility"]
    missing_rates = {
        c: round(float(df[c].isna().mean() * 100.0), 2) if c in df.columns else 100.0
        for c in key_cols
    }
    dup_rate = round(float(df["leg_key"].duplicated().mean() * 100.0), 3) if "leg_key" in df.columns else 0.0
    neg_wx = int(
        (
            (pd.to_numeric(df.get("precipitation", 0), errors="coerce") < 0)
            | (pd.to_numeric(df.get("visibility", 0), errors="coerce") < 0)
        ).sum()
    )
    valid_labels = df["status_clean"].isin(["on_time", "delayed", "cancelled"]).mean() * 100.0 if "status_clean" in df.columns else 100.0
    status = "OK" if (max(missing_rates.values()) < 5.0 and dup_rate < 1.0 and neg_wx == 0) else "WARNING"
    return {
        "status": status,
        "rows_checked": int(len(df)),
        "missing_pct": missing_rates,
        "duplicate_leg_pct": dup_rate,
        "invalid_weather_rows": neg_wx,
        "valid_outcome_label_pct": round(float(valid_labels), 2),
    }
