# -*- coding: utf-8 -*-
"""[STEP 2] Data Quality Profiling, Early-Departure & Excluded-Outcome Policy Report."""
from typing import Dict

import numpy as np
import pandas as pd

from flight_delay.common.constants import EARLY_LIMIT, FIELDS

FORMAT_RULES = {
    "flight_date": r"^\d{4}-\d{2}-\d{2}$",
    "scheduled_time": r"^([01]\d|2[0-3]):[0-5]\d$",
    "flight_no": r"^[A-Z0-9]{2}\s?\d{1,4}[A-Z]?$",
    "direction": r"^(DEPARTURES|ARRIVALS)$",
    "airport": r"^[A-Z]{3}$",
}


def compute_raw_format_report(df: pd.DataFrame) -> list:
    rows = []
    n = len(df)
    for c in FIELDS:
        if c not in df.columns:
            continue
        s = df[c].astype(str).str.strip()
        empty = s.isin(["", "nan", "None", "NaN"])
        item = {"column": c, "missing": int(empty.sum()), "missing_pct": round(float(empty.mean() * 100), 2)}
        if c in FORMAT_RULES:
            bad = ~empty & ~s.str.upper().str.match(FORMAT_RULES[c])
            item["invalid_format"] = int(bad.sum())
            item["invalid_format_pct"] = round(float(bad.sum() / n * 100), 2) if n else 0.0
        rows.append(item)
    return rows


def build_silver_quality_report(
    raw_std: pd.DataFrame,
    leg: pd.DataFrame,
    valid: pd.DataFrame,
    excluded: pd.DataFrame,
    stats: Dict,
) -> dict:
    """Summarize Bronze->Silver quality checks, missingness, early/extreme delays, and excluded policies."""
    delays = pd.to_numeric(valid["delay_min"], errors="coerce").dropna()
    early_flights = delays[delays < 0]
    extreme_delays = delays[delays > 180]
    missing_pct = valid.replace("", np.nan).isna().mean().mul(100).round(2).to_dict()

    return {
        "bronze_standardized_rows": int(len(raw_std)),
        "raw_format_audit": compute_raw_format_report(raw_std),
        "deduplication_stats": stats,
        "silver_valid_rows": int(len(valid)),
        "excluded_outcome_rows": int(len(excluded)),
        "excluded_status_breakdown": excluded["status_clean"].value_counts().to_dict() if not excluded.empty else {},
        "outcome_distribution_3class": valid["status_clean"].value_counts().to_dict(),
        "outcome_detail_distribution": valid["status_detail"].value_counts().to_dict(),
        "delay_duration_policy": {
            "early_limit_min": EARLY_LIMIT,
            "valid_non_null_delays": int(len(delays)),
            "cancelled_with_null_delay": int((valid["status_clean"] == "cancelled").sum()),
            "early_departure_count": int(len(early_flights)),
            "early_departure_pct": round(float(len(early_flights) / max(len(delays), 1) * 100), 2),
            "early_min_minutes": float(early_flights.min()) if len(early_flights) else None,
            "extreme_delay_gt_180m_count": int(len(extreme_delays)),
            "max_delay_minutes": float(delays.max()) if len(delays) else None,
        },
        "column_missing_pct": missing_pct,
        "integration_coverage": {
            "leg_matched_pct": round(float(leg["leg_matched"].mean() * 100), 2) if "leg_matched" in leg else 100.0,
            "both_sides_seen_pct": round(float((leg["n_sides_seen"] > 1).mean() * 100), 2) if "n_sides_seen" in leg else 0.0,
            "weather_matched_pct": round(float(leg["precipitation"].notna().mean() * 100), 2),
            "airport_load_matched_pct": round(float(leg["airport_load"].notna().mean() * 100), 2),
        },
    }
