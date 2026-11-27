# -*- coding: utf-8 -*-
"""[STEP 3] Univariate Distributions for Outcome, Delay, Airline & Airport."""
import pandas as pd


def compute_univariate_summary(df: pd.DataFrame) -> dict:
    total = len(df)
    status_counts = df["status_clean"].value_counts().to_dict() if "status_clean" in df.columns else {}
    status_dist = {
        k: {"count": int(v), "pct": round(float(v / total * 100), 2) if total else 0.0}
        for k, v in status_counts.items()
    }
    delays = pd.to_numeric(df.get("delay_min", pd.Series(dtype=float)), errors="coerce").dropna()
    delay_stats = {
        "count": int(len(delays)),
        "mean": round(float(delays.mean()), 2) if len(delays) else 0.0,
        "std": round(float(delays.std()), 2) if len(delays) else 0.0,
        "median": round(float(delays.median()), 2) if len(delays) else 0.0,
        "q25": round(float(delays.quantile(0.25)), 2) if len(delays) else 0.0,
        "q75": round(float(delays.quantile(0.75)), 2) if len(delays) else 0.0,
        "p95": round(float(delays.quantile(0.95)), 2) if len(delays) else 0.0,
        "min": round(float(delays.min()), 2) if len(delays) else 0.0,
        "max": round(float(delays.max()), 2) if len(delays) else 0.0,
    }
    date_min = str(df["flight_date"].min()) if "flight_date" in df.columns else ""
    date_max = str(df["flight_date"].max()) if "flight_date" in df.columns else ""
    return {
        "total_flights": int(total),
        "date_range": {"min": date_min, "max": date_max},
        "status_distribution": status_dist,
        "delay_min_distribution": delay_stats,
    }
