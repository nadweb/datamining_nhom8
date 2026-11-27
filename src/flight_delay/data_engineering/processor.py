# -*- coding: utf-8 -*-
"""[STEP 2] Data Preparation Pipeline: Bronze 12-Field Snapshots -> Silver Cleaned Dataset."""
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from flight_delay.common.constants import (
    CLEANED_FIELDS,
    DELAY_THRESHOLD_MIN,
    FILE_CLEANED,
    FILE_EXCLUDED,
    FILE_SILVER,
    REPORTS_DIR,
    VALID_OUTCOME_LABELS,
    VN_AIRLINES,
)
from flight_delay.common.utils import write_json
from flight_delay.data_engineering.contracts import validate_silver_contract
from flight_delay.data_engineering.deduplication import deduplicate_snapshots
from flight_delay.data_engineering.integration import (
    airline_benchmark_table,
    attach_weather,
    source_agreement,
)
from flight_delay.data_engineering.quality import build_silver_quality_report
from flight_delay.data_engineering.transformation import (
    add_leg_key,
    add_time_features,
    apply_labels,
    calculate_delay,
    compute_airport_load,
    normalize_flight_no,
    sched_datetime,
)
from flight_delay.data_engineering.validation import validate_flights_data
from flight_delay.ingestion.collector import collect_bronze_and_standardize

log = logging.getLogger(__name__)


def run_preparation_pipeline(
    delay_threshold: int = DELAY_THRESHOLD_MIN,
    prefer_side: str = "DEPARTURES",
    unit: str = "leg",
    min_age_hours: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Execute Step 2: Bronze -> Clean -> Deduplicate -> Weather Integrate -> Silver Contracts."""
    raw_std = collect_bronze_and_standardize()
    if raw_std.empty:
        raise RuntimeError("No Bronze snapshots available in data/raw/")

    df = raw_std.copy()
    for c in ["airport", "direction", "flight_date", "flight_no", "scheduled_time", "airline_code"]:
        df[c] = df[c].astype(str).str.strip().str.upper()
    df["status"] = df["status"].astype(str).str.strip()

    done = df["status"].str.contains(r"arrive|depart|land|hacan|catcan", case=False, regex=True, na=False)
    empty_act = df["actual_time"].astype(str).str.strip().isin(["", "nan", "None"])
    has_est = ~df["estimated_time"].astype(str).str.strip().isin(["", "nan", "None"])
    df.loc[done & empty_act & has_est, "actual_time"] = df.loc[done & empty_act & has_est, "estimated_time"]

    df["flight_no"] = normalize_flight_no(df["flight_no"])
    df["sched_dt"] = sched_datetime(df)
    df["hour_floor"] = df["sched_dt"].dt.floor("h")
    load = compute_airport_load(df)

    df = df[df["airline_code"].isin(VN_AIRLINES)].copy()
    df["delay_min"] = df.apply(
        lambda r: calculate_delay(r["scheduled_time"], r["estimated_time"], r["actual_time"], r["status"]), axis=1
    )
    df = add_leg_key(df)

    if min_age_hours:
        cutoff = pd.to_datetime(df["scraped_at"], errors="coerce").max() - pd.Timedelta(hours=min_age_hours)
        df = df[df["sched_dt"].isna() | (df["sched_dt"] <= cutoff)].copy()

    side, leg, per_src, dedup_stats = deduplicate_snapshots(df, prefer_side=prefer_side, unit=unit)
    leg = add_time_features(leg, load)
    leg = attach_weather(leg)
    leg = apply_labels(leg, thr=delay_threshold)

    valid_mask = leg["status_clean"].isin(VALID_OUTCOME_LABELS)
    valid_rich = leg[valid_mask].copy()
    excluded = leg[~valid_mask].copy()

    silver_24 = valid_rich.reindex(columns=CLEANED_FIELDS).replace("", np.nan)
    ok_contract, errs = validate_silver_contract(silver_24)
    if not ok_contract:
        raise RuntimeError(f"Silver contract failed: {errs}")

    # Persist Silver datasets & quality reports
    FILE_SILVER.parent.mkdir(parents=True, exist_ok=True)
    valid_rich.to_csv(FILE_SILVER, index=False, encoding="utf-8-sig")
    excluded.to_csv(FILE_EXCLUDED, index=False, encoding="utf-8-sig")
    silver_24.to_csv(FILE_CLEANED, index=False, encoding="utf-8-sig")

    passed_8, checks_8, metrics_8 = validate_flights_data(silver_24)
    quality_report = build_silver_quality_report(raw_std, leg, valid_rich, excluded, dedup_stats)
    sa_df = source_agreement(per_src, delay_threshold)
    bench_df = airline_benchmark_table(valid_rich)
    quality_report["source_agreement"] = sa_df.to_dict("records") if not sa_df.empty else []
    quality_report["airline_caav_benchmark"] = bench_df.to_dict("records")
    quality_report["data_contract_8_checks"] = {"passed": passed_8, "checks": checks_8, "metrics": metrics_8}

    write_json(REPORTS_DIR / "data_quality" / "silver_quality_report.json", quality_report)
    log.info(
        "Silver preparation complete: valid=%d (cancelled=%d, delayed=%d, on_time=%d), excluded=%d, contract=%s",
        len(silver_24),
        metrics_8["cancelled_count"],
        metrics_8["delayed_count"],
        metrics_8["on_time_count"],
        len(excluded),
        passed_8,
    )
    return silver_24, excluded, quality_report
