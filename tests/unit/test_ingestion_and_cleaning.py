# -*- coding: utf-8 -*-
"""[STAGE 01 & 02 TESTS] Bronze Ingestion Standardization & Silver SHA-256 Deduplication."""
import pandas as pd

from flight_delay.common.constants import FIELDS, FILE_STANDARDIZED
from flight_delay.data_engineering.contracts import validate_raw_contract
from flight_delay.data_engineering.deduplication import deduplicate_snapshots
from flight_delay.data_engineering.transformation import (
    add_leg_key,
    apply_labels,
    calculate_delay,
    normalize_flight_no,
    sched_datetime,
)


def test_bronze_standardized_preserves_12_fields_and_raw_status():
    assert FILE_STANDARDIZED.exists(), "Missing data/bronze/standardized_snapshots.csv"
    df = pd.read_csv(FILE_STANDARDIZED, nrows=200, dtype=str, keep_default_na=False)
    assert list(df.columns) == FIELDS
    ok, errs = validate_raw_contract(df)
    assert ok, f"Bronze contract error: {errs}"
    # Confirm Bronze retains descriptive source statuses prior to Silver collapsing
    raw_statuses = set(df["status"].unique())
    assert len(raw_statuses) >= 2


def test_three_tier_deduplication_prefers_final_completed_leg():
    sample = pd.DataFrame([
        {
            "source": "sgn",
            "airport": "SGN",
            "direction": "DEPARTURES",
            "scraped_at": "2026-10-20 06:00:00",
            "flight_date": "2026-10-20",
            "airline_code": "VN",
            "flight_no": "VN 0210",
            "other_airport": "HAN",
            "scheduled_time": "10:00",
            "estimated_time": "10:05",
            "actual_time": "",
            "status": "Scheduled",
        },
        {
            "source": "sgn",
            "airport": "SGN",
            "direction": "DEPARTURES",
            "scraped_at": "2026-10-20 10:25:00",
            "flight_date": "2026-10-20",
            "airline_code": "VN",
            "flight_no": "VN210",
            "other_airport": "HAN",
            "scheduled_time": "10:00",
            "estimated_time": "10:20",
            "actual_time": "10:22",
            "status": "Departed 22 minutes late",
        },
    ])
    sample["flight_no"] = normalize_flight_no(sample["flight_no"])
    sample["sched_dt"] = sched_datetime(sample)
    sample["delay_min"] = [calculate_delay(r["scheduled_time"], r["estimated_time"], r["actual_time"], r["status"]) for _, r in sample.iterrows()]
    sample = add_leg_key(sample)

    _side, leg, _per_src, stats = deduplicate_snapshots(sample, prefer_side="DEPARTURES", unit="leg")
    leg = apply_labels(leg)
    assert len(leg) == 1
    assert leg.iloc[0]["status_clean"] == "delayed"
    assert float(leg.iloc[0]["delay_min"]) == 22.0
    assert stats["after_leg_dedup"] == 1
    assert stats["unit"] == "leg"
    assert int(leg.iloc[0]["has_t3h_snapshot"]) == 1
