# -*- coding: utf-8 -*-
"""[STEP 2] Independent Data Contracts for Bronze (Raw), Silver (Cleaned), and Gold (Modeling)."""
from typing import List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from flight_delay.common.constants import (
    CLEANED_FIELDS,
    EARLY_LIMIT,
    FIELDS,
    FORBIDDEN_POST_FLIGHT_COLUMNS,
    VALID_OUTCOME_LABELS,
)


class RawSnapshotContract(BaseModel):
    """Contract 1: Bronze standardized 12-field snapshot."""
    source: str
    airport: str
    direction: str
    scraped_at: str
    flight_date: str
    airline_code: str = ""
    flight_no: str
    other_airport: str = ""
    scheduled_time: str = ""
    estimated_time: str = ""
    actual_time: str = ""
    status: str = ""


class SilverFlightContract(BaseModel):
    """Contract 2: Silver canonical 24-column clean flight leg."""
    leg_key: str = Field(min_length=5)
    flight_date: str
    flight_no: str
    airline_code: str
    origin: str
    destination: str
    route_type: str
    airport: str
    direction: str
    scheduled_time: str
    actual_time: Optional[str] = None
    delay_min: Optional[float] = None
    time_slot: Optional[str] = None
    day_type: Optional[str] = None
    is_holiday: bool
    airport_load: Optional[float] = None
    precipitation: Optional[float] = None
    wind_gusts_10m: Optional[float] = None
    visibility: Optional[float] = None
    weather_code: Optional[float] = None
    status: str
    status_detail: str
    status_clean: str
    source: str

    @model_validator(mode="after")
    def _enforce_domain_invariants(self):
        if self.status_clean not in VALID_OUTCOME_LABELS:
            raise ValueError(f"Invalid status_clean={self.status_clean!r}")
        if self.status_clean == "cancelled" and self.delay_min is not None and pd.notna(self.delay_min):
            raise ValueError("Cancelled flights must have NaN/None delay_min.")
        if self.delay_min is not None and pd.notna(self.delay_min) and self.delay_min < EARLY_LIMIT:
            raise ValueError(f"Implausible early delay_min={self.delay_min} < {EARLY_LIMIT}")
        return self


class GoldDatasetContract(BaseModel):
    """Contract 3: Gold analytical & prediction-horizon contract."""
    leg_key: str
    prediction_time: str
    scheduled_dt: str
    horizon_hours: int = Field(ge=1)
    feature_columns: List[str]

    @model_validator(mode="after")
    def _enforce_horizon_and_no_leakage(self):
        pred_ts = pd.Timestamp(self.prediction_time)
        sched_ts = pd.Timestamp(self.scheduled_dt)
        if pred_ts > sched_ts:
            raise ValueError(f"Leakage: prediction_time {pred_ts} > scheduled_dt {sched_ts}")
        bad = set(self.feature_columns) & set(FORBIDDEN_POST_FLIGHT_COLUMNS)
        if bad:
            raise ValueError(f"Forbidden post-flight columns in features: {sorted(bad)}")
        return self


def validate_raw_contract(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    errors = []
    missing = [c for c in FIELDS if c not in df.columns]
    if missing:
        errors.append(f"Missing bronze standardized columns: {missing}")
    if df.empty:
        errors.append("Bronze dataset is empty.")
    return len(errors) == 0, errors


def validate_silver_contract(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    errors = []
    if list(df.columns) != CLEANED_FIELDS and not set(CLEANED_FIELDS).issubset(df.columns):
        missing = sorted(set(CLEANED_FIELDS) - set(df.columns))
        errors.append(f"Silver schema mismatch; missing={missing}")
    if df.empty:
        errors.append("Silver dataset is empty.")
        return False, errors
    if not df["leg_key"].is_unique:
        errors.append("Duplicate leg_key detected in Silver dataset.")
    bad_labels = set(df["status_clean"].dropna().astype(str)) - set(VALID_OUTCOME_LABELS)
    if bad_labels:
        errors.append(f"Unexpected status_clean labels in Silver: {sorted(bad_labels)}")
    if (df.loc[df["status_clean"] == "cancelled", "delay_min"].notna()).any():
        errors.append("Cancelled rows in Silver contain non-null delay_min.")
    return len(errors) == 0, errors


def validate_gold_contract(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[bool, List[str]]:
    errors = []
    bad = sorted(set(feature_cols) & set(FORBIDDEN_POST_FLIGHT_COLUMNS))
    if bad:
        errors.append(f"Forbidden post-flight columns in Gold feature_cols: {bad}")
    if "prediction_time" not in df.columns or "scheduled_dt" not in df.columns:
        errors.append("Gold dataset missing prediction_time or scheduled_dt timestamp.")
    else:
        pred = pd.to_datetime(df["prediction_time"], errors="coerce")
        sched = pd.to_datetime(df["scheduled_dt"], errors="coerce")
        if pred.isna().any() or sched.isna().any():
            errors.append("Invalid timestamps in prediction_time / scheduled_dt.")
        elif (pred > sched).any():
            errors.append("Temporal leakage: prediction_time > scheduled_dt detected.")
    return len(errors) == 0, errors
