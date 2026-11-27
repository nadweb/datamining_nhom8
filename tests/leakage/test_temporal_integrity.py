# -*- coding: utf-8 -*-
"""[TEMPORAL INTEGRITY] Chronological Split Order, Boundary Isolation & Horizon Bounds."""
import pandas as pd
import pytest

from flight_delay.common.constants import FILE_GOLD, PROCESSED_DIR
from flight_delay.common.exceptions import InvalidPredictionTimeError
from flight_delay.common.utils import read_json
from flight_delay.inference.feature_validation import validate_flight_input


def test_chronological_split_ordering_and_causal_history_cutoff():
    gold = pd.read_csv(FILE_GOLD, low_memory=False)
    tr = gold[gold["split_role"] == "train"]
    va = gold[gold["split_role"] == "val"]
    te = gold[gold["split_role"] == "test"]
    assert not tr.empty and not va.empty and not te.empty

    tr_ts = tr["flight_date"].astype(str) + " " + tr["scheduled_time"].astype(str)
    va_ts = va["flight_date"].astype(str) + " " + va["scheduled_time"].astype(str)
    te_ts = te["flight_date"].astype(str) + " " + te["scheduled_time"].astype(str)

    # Strict monotonic boundaries: no timestamp overlap across Train/Val/Test
    assert tr_ts.max() < va_ts.min()
    assert va_ts.max() < te_ts.min()

    # Every Gold row respects prediction_time <= scheduled_dt
    pred_dt = pd.to_datetime(gold["prediction_time"])
    sched_dt = pd.to_datetime(gold["scheduled_dt"])
    assert (pred_dt <= sched_dt).all()

    # Causal store must contain zero test observations
    store = read_json(PROCESSED_DIR / "causal_history_store.json")
    max_obs = pd.Timestamp(store["max_observed_ts"])
    assert max_obs.date() <= pd.Timestamp(te["flight_date"].min()).date()


def test_prediction_time_after_departure_fails_closed():
    with pytest.raises(InvalidPredictionTimeError):
        validate_flight_input(
            {"airline_code": "VN", "airport": "HAN", "scheduled_time": "10:00", "flight_date": "2026-11-06"},
            prediction_time="2026-11-06 10:05:00",
        )
