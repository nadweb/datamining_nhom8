# -*- coding: utf-8 -*-
"""[LEAKAGE GUARD] Verify Zero Future Information Leakage Across Features, Matrices & API."""
import pandas as pd
import pytest

from flight_delay.common.constants import FORBIDDEN_POST_FLIGHT_COLUMNS, PROCESSED_DIR
from flight_delay.common.exceptions import LeakageError
from flight_delay.common.utils import read_json
from flight_delay.data_mining.feature_engineering.historical import CausalHistoryStore
from flight_delay.inference.feature_validation import validate_flight_input


def test_modeled_matrix_contains_zero_forbidden_columns():
    meta = read_json(PROCESSED_DIR / "preprocessing_metadata.json")
    feature_names = set(meta["feature_names"])
    assert feature_names.isdisjoint(FORBIDDEN_POST_FLIGHT_COLUMNS)
    for split_file in ("train.csv", "val.csv", "test.csv"):
        cols = set(pd.read_csv(PROCESSED_DIR / split_file, nrows=1).columns)
        # Only explicit supervised targets are stored outside feature_names
        assert (cols - feature_names) == {"target", "target_binary", "delay_min"}


def test_causal_history_strictly_ignores_future_outcomes():
    obs = pd.DataFrame([
        {
            "airline_code": "VN",
            "airport": "SGN",
            "flight_date": "2026-10-10",
            "scheduled_time": "08:00",
            "delay_min": 0.0,
            "status_clean": "on_time",
            "outcome_known_at": "2026-10-10 08:00:00",
        },
        {
            "airline_code": "VN",
            "airport": "SGN",
            "flight_date": "2026-10-11",
            "scheduled_time": "08:00",
            "delay_min": 95.0,
            "status_clean": "delayed",
            "outcome_known_at": "2026-10-11 09:35:00",
        },
    ])
    store = CausalHistoryStore.from_observations(obs, prior_smooth=1.0, fallback_prior=0.25)

    # Prediction at 2026-10-10 12:00 MUST NOT see 2026-10-11 delayed flight
    r_al_early, r_ap_early = store.rates_for_row("VN", "SGN", "2026-10-10 12:00:00")
    # Prediction at 2026-10-12 12:00 MUST see both
    r_al_later, _ = store.rates_for_row("VN", "SGN", "2026-10-12 12:00:00")
    assert r_al_early < r_al_later


def test_inference_validation_rejects_post_flight_leakage_keys():
    with pytest.raises(LeakageError):
        validate_flight_input({
            "airline_code": "VJ",
            "airport": "SGN",
            "scheduled_time": "19:30",
            "delay_min": 45.0,
        })
