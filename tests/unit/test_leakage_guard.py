# -*- coding: utf-8 -*-
"""[UNIT / LEAKAGE] Verify runtime LeakageGuard blocks forbidden columns, future horizons & split overlap."""
import pandas as pd
import pytest

from flight_delay.common.exceptions import LeakageError
from flight_delay.data_mining.leakage_guard import (
    assert_causal_horizon,
    assert_chronological_splits,
    assert_no_forbidden_features,
    assert_target_separation,
)


def test_leakage_guard_blocks_forbidden_and_target_columns():
    assert_no_forbidden_features(["airline_code_VN", "airport_load", "weather_hazard_score"])
    with pytest.raises(LeakageError):
        assert_no_forbidden_features(["airline_code_VN", "delay_min"])
    with pytest.raises(LeakageError):
        assert_no_forbidden_features(["airline_code_VN", "actual_time"])


def test_leakage_guard_checks_horizon_and_chronology():
    valid_gold = pd.DataFrame([
        {"prediction_time": "2026-10-10 05:00:00", "scheduled_dt": "2026-10-10 08:00:00"},
    ])
    assert_causal_horizon(valid_gold, min_horizon_hours=3)

    bad_gold = pd.DataFrame([
        {"prediction_time": "2026-10-10 07:30:00", "scheduled_dt": "2026-10-10 08:00:00"},
    ])
    with pytest.raises(LeakageError):
        assert_causal_horizon(bad_gold, min_horizon_hours=3)

    tr = pd.DataFrame([{"flight_date": "2026-10-10"}])
    va = pd.DataFrame([{"flight_date": "2026-10-20"}])
    te = pd.DataFrame([{"flight_date": "2026-10-25"}])
    assert_chronological_splits(tr, va, te)

    bad_va = pd.DataFrame([{"flight_date": "2026-10-08"}])
    with pytest.raises(LeakageError):
        assert_chronological_splits(tr, bad_va, te)


def test_leakage_guard_enforces_exact_target_separation():
    feats = ["f1", "f2"]
    assert_target_separation(["f1", "f2", "target", "target_binary", "delay_min"], feats, split_name="test.csv")
    with pytest.raises(LeakageError):
        assert_target_separation(["f1", "f2", "target", "actual_time"], feats, split_name="test.csv")
