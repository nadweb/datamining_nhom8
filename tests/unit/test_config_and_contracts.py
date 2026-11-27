# -*- coding: utf-8 -*-
"""[UNIT] Config, Feature Registry & Pydantic Data Contract Tests."""
import pytest
from pydantic import ValidationError

from flight_delay.common.constants import (
    AIRPORT_COORD,
    CLEANED_FIELDS,
    FIELDS,
    FORBIDDEN_POST_FLIGHT_COLUMNS,
    LABEL_MAP_3CLASS,
    LABEL_MAP_BINARY,
    PRIMARY_HORIZON_HOURS,
)
from flight_delay.data_engineering.contracts import GoldDatasetContract, SilverFlightContract
from flight_delay.data_mining.feature_engineering.registry import FEATURE_REGISTRY


def test_schema_dimensions_and_labels():
    assert len(FIELDS) == 12
    assert len(CLEANED_FIELDS) == 24
    assert LABEL_MAP_3CLASS == {"on_time": 0, "delayed": 1, "cancelled": 2}
    assert LABEL_MAP_BINARY == {"on_time": 0, "disrupted": 1}
    assert PRIMARY_HORIZON_HOURS == 3
    assert set(["SGN", "HAN", "DAD", "CXR", "PQC"]).issubset(AIRPORT_COORD.keys())


def test_feature_registry_has_zero_post_flight_columns():
    cols = set(FEATURE_REGISTRY.model_input_columns)
    assert cols.isdisjoint(FORBIDDEN_POST_FLIGHT_COLUMNS)
    assert len(FEATURE_REGISTRY.model_input_columns) >= 25


def test_silver_contract_rejects_cancelled_with_delay_or_diverted():
    base = dict(
        leg_key="VN210|2026-10-15|SGN>HAN",
        flight_date="2026-10-15",
        flight_no="VN210",
        airline_code="VN",
        origin="SGN",
        destination="HAN",
        route_type="domestic",
        airport="SGN",
        direction="DEPARTURES",
        scheduled_time="10:00",
        is_holiday=False,
        status="Cancelled",
        status_detail="cancelled",
        status_clean="cancelled",
        source="sgn",
    )
    ok_obj = SilverFlightContract(**base, delay_min=None)
    assert ok_obj.status_clean == "cancelled"

    with pytest.raises(ValidationError):
        SilverFlightContract(**base, delay_min=0.0)

    diverted_row = {**base, "status_clean": "diverted"}
    with pytest.raises(ValidationError):
        SilverFlightContract(**diverted_row)


def test_gold_contract_rejects_future_prediction_time():
    with pytest.raises(ValidationError):
        GoldDatasetContract(
            leg_key="VJ120|2026-11-01|SGN>HAN",
            prediction_time="2026-11-01 15:00:00",
            scheduled_dt="2026-11-01 14:00:00",
            horizon_hours=3,
            feature_columns=["airline_code", "airport"],
        )
