# -*- coding: utf-8 -*-
"""[DATA QUALITY] Validate Generated Bronze, Silver, Excluded Outcomes & Gold Artifacts."""
import pandas as pd

from flight_delay.common.constants import (
    FILE_EXCLUDED,
    FILE_GOLD,
    FILE_SILVER,
    LABEL_MAP_3CLASS,
    VALID_OUTCOME_LABELS,
)
from flight_delay.data_engineering.contracts import validate_gold_contract, validate_silver_contract
from flight_delay.data_mining.feature_engineering.registry import FEATURE_REGISTRY


def test_silver_dataset_contracts_and_excluded_segregation():
    assert FILE_SILVER.exists(), "Run `python main.py prepare` before running data quality tests."
    df = pd.read_csv(FILE_SILVER, low_memory=False)
    assert len(df) > 10000
    assert df["leg_key"].is_unique
    assert set(df["status_clean"].unique()).issubset(VALID_OUTCOME_LABELS)

    cancelled = df[df["status_clean"] == "cancelled"]
    assert cancelled["delay_min"].isna().all(), "Cancelled flights must never assign numeric delay_min!"

    ok, errs = validate_silver_contract(df)
    assert ok, f"Silver contract violations: {errs}"

    assert FILE_EXCLUDED.exists()
    excl = pd.read_csv(FILE_EXCLUDED)
    assert not excl.empty
    assert set(excl["status_clean"].unique()).issubset({"diverted", "scheduled", "unknown"})


def test_gold_dataset_contracts_and_split_columns():
    assert FILE_GOLD.exists(), "Run `python main.py dataset build` before running data quality tests."
    gold = pd.read_csv(FILE_GOLD, low_memory=False)
    ok, errs = validate_gold_contract(gold, FEATURE_REGISTRY.model_input_columns)
    assert ok, f"Gold contract violations: {errs}"
    assert set(gold["split_role"].unique()) == {"train", "val", "test"}
    assert set(gold["status_clean"].unique()).issubset(LABEL_MAP_3CLASS.keys())
