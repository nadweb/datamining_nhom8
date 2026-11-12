# -*- coding: utf-8 -*-
"""Automated tests for data contract and data validation rules."""
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from config import BASE_DIR, CLEANED_FIELDS
from pipeline.check_data import validate_flights_data


class TestDataValidation(unittest.TestCase):
    """Test suite verifying validation functions on sample data and synthetic invalid data."""

    def setUp(self):
        self.sample_path = BASE_DIR / "data" / "flights_sample.csv"

    def test_sample_dataset_passes_all_checks(self):
        """The committed 300-row sample dataset must pass 100% of data quality checks."""
        self.assertTrue(self.sample_path.exists(), "Sample dataset flights_sample.csv must exist!")
        passed, checks, metrics = validate_flights_data(self.sample_path)
        for c in checks:
            self.assertTrue(c["passed"], f"Failed check: {c['check']} - {c['detail']}")
        self.assertTrue(passed)
        self.assertEqual(metrics["total_cols"], 24)
        self.assertGreaterEqual(metrics["total_rows"], 100)

    def test_detects_missing_column(self):
        """Validation must fail if a required schema column is missing."""
        df = pd.read_csv(self.sample_path).drop(columns=["airline_code"])
        passed, checks, _ = validate_flights_data(df)
        self.assertFalse(passed)
        cols_check = next(c for c in checks if "24 columns" in c["check"])
        self.assertFalse(cols_check["passed"])

    def test_detects_duplicate_leg_keys(self):
        """Validation must fail if leg_key is duplicated."""
        df = pd.read_csv(self.sample_path)
        # Duplicate first row's leg_key onto second row
        df.loc[1, "leg_key"] = df.loc[0, "leg_key"]
        passed, checks, _ = validate_flights_data(df)
        self.assertFalse(passed)
        key_check = next(c for c in checks if "Unique leg_key" in c["check"])
        self.assertFalse(key_check["passed"])

    def test_detects_invalid_status_clean_label(self):
        """Validation must fail if an unknown label exists in status_clean."""
        df = pd.read_csv(self.sample_path)
        df.loc[0, "status_clean"] = "diverted_unknown"
        passed, checks, _ = validate_flights_data(df)
        self.assertFalse(passed)
        label_check = next(c for c in checks if "Labels strictly" in c["check"])
        self.assertFalse(label_check["passed"])

    def test_detects_cancelled_flight_with_delay_min(self):
        """Validation must fail if a cancelled flight has non-null delay_min."""
        df = pd.read_csv(self.sample_path)
        # Force a cancelled flight to have delay_min = 25.0
        df.loc[0, "status_clean"] = "cancelled"
        df.loc[0, "status"] = "Cancelled"
        df.loc[0, "delay_min"] = 25.0
        passed, checks, _ = validate_flights_data(df)
        self.assertFalse(passed)
        cancel_check = next(c for c in checks if "Cancelled flights have no delay_min" in c["check"])
        self.assertFalse(cancel_check["passed"])


if __name__ == "__main__":
    unittest.main()

