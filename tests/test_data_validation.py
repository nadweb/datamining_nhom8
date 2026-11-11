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
        if self.sample_path.exists():
            self.sample_df = pd.read_csv(self.sample_path)
        else:
            data = []
            for i in range(100):
                is_can = (i == 99)
                is_del = (70 <= i < 99)
                data.append({
                    "leg_key": f"VN{100+i}|2026-10-28|SGN>HAN",
                    "flight_date": "2026-10-28",
                    "flight_no": f"VN{100+i}",
                    "airline_code": "VN",
                    "origin": "SGN",
                    "destination": "HAN",
                    "route_type": "domestic",
                    "airport": "SGN",
                    "direction": "DEPARTURES",
                    "scheduled_time": "08:00",
                    "actual_time": "08:00" if not is_del and not is_can else ("08:30" if is_del else None),
                    "delay_min": np.nan if is_can else (20.0 if is_del else 0.0),
                    "time_slot": "06h-12h",
                    "day_type": "weekday",
                    "is_holiday": False,
                    "airport_load": 20,
                    "precipitation": 0.0,
                    "wind_gusts_10m": 12.0,
                    "visibility": 10000.0,
                    "weather_code": 0,
                    "status": "Cancelled" if is_can else ("Delayed 20m" if is_del else "Departed On-Time"),
                    "status_detail": "cancelled" if is_can else ("departed_late" if is_del else "departed_on_time"),
                    "status_clean": "cancelled" if is_can else ("delayed" if is_del else "on_time"),
                    "source": "guide_sgn" if i % 2 == 0 else "guide_han"
                })
            self.sample_df = pd.DataFrame(data)

    def test_sample_dataset_passes_all_checks(self):
        """The sample dataset must pass 100% of data quality checks."""
        passed, checks, metrics = validate_flights_data(self.sample_df)
        for c in checks:
            self.assertTrue(c["passed"], f"Failed check: {c['check']} - {c['detail']}")
        self.assertTrue(passed)
        self.assertEqual(metrics["total_cols"], 24)
        self.assertGreaterEqual(metrics["total_rows"], 100)

    def test_detects_missing_column(self):
        """Validation must fail if a required schema column is missing."""
        df = self.sample_df.copy().drop(columns=["airline_code"])
        passed, checks, _ = validate_flights_data(df)
        self.assertFalse(passed)
        cols_check = next(c for c in checks if "24 columns" in c["check"])
        self.assertFalse(cols_check["passed"])

    def test_detects_duplicate_leg_keys(self):
        """Validation must fail if leg_key is duplicated."""
        df = self.sample_df.copy()
        # Duplicate first row's leg_key onto second row
        df.loc[1, "leg_key"] = df.loc[0, "leg_key"]
        passed, checks, _ = validate_flights_data(df)
        self.assertFalse(passed)
        key_check = next(c for c in checks if "Unique leg_key" in c["check"])
        self.assertFalse(key_check["passed"])

    def test_detects_invalid_status_clean_label(self):
        """Validation must fail if an unknown label exists in status_clean."""
        df = self.sample_df.copy()
        df.loc[0, "status_clean"] = "diverted_unknown"
        passed, checks, _ = validate_flights_data(df)
        self.assertFalse(passed)
        label_check = next(c for c in checks if "Labels strictly" in c["check"])
        self.assertFalse(label_check["passed"])

    def test_detects_cancelled_flight_with_delay_min(self):
        """Validation must fail if a cancelled flight has non-null delay_min."""
        df = self.sample_df.copy()
        # Force a cancelled flight to have delay_min = 25.0
        df.loc[0, "status_clean"] = "cancelled"
        df.loc[0, "status"] = "Cancelled"
        df.loc[0, "delay_min"] = 25.0
        passed, checks, _ = validate_flights_data(df)
        self.assertFalse(passed)
        cancel_check = next(c for c in checks if "Cancelled flights have no delay_min" in c["check"])
        self.assertFalse(cancel_check["passed"])

    def test_missing_required_columns_does_not_raise_keyerror(self):
        """Malformed input should be reported as a failed contract, not crash."""
        df = self.sample_df.drop(columns=["leg_key", "status", "status_clean", "delay_min"])
        passed, checks, _ = validate_flights_data(df)
        self.assertFalse(passed)
        self.assertEqual(len(checks), 8)
        self.assertTrue(all(not check["passed"] for check in checks if "leg_key" in check["check"] or "status_clean" in check["check"]))


if __name__ == "__main__":
    unittest.main()

