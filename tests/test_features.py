# -*- coding: utf-8 -*-
"""Automated tests for feature engineering logic."""
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from features.features import (
    build_engineered_features, compute_cyclical_time,
    compute_weather_hazard, extract_single_flight_features
)


class TestFeatureEngineering(unittest.TestCase):
    """Test suite for feature transformations and zero-leakage requirements."""

    def setUp(self):
        self.sample_row = pd.DataFrame([{
            "leg_key": "VN123|2026-11-06|SGN>HAN",
            "flight_date": "2026-11-06",
            "flight_no": "VN123",
            "airline_code": "VN",
            "origin": "SGN",
            "destination": "HAN",
            "route_type": "domestic",
            "airport": "SGN",
            "direction": "DEPARTURES",
            "scheduled_time": "18:30",
            "actual_time": "18:45",
            "delay_min": 15.0,
            "time_slot": "18h-24h",
            "day_type": "weekday",
            "is_holiday": False,
            "airport_load": 32,
            "precipitation": 4.5,
            "wind_gusts_10m": 42.0,
            "visibility": 3500.0,
            "weather_code": 95,
            "status": "Departed 15 minutes late",
            "status_detail": "departed_late",
            "status_clean": "delayed",
            "source": "guide_sgn",
        }])

    def test_cyclical_time_bounds(self):
        """Sine and cosine hour transformations must strictly be within [-1, 1]."""
        df = compute_cyclical_time(self.sample_row)
        self.assertTrue(-1.0 <= df["hour_sin"].iloc[0] <= 1.0)
        self.assertTrue(-1.0 <= df["hour_cos"].iloc[0] <= 1.0)
        self.assertTrue(-1.0 <= df["dow_sin"].iloc[0] <= 1.0)
        self.assertTrue(-1.0 <= df["dow_cos"].iloc[0] <= 1.0)

    def test_weather_hazard_scoring(self):
        """Adverse weather conditions must trigger hazard flags and elevated score."""
        df = compute_weather_hazard(self.sample_row)
        self.assertEqual(df["is_rain"].iloc[0], 1)
        self.assertEqual(df["is_heavy_rain"].iloc[0], 1)
        self.assertEqual(df["is_high_wind"].iloc[0], 1)
        self.assertEqual(df["is_storm"].iloc[0], 1)
        self.assertGreater(df["weather_hazard_score"].iloc[0], 0.35)

    def test_single_flight_feature_extraction(self):
        """Single flight feature extractor produces all required model features."""
        df_feat = extract_single_flight_features(
            airline_code="VJ", airport="SGN", direction="DEPARTURES",
            route_type="domestic", scheduled_time="07:15", flight_date="2026-11-06"
        )
        self.assertIn("hour_sin", df_feat.columns)
        self.assertIn("weather_hazard_score", df_feat.columns)
        self.assertIn("is_peak_hour", df_feat.columns)
        self.assertEqual(df_feat["is_peak_hour"].iloc[0], 1)  # 07:15 is morning peak


if __name__ == "__main__":
    unittest.main()

