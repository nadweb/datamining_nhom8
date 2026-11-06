# -*- coding: utf-8 -*-
"""Automated tests for modeling, train/test splitting, and predict_risk contract."""
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from modeling.encoding import LEAKAGE_COLUMNS, split_by_date
from modeling.predict import predict_risk


class TestModelingPipeline(unittest.TestCase):
    """Test suite ensuring data leakage prevention and valid inference contract."""

    def test_zero_leakage_in_training_features(self):
        """Preprocessed train and test datasets must NEVER contain leakage columns."""
        train_path = ROOT_DIR / "data" / "processed" / "train.csv"
        if train_path.exists():
            df = pd.read_csv(train_path, nrows=5)
            # The only label allowed is 'target' and delay_min as separate regression target
            feature_cols = [c for c in df.columns if c not in ["target", "delay_min"]]
            for leak in LEAKAGE_COLUMNS:
                self.assertNotIn(leak, feature_cols, f"Data leakage detected! {leak} is in feature columns!")

    def test_chronological_split_prevents_lookahead(self):
        """Train dates must precede or equal test dates in chronological split."""
        dummy = pd.DataFrame({
            "flight_date": ["2026-10-25", "2026-10-26", "2026-10-27", "2026-10-28", "2026-10-29"],
            "scheduled_time": ["08:00", "09:00", "10:00", "11:00", "12:00"]
        })
        train, test, cutoff = split_by_date(dummy, train_ratio=0.6)
        self.assertLessEqual(train["flight_date"].max(), test["flight_date"].min())
        self.assertEqual(len(train), 3)
        self.assertEqual(len(test), 2)

    def test_predict_risk_response_schema(self):
        """predict_risk() must strictly adhere to the contract schema defined in docs/schema.md."""
        res = predict_risk(
            airline_code="VN",
            airport="HAN",
            direction="DEPARTURES",
            route_type="domestic",
            scheduled_time="10:00",
            flight_date="2026-11-06",
            precipitation=0.0,
            wind_gusts_10m=15.0,
            visibility=10000.0,
            weather_code=0,
            airport_load=22
        )
        required_keys = [
            "prediction", "risk_score", "risk_level",
            "estimated_delay_min", "probabilities",
            "cluster_profile", "matched_rules", "flight_summary"
        ]
        for k in required_keys:
            self.assertIn(k, res, f"predict_risk response missing key '{k}'!")

        self.assertIn(res["prediction"], ["on_time", "delayed", "cancelled"])
        self.assertIn(res["risk_level"], ["LOW", "MEDIUM", "HIGH"])
        self.assertTrue(0.0 <= res["risk_score"] <= 1.0)
        self.assertIsInstance(res["estimated_delay_min"], int)
        self.assertIn("cluster_id", res["cluster_profile"])


if __name__ == "__main__":
    unittest.main()

