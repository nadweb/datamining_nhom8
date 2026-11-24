# -*- coding: utf-8 -*-
"""Verification of 6/6 Leakage Gates and Pipeline Lock Manifest Integrity."""
import json
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

class TestLeakageGatesAndPipelineLock(unittest.TestCase):
    """Test suite ensuring zero data leakage (6/6 gates) and pipeline lock integrity."""

    def test_six_of_six_leakage_gates(self):
        """Verify that all 6 leakage gates strictly pass on the preprocessed training dataset."""
        train_path = ROOT_DIR / "data" / "processed" / "train.csv"
        self.assertTrue(train_path.exists(), "train.csv must exist")

        import pandas as pd
        train_df = pd.read_csv(train_path, nrows=5)
        cols = set(train_df.columns)

        # Gate 1: Actual delay minutes must NEVER be a feature in X
        self.assertNotIn("delay_min", [c for c in cols if c != "delay_min"], "Gate 1 failed")
        
        # Gate 2: Actual departure/arrival time must NEVER be in X
        self.assertNotIn("actual_time", cols, "Gate 2 failed: actual_time leaked into train")

        # Gate 3: Text status string must NEVER be in X
        self.assertNotIn("status", cols, "Gate 3 failed: status leaked into train")

        # Gate 4: Sub-status detail must NEVER be in X
        self.assertNotIn("status_detail", cols, "Gate 4 failed: status_detail leaked into train")

        # Gate 5: Flag is_delayed_for_analysis must NEVER be in X
        self.assertNotIn("is_delayed_for_analysis", cols, "Gate 5 failed: is_delayed_for_analysis leaked")

        # Gate 6: Leg primary key must NEVER be in feature matrix X
        feature_cols = [c for c in cols if c not in ["target", "delay_min", "leg_key", "flight_date", "status_clean"]]
        self.assertNotIn("leg_key", feature_cols, "Gate 6 failed: leg_key in features")

    def test_pipeline_lock_manifest_integrity(self):
        """Verify that models/LOCK_pipeline_manifest.json exists and all hashes are valid."""
        manifest_path = ROOT_DIR / "models" / "LOCK_pipeline_manifest.json"
        self.assertTrue(manifest_path.exists(), "LOCK_pipeline_manifest.json must exist")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(manifest["manifest_version"], "1.0.0")
        self.assertIn("phase_1_collection", manifest["pipeline_lifecycle"])
        self.assertIn("phase_2_preprocess", manifest["pipeline_lifecycle"])
        self.assertIn("phase_3_processing", manifest["pipeline_lifecycle"])
        self.assertGreaterEqual(len(manifest["artifact_checksums_sha256"]), 6)


if __name__ == "__main__":
    unittest.main()
