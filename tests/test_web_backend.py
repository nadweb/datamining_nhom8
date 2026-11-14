# -*- coding: utf-8 -*-
"""Backend contract and HTTP tests using a consistent synthetic artifact bundle."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier

from modeling import predict
from modeling.encoding import FlightDataPreprocessor
from tests.ml_synth import make_ml_features
from web import app as web


class TestBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.processed, cls.models = cls.root / "processed", cls.root / "models"
        cls.processed.mkdir()
        cls.models.mkdir()
        df = make_ml_features(180, seed=17)
        prep = FlightDataPreprocessor().fit(df)
        prep.save(cls.processed, extra_meta={"train_rows": 180, "test_rows": 20, "cutoff_date": "2026-10-30"})
        X = prep.to_dataframe(prep.transform(df))
        y = np.arange(len(df)) % 3
        joblib.dump(RandomForestClassifier(n_estimators=4, random_state=42).fit(X, y), cls.models / "best_classifier.joblib")
        joblib.dump(DummyRegressor(strategy="mean").fit(X, np.arange(len(df)) % 40), cls.models / "best_regressor.joblib")
        joblib.dump(KMeans(n_clusters=2, n_init=2, random_state=42).fit(X), cls.models / "kmeans_model.joblib")
        profiles = {str(i): {"cluster_id": i, "name": f"Cluster {i}", "delay_rate_pct": 20.0,
                             "cancellation_rate_pct": 1.0, "flight_count": 90,
                             "percentage": 50, "avg_delay_minutes": 10, "risk_tier": "MEDIUM"}
                    for i in range(2)}
        (cls.models / "cluster_profiles.json").write_text(json.dumps(profiles), encoding="utf-8")
        cls.supervised = {"split": {"train_rows": 180, "test_rows": 20, "cutoff_date": "2026-10-30"},
                          "environment": {"sklearn": sklearn.__version__}, "classification": {}, "regression": {}}
        (cls.models / "supervised_metrics.json").write_text(json.dumps(cls.supervised), encoding="utf-8")
        unsup = {"environment": {"n_train": 180, "n_test": 20, "sklearn": sklearn.__version__},
                 "isolation_forest": {"anomaly_count": 4, "anomaly_percentage": 2}, "pca": {}}
        (cls.models / "unsupervised_metrics.json").write_text(json.dumps(unsup), encoding="utf-8")
        cls.rules = cls.root / "rules.csv"
        pd.DataFrame([
            {"Antecedent": "Airline=VJ & Airport=SGN & Slot=18h-24h", "Consequent": "Status=delayed",
             "Confidence (%)": 42.5, "Lift": 1.62, "Support (%)": 2.0},
            {"Antecedent": "Airline=VJ & Load=high", "Consequent": "Status=delayed",
             "Confidence (%)": 50.0, "Lift": 2.0, "Support (%)": 1.0},
        ]).to_csv(cls.rules, index=False)
        cls.server = web.ThreadingHTTPServer(("127.0.0.1", 0), web.FlightApiHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        cls.tmp.cleanup()

    def setUp(self):
        self.pred_patch = mock.patch.multiple(
            predict, PROCESSED_DIR=self.processed, FILE_CLASSIFIER=self.models / "best_classifier.joblib",
            FILE_REGRESSOR=self.models / "best_regressor.joblib", FILE_KMEANS=self.models / "kmeans_model.joblib",
            FILE_PROFILES=self.models / "cluster_profiles.json", FILE_SUPERVISED_METRICS=self.models / "supervised_metrics.json",
            FILE_UNSUPERVISED_METRICS=self.models / "unsupervised_metrics.json", FILE_RULES=self.rules)
        self.web_patch = mock.patch.multiple(web, MODELS_DIR=self.models, FILE_RULES=self.rules)
        self.pred_patch.start()
        self.web_patch.start()
        predict._CACHE.clear()
        self.payload = {"airline_code": "VJ", "airport": "SGN", "direction": "DEPARTURES",
                        "route_type": "domestic", "scheduled_time": "19:30", "flight_date": "2026-11-07",
                        "precipitation": 1.0, "wind_gusts_10m": 35, "visibility": 4000,
                        "weather_code": 95, "airport_load": 34}

    def tearDown(self):
        predict._CACHE.clear()
        self.web_patch.stop()
        self.pred_patch.stop()

    def request(self, path, payload=None, content_type="application/json"):
        data = None if payload is None else (payload if isinstance(payload, bytes) else json.dumps(payload).encode())
        req = Request(self.url + path, data=data, headers={"Content-Type": content_type})
        try:
            with urlopen(req, timeout=10) as response:
                return response.status, json.loads(response.read())
        except HTTPError as exc:
            try:
                body = exc.read()
                return exc.code, json.loads(body) if exc.headers.get_content_type() == "application/json" else body
            finally:
                exc.close()

    def test_prediction_contract_and_complete_rule_matching(self):
        result = predict.predict_risk(**self.payload)
        self.assertEqual(set(result["probabilities"]), {"on_time", "delayed", "cancelled"})
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=3)
        self.assertAlmostEqual(result["risk_score"], result["probabilities"]["delayed"] +
                               result["probabilities"]["cancelled"], places=3)
        self.assertIn(result["risk_level"], {"LOW", "MEDIUM", "HIGH"})
        self.assertIn("cluster_name", result["cluster_profile"])
        self.assertIn("cluster_risk_avg", result["cluster_profile"])
        self.assertIn("name", result["cluster_profile"])  # frontend compatibility
        self.assertEqual(len(result["matched_rules"]), 1)
        self.assertEqual(result["matched_rules"][0]["confidence"], 42.5)
        self.assertEqual(result["matched_rules"][0]["confidence_pct"], 42.5)
        changed = dict(self.payload, scheduled_time="10:00")
        self.assertEqual(predict.predict_risk(**changed)["matched_rules"], [])

    def test_classifier_classes_are_mapped_by_label_not_position(self):
        artifacts = predict._load_artifacts()
        fake = mock.Mock()
        fake.classes_ = np.array([2, 0, 1])
        fake.predict_proba.return_value = np.array([[0.1, 0.7, 0.2]])
        artifacts["classifier"] = fake
        result = predict.predict_risk(**self.payload)
        self.assertEqual(result["prediction"], "on_time")
        self.assertEqual(result["probabilities"], {"on_time": 0.7, "delayed": 0.2, "cancelled": 0.1})

    def test_weekend_and_vietnam_holiday_flags_reach_preprocessor(self):
        prep = predict._load_artifacts()["preprocessor"]
        original = prep.transform
        observed = []
        def capture(frame):
            observed.append((int(frame["is_weekend_flag"].iloc[0]), int(frame["is_holiday_flag"].iloc[0])))
            return original(frame)
        with mock.patch.object(prep, "transform", side_effect=capture):
            predict.predict_risk(**self.payload)  # Saturday
            predict.predict_risk(**dict(self.payload, flight_date="2026-01-01"))  # New Year's Day
        self.assertEqual(observed, [(1, 0), (0, 1)])

    def test_bad_inputs_are_rejected_before_inference(self):
        for changes in ({"flight_date": "2026-02-30"}, {"scheduled_time": "24:00"},
                        {"airport": "XXX"}, {"visibility": -1}, {"airport_load": 3.5},
                        {"precipitation": float("nan")}, {"weather_code": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                predict.predict_risk(**dict(self.payload, **changes))

    def test_rule_bin_boundaries_and_unknown_load(self):
        values = predict._validate_inputs(**dict(self.payload, precipitation=2.5, wind_gusts_10m=30,
                                                  visibility=5000, weather_code=0, origin="SGN", destination="HAN"))
        items = predict._rule_items(values)
        self.assertTrue({"Rain=heavy", "Gust=30-50", "Vis=>=5km", "Day=weekend"} <= items)
        self.assertFalse(any(item.startswith("Load=") for item in items))
        values["precipitation"] = values["wind_gusts_10m"] = values["visibility"] = 1e9
        self.assertFalse(any(item.startswith(("Rain=", "Gust=", "Vis=")) for item in predict._rule_items(values)))

    def test_mismatched_artifacts_fail_closed_and_cache_stays_empty(self):
        bad = self.root / "wrong_metrics.json"
        bad.write_text(json.dumps(dict(self.supervised, split={"train_rows": 999})), encoding="utf-8")
        with mock.patch.object(predict, "FILE_SUPERVISED_METRICS", bad):
            with self.assertRaisesRegex(predict.ArtifactError, "không khớp"):
                predict._load_artifacts()
            self.assertEqual(predict._CACHE, {})
            status, body = self.request("/health")
            self.assertEqual(status, 503)
            self.assertFalse(body["ready"])
            status, body = self.request("/api/predict", self.payload)
            self.assertEqual(status, 503)
            self.assertIn("không khớp", body["error"])

    def test_lineage_accepts_patch_version_only_and_checks_legacy_rows(self):
        major, minor = sklearn.__version__.split(".")[:2]
        predict._check_training_lineage({"sklearn_version": f"{major}.{minor}.999"}, {}, {})
        with self.assertRaisesRegex(predict.ArtifactError, "scikit-learn"):
            predict._check_training_lineage({"sklearn_version": f"{major}.{int(minor) + 1}.0"}, {}, {})
        with self.assertRaisesRegex(predict.ArtifactError, "train_rows"):
            predict._check_training_lineage({"train_rows": 180}, {"metadata": {"train_rows": 181}}, {})

    def test_encoder_metadata_mismatch_is_rejected(self):
        artifacts = predict._load_artifacts()
        prep = artifacts["preprocessor"]
        prep.encoded_cat_names[0] = "wrong_category"
        with self.assertRaisesRegex(predict.ArtifactError, "encoder"):
            predict._check_feature_schema(prep, artifacts)

    def test_missing_artifact_does_not_create_partial_cache(self):
        with mock.patch.object(predict, "FILE_KMEANS", self.root / "missing.joblib"):
            with self.assertRaisesRegex(predict.ArtifactError, "missing.joblib"):
                predict._load_artifacts()
            self.assertFalse(predict._CACHE)
        with mock.patch.object(web, "FILE_RULES", self.root / "missing.csv"):
            status, body = self.request("/api/rules")
            self.assertEqual(status, 503)
            self.assertIn("error", body)

    def test_http_endpoints_and_invalid_payloads(self):
        for path in ("/health", "/api/stats", "/api/clusters", "/api/rules"):
            with self.subTest(path=path):
                self.assertEqual(self.request(path)[0], 200)
        status, result = self.request("/api/predict", self.payload)
        self.assertEqual(status, 200)
        self.assertIn("flight_summary", result)
        status, rules = self.request("/api/rules")
        self.assertEqual([r["Lift"] for r in rules], [2.0, 1.62])
        for invalid in (b"{", b"[]", b'{"x":NaN}', json.dumps(dict(self.payload, delay_min=10)).encode()):
            self.assertEqual(self.request("/api/predict", invalid)[0], 400)
        self.assertEqual(self.request("/api/predict", self.payload, "text/plain")[0], 415)
        self.assertEqual(self.request("/api/predict", {})[0], 400)
        self.assertEqual(self.request("/api/predict", b"x" * (web.MAX_BODY_BYTES + 1))[0], 413)
        self.assertEqual(self.request("/static/%2e%2e/config.py")[0], 404)


if __name__ == "__main__":
    unittest.main()
