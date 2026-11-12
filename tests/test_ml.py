# -*- coding: utf-8 -*-
"""Kiểm thử Part 5 (học có giám sát và không giám sát). Chạy từ thư mục gốc dự án:

    python -m unittest tests.test_ml -v

Dùng dữ liệu giả lập tự sinh trong thư mục tạm; không đụng tới data/ và models/ thật.
"""
import inspect
import json
import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

if "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from modeling import encoding as enc
from modeling import train as tr
from modeling import unsupervised as un
from tests.ml_synth import make_ml_features

warnings.filterwarnings("ignore")


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        cls.proc, cls.models = d / "processed", d / "models"
        cls.proc.mkdir()
        cls.features = make_ml_features(2500, seed=4)
        train_raw, test_raw, cls.cutoff = enc.split_by_date(cls.features)
        cls.train_raw, cls.test_raw = train_raw, test_raw
        pre = enc.FlightDataPreprocessor().fit(train_raw)
        for name, raw in [("train.csv", train_raw), ("test.csv", test_raw)]:
            out = pre.to_dataframe(pre.transform(raw))
            out["target"] = raw["status_clean"].map(enc.LABEL_MAP).astype(int).values
            out["delay_min"] = raw["delay_min"].values
            out.to_csv(cls.proc / name, index=False)
        pre.save(cls.proc, extra_meta={"cutoff_date": cls.cutoff, "train_rows": len(train_raw), "test_rows": len(test_raw)})
        rules = pd.DataFrame({"antecedent": list("abcde"), "lift": [1.2, 3.1, 0.9, 2.2, 1.7], "confidence": 0.5, "support": 0.1})
        rules.to_csv(d / "rules.csv", index=False)

        cls.patches = [
            mock.patch.multiple(tr, FILE_TRAIN=cls.proc / "train.csv", FILE_TEST=cls.proc / "test.csv",
                                FILE_METADATA=cls.proc / "preprocessing_metadata.json", MODELS_DIR=cls.models,
                                FILE_CLASSIFIER=cls.models / "best_classifier.joblib",
                                FILE_REGRESSOR=cls.models / "best_regressor.joblib",
                                FILE_METRICS=cls.models / "supervised_metrics.json"),
            mock.patch.multiple(un, FILE_TRAIN=cls.proc / "train.csv", FILE_TEST=cls.proc / "test.csv",
                                PROCESSED_DIR=cls.proc, MODELS_DIR=cls.models, FILE_RULES=d / "rules.csv",
                                FILE_KMEANS=cls.models / "kmeans_model.joblib", FILE_PCA=cls.models / "pca_model.joblib",
                                FILE_ISOLATION_FOREST=cls.models / "isolation_forest.joblib",
                                FILE_CLUSTER_PROFILES=cls.models / "cluster_profiles.json",
                                FILE_UNSUPERVISED_METRICS=cls.models / "unsupervised_metrics.json"),
        ]
        for p in cls.patches:
            p.start()

    @classmethod
    def tearDownClass(cls):
        for p in cls.patches:
            p.stop()
        cls.tmp.cleanup()


class TestSupervised(_Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.metrics = tr.run_supervised_training()
        cls.Xtr, cls.Xte, cls.ytr, cls.yte, cls.dtr, cls.dte = tr.load_train_test_data()

    def test_split_is_chronological(self):
        self.assertLessEqual(self.train_raw["flight_date"].max(), self.test_raw["flight_date"].min())
        self.assertEqual(self.metrics["split"]["train_rows"] + self.metrics["split"]["test_rows"], len(self.features))

    def test_no_leakage_columns_in_model_inputs(self):
        for c in list(enc.LEAKAGE_COLUMNS) + ["target", "status_clean", "delay_min"]:
            self.assertNotIn(c, self.Xtr.columns)
        clf = joblib.load(self.models / "best_classifier.joblib")
        self.assertEqual(clf.n_features_in_, self.Xtr.shape[1])

    def test_model_selection_never_sees_test_data(self):
        params = list(inspect.signature(tr.select_classifier_by_time_cv).parameters)
        self.assertEqual(params[:2], ["X_train", "y_train"])
        self.assertNotIn("X_test", params)
        a = tr.select_classifier_by_time_cv(self.Xtr, self.ytr)
        b = tr.select_classifier_by_time_cv(self.Xtr, self.ytr)
        self.assertEqual(a, b)                                    # có hạt giống cố định: lặp lại cho kết quả giống hệt
        self.assertNotEqual(a[0], "Baseline_MostFrequent")

    def test_metrics_schema_is_backward_compatible(self):
        clf = self.metrics["classification"]
        self.assertIn(clf["best_classifier_name"], ["LogisticRegression", "RandomForest", "HistGradientBoosting"])
        for name in ["LogisticRegression", "RandomForest", "HistGradientBoosting"]:
            for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "roc_auc_ovr",
                        "confusion_matrix", "classification_report"]:
                self.assertIn(key, clf[name])
            self.assertEqual(np.array(clf[name]["confusion_matrix"]).shape, (3, 3))
        for key in ["MAE_minutes", "RMSE_minutes", "R2_Score"]:
            self.assertIn(key, self.metrics["regression"])
        saved = json.loads((self.models / "supervised_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["classification"]["best_classifier_name"], clf["best_classifier_name"])

    def test_baseline_reported_and_beaten_on_macro_f1(self):
        base = self.metrics["baselines"]["classifier_most_frequent"]
        best = max(self.metrics["classification"][n]["f1_macro"] for n in ["LogisticRegression", "RandomForest", "HistGradientBoosting"])
        self.assertLessEqual(base["f1_macro"], best)
        self.assertIn("Baseline_MostFrequent", self.metrics["model_selection"]["cv"])
        self.assertIn("baseline_median_MAE_minutes", self.metrics["regression_details"])

    def test_feature_importances_not_empty_and_normalised(self):
        fi = self.metrics["feature_importances"]
        self.assertGreater(len(fi), 5)
        self.assertLessEqual(sum(i["importance"] for i in fi), 1.0 + 1e-6)
        self.assertTrue(self.metrics["feature_importance_by_variable"])

    def test_histgradientboosting_gets_permutation_importance(self):
        hgb = tr.HistGradientBoostingClassifier(max_iter=60, class_weight="balanced", random_state=42).fit(self.Xtr, self.ytr)
        self.assertFalse(hasattr(hgb, "feature_importances_"))     # nguyên nhân của feature_importances = [] ở bản cũ
        t = tr._importance_table(hgb, list(self.Xtr.columns), self.Xte, self.yte)
        self.assertEqual(len(t), self.Xtr.shape[1])
        self.assertIn("permutation", t["method"].iloc[0])
        self.assertAlmostEqual(float(t["importance"].sum()), 1.0, places=6)

    def test_group_importances_keeps_airport_load_separate(self):
        t = pd.DataFrame({"feature": ["airport_SGN", "airport_HAN", "airport_load", "airline_code_VN"],
                          "importance": [0.1, 0.2, 0.3, 0.4]})
        g = {r["variable"]: r["importance"] for r in tr.group_importances_by_variable(t)}
        self.assertAlmostEqual(g["airport"], 0.3)
        self.assertAlmostEqual(g["airport_load"], 0.3)
        self.assertAlmostEqual(g["airline_code"], 0.4)

    def test_calibration_table_counts(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 1, 500)
        y = (rng.uniform(0, 1, 500) < p).astype(int)
        cal = tr.calibration_table(y, p, bins=5)
        self.assertEqual(sum(b["n"] for b in cal["bins"]), 500)
        self.assertIn("brier_delayed", cal)
        self.assertIn("probability_calibration", self.metrics)


class TestUnsupervised(_Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.summary = un.run_unsupervised_pipeline()

    def test_profiles_have_four_unique_data_derived_names(self):
        profiles = self.summary["kmeans"]["profiles"]
        self.assertEqual(len(profiles), 4)
        names = [p["name"] for p in profiles.values()]
        self.assertEqual(len(set(names)), 4)
        self.assertTrue(any("Rain/Storm" in n for n in names), names)        # dữ liệu giả lập có nhóm mưa lớn thật
        rainy = next(p for p in profiles.values() if "Rain/Storm" in p["name"])
        self.assertGreater(rainy["avg_precipitation_mm"], 1.0)
        self.assertEqual(un.CLUSTER_NAMES[rainy["cluster_id"]], rainy["name"])

    def test_profile_keys_and_stability(self):
        for p in self.summary["kmeans"]["profiles"].values():
            for key in ["cluster_id", "name", "flight_count", "percentage", "delay_rate_pct", "cancellation_rate_pct",
                        "avg_delay_minutes", "risk_tier", "train_delay_rate_pct", "test_delay_rate_pct"]:
                self.assertIn(key, p)
            self.assertIn(p["risk_tier"], ["LOW", "MEDIUM", "HIGH"])
        self.assertEqual(sum(p["flight_count"] for p in self.summary["kmeans"]["profiles"].values()), len(self.features))

    def test_k_sweep_is_reproducible_and_stored(self):
        self.assertEqual([r["k"] for r in self.summary["kmeans"]["k_selection"]], list(range(2, 9)))
        X_tr, *_ = un.load_train_test_matrices()
        self.assertEqual(un.sweep_k(X_tr, range(2, 5)), un.sweep_k(X_tr, range(2, 5)))
        self.assertIn("silhouette", self.summary["kmeans"])

    def test_pca_keeps_old_and_new_keys(self):
        pca = self.summary["pca"]
        for key in ["explained_variance_ratios", "total_explained_variance_pct", "explained_variance_ratio_2d",
                    "cumulative_variance_2d", "top_loadings"]:
            self.assertIn(key, pca)
        self.assertLessEqual(pca["cumulative_variance_2d"], 1.0)

    def test_isolation_forest_reports_what_makes_flights_anomalous(self):
        iso = self.summary["isolation_forest"]
        self.assertAlmostEqual(iso["train_flagged_pct"], 3.0, delta=0.5)      # do contamination quy định
        self.assertIn("anomaly_pct", iso)
        self.assertGreater(len(iso["top_distinguishing_features"]), 0)
        self.assertIn("delay_rate_flagged_pct", iso)

    def test_rules_sorted_by_lift_descending(self):
        rules = self.summary["association_rules"]
        lifts = [r["lift"] for r in rules]
        self.assertEqual(lifts, sorted(lifts, reverse=True))
        self.assertEqual(lifts[0], 3.1)

    def test_dominant_ignores_airport_load_column(self):
        X = pd.DataFrame({"airport_load": [9.0, 9.0], "airport_SGN": [1.0, 1.0], "airport_HAN": [0.0, 0.0]})
        self.assertEqual(un._dominant(X, np.array([True, True]), "airport_"), "SGN")


if __name__ == "__main__":
    unittest.main()
