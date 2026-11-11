# -*- coding: utf-8 -*-
"""Kiểm thử Part 4 (Encoding & Normalization). Chạy từ thư mục gốc dự án:

    python -m unittest tests.test_encoding -v

Dùng dữ liệu giả lập tự sinh trong bài test, nên không cần data/ thật.
"""
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from modeling import encoding as enc
from modeling.encoding import (
    CATEGORICAL_COLS, LEAKAGE_COLUMNS, NUMERIC_CONTINUOUS_COLS, PASSTHROUGH_BINARY_COLS,
    FlightDataPreprocessor, assert_no_leakage, split_by_date,
)


def make_features_df(n=600, seed=0):
    """Bảng đặc trưng giả lập có đủ cột như flights_features.csv (kèm cột rò rỉ và định danh)."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2026-10-20", periods=10).strftime("%Y-%m-%d")
    df = pd.DataFrame({
        "leg_key": [f"VN{i}|{days[i % 10]}|SGN>HAN" for i in range(n)],
        "flight_date": rng.choice(days, n),
        "scheduled_time": [f"{h:02d}:{m:02d}" for h, m in zip(rng.integers(5, 23, n), rng.choice([0, 15, 30, 45], n))],
        "airline_code": rng.choice(["VN", "VJ", "QH", "BL", "9G"], n),
        "airport": rng.choice(["SGN", "HAN", "DAD", "CXR", "PQC"], n),
        "direction": rng.choice(["ARRIVALS", "DEPARTURES"], n),
        "route_type": rng.choice(["domestic", "international"], n, p=[0.85, 0.15]),
        "carrier_category": rng.choice(["FSC", "LCC", "REGIONAL"], n),
        "status_clean": rng.choice(["on_time", "delayed", "cancelled"], n, p=[0.55, 0.38, 0.07]),
    })
    df["delay_min"] = np.where(df["status_clean"] == "delayed", rng.integers(15, 120, n), rng.integers(-5, 10, n)).astype(float)
    df["actual_time"] = "10:00"
    df["status"] = "Departed On-Time"
    df["status_detail"] = "departed_on_time"
    df["airport_load"] = rng.integers(5, 45, n).astype(float)
    df["precipitation"] = rng.exponential(0.4, n)
    df["wind_gusts_10m"] = rng.uniform(5, 60, n)
    df["visibility"] = rng.choice([24000.0, 12000.0, 8000.0, 3000.0], n)
    df["weather_hazard_score"] = rng.uniform(0, 1, n)
    df["load_x_weather"] = df["airport_load"] * df["weather_hazard_score"]
    h = rng.uniform(0, 24, n)
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    d = rng.integers(0, 7, n)
    df["dow_sin"], df["dow_cos"] = np.sin(2 * np.pi * d / 7), np.cos(2 * np.pi * d / 7)
    for c in PASSTHROUGH_BINARY_COLS:
        df[c] = rng.integers(0, 2, n)
    return df


class TestEncoding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_features_df()
        cls.train, cls.test, cls.cutoff = split_by_date(cls.df)
        cls.pre = FlightDataPreprocessor().fit(cls.train)

    # ---------- chống rò rỉ ----------
    def test_zero_leakage_in_feature_names(self):
        for col in LEAKAGE_COLUMNS + ["status_clean", "target"]:
            self.assertNotIn(col, self.pre.feature_names)
        assert_no_leakage(CATEGORICAL_COLS + NUMERIC_CONTINUOUS_COLS + PASSTHROUGH_BINARY_COLS)

    def test_assert_no_leakage_raises(self):
        with self.assertRaises(ValueError):
            assert_no_leakage(["airport_load", "delay_min"])

    def test_transform_does_not_depend_on_leakage_columns(self):
        """Xáo trộn các cột rò rỉ và nhãn: ma trận X phải giữ nguyên (X không phụ thuộc chúng)."""
        shuffled = self.test.copy()
        for c in LEAKAGE_COLUMNS + ["status_clean"]:
            shuffled[c] = shuffled[c].sample(frac=1, random_state=1).values
        np.testing.assert_array_equal(self.pre.transform(self.test), self.pre.transform(shuffled))

    # ---------- One-Hot ----------
    def test_output_shape_matches_feature_names(self):
        X = self.pre.transform(self.test)
        self.assertEqual(X.shape, (len(self.test), len(self.pre.feature_names)))
        self.assertEqual(len(set(self.pre.feature_names)), len(self.pre.feature_names))
        n_onehot = sum(len(c) for c in self.pre.encoder.categories_)
        self.assertEqual(len(self.pre.feature_names), n_onehot + len(NUMERIC_CONTINUOUS_COLS) + len(PASSTHROUGH_BINARY_COLS))

    def test_each_categorical_has_exactly_one_hot(self):
        X = self.pre.to_dataframe(self.pre.transform(self.test))
        for col in CATEGORICAL_COLS:
            cols = [c for c in self.pre.encoded_cat_names if c.startswith(col + "_")]
            self.assertTrue((X[cols].sum(axis=1) == 1).all(), col)

    def test_unknown_category_gives_zero_vector(self):
        row = self.test.head(1).copy()
        row["airline_code"], row["airport"] = "ZZ", "XXX"
        X = self.pre.to_dataframe(self.pre.transform(row))          # không được crash
        for col in ["airline_code", "airport"]:
            cols = [c for c in self.pre.encoded_cat_names if c.startswith(col + "_")]
            self.assertEqual(float(X[cols].sum(axis=1).iloc[0]), 0.0)

    # ---------- StandardScaler ----------
    def test_scaler_fitted_on_train_only(self):
        self.assertEqual(int(self.pre.scaler.n_samples_seen_), len(self.train))
        Xtr = self.pre.to_dataframe(self.pre.transform(self.train))[NUMERIC_CONTINUOUS_COLS]
        np.testing.assert_allclose(Xtr.mean().values, 0.0, atol=1e-9)
        np.testing.assert_allclose(Xtr.std(ddof=0).values, 1.0, atol=1e-9)
        Xte = self.pre.to_dataframe(self.pre.transform(self.test))[NUMERIC_CONTINUOUS_COLS]
        self.assertGreater(np.abs(Xte.mean().values).max(), 1e-6)     # Test KHÔNG được có trung bình đúng 0

    def test_binary_passthrough(self):
        X = self.pre.to_dataframe(self.pre.transform(self.test))[PASSTHROUGH_BINARY_COLS]
        self.assertTrue(set(np.unique(X.values)) <= {0.0, 1.0})
        np.testing.assert_array_equal(X.values, self.test[PASSTHROUGH_BINARY_COLS].astype(float).values)

    # ---------- giá trị thiếu ----------
    def test_missing_values_imputed_with_train_median(self):
        row = self.test.head(1).copy()
        row["visibility"], row["airport_load"] = np.nan, np.nan
        out = self.pre.to_dataframe(self.pre.transform(row))
        for col in ["visibility", "airport_load"]:
            med = self.train[col].median()
            i = NUMERIC_CONTINUOUS_COLS.index(col)
            expected = (med - self.pre.scaler.mean_[i]) / self.pre.scaler.scale_[i]
            self.assertAlmostEqual(float(out[col].iloc[0]), float(expected), places=9)

    def test_no_nan_in_output(self):
        dirty = self.test.copy()
        dirty.loc[dirty.index[:20], NUMERIC_CONTINUOUS_COLS] = np.nan
        dirty.loc[dirty.index[:20], CATEGORICAL_COLS] = np.nan
        dirty.loc[dirty.index[:20], PASSTHROUGH_BINARY_COLS] = np.nan
        self.assertFalse(np.isnan(self.pre.transform(dirty)).any())

    def test_missing_input_column_raises(self):
        with self.assertRaises(KeyError):
            self.pre.transform(self.test.drop(columns=["visibility"]))

    def test_transform_before_fit_raises(self):
        with self.assertRaises(ValueError):
            FlightDataPreprocessor().transform(self.test)

    # ---------- lưu / nạp ----------
    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            self.pre.save(Path(d), extra_meta={"cutoff_date": self.cutoff})
            for name in ["encoder.joblib", "scaler.joblib", "preprocessing_metadata.json"]:
                self.assertTrue((Path(d) / name).exists(), name)
            loaded = FlightDataPreprocessor.load(Path(d))
            np.testing.assert_allclose(loaded.transform(self.test), self.pre.transform(self.test))
            self.assertEqual(loaded.feature_names, self.pre.feature_names)
            meta = json.loads((Path(d) / "preprocessing_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["cutoff_date"], self.cutoff)
            self.assertEqual(meta["label_mapping"], {"on_time": 0, "delayed": 1, "cancelled": 2})

    def test_load_old_artifacts_without_medians(self):
        """Artifact cũ (metadata không có numeric_medians) vẫn dùng được: fallback điền bằng trung bình scaler."""
        with tempfile.TemporaryDirectory() as d:
            self.pre.save(Path(d))
            p = Path(d) / "preprocessing_metadata.json"
            meta = json.loads(p.read_text(encoding="utf-8"))
            for k in ["numeric_medians", "sklearn_version"]:
                meta.pop(k, None)
            p.write_text(json.dumps(meta), encoding="utf-8")
            old = FlightDataPreprocessor.load(Path(d))
            row = self.test.head(1).copy()
            row["visibility"] = np.nan
            out = old.to_dataframe(old.transform(row))
            self.assertAlmostEqual(float(out["visibility"].iloc[0]), 0.0, places=9)   # trung bình -> z = 0

    # ---------- chia thời gian ----------
    def test_split_by_date_is_chronological(self):
        self.assertLessEqual(self.train["flight_date"].max(), self.test["flight_date"].min())
        self.assertEqual(len(self.train) + len(self.test), len(self.df))
        self.assertAlmostEqual(len(self.train) / len(self.df), 0.8, places=2)


if __name__ == "__main__":
    unittest.main()
