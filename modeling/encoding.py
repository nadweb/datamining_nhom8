# -*- coding: utf-8 -*-
"""Encoding and Normalization Module (Part 4 - Member 4)
Handles:
1. Feature subset selection (Strict zero-leakage enforcement)
2. One-Hot Encoding for categorical features (fit on feature matrix)
3. Standard Scaling for continuous numerical features (fit on feature matrix)
4. Passthrough preservation for binary flag indicators
5. FlightDataPreprocessor packaging scikit-learn transformers
6. Exporting transformer artifacts: scaler.joblib, encoder.joblib, preprocessing_metadata.json

Note: Chronological train/test splitting and model training are handled by Member 5 in modeling/train.py.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import BASE_DIR
from features.features import build_engineered_features

PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FILE_SCALER = PROCESSED_DIR / "scaler.joblib"
FILE_ENCODER = PROCESSED_DIR / "encoder.joblib"
FILE_METADATA = PROCESSED_DIR / "preprocessing_metadata.json"
FILE_TRAIN = PROCESSED_DIR / "train.csv"
FILE_TEST = PROCESSED_DIR / "test.csv"
FILE_PROCESSED = PROCESSED_DIR / "flights_processed.csv"

# Target mapping for classification
LABEL_MAP = {
    "on_time": 0,
    "delayed": 1,
    "cancelled": 2,
}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# LEAKAGE COLUMNS - MUST NEVER BE INCLUDED IN X
LEAKAGE_COLUMNS = [
    "delay_min", "actual_time", "status", "status_detail"
]

# IDENTIFIERS TO EXCLUDE FROM X
EXCLUDE_COLUMNS = LEAKAGE_COLUMNS + [
    "leg_key", "source", "status_clean", "flight_date", "flight_no",
    "scheduled_time", "origin", "destination", "other_code", "other_airport"
]

# Features to encode and scale
CATEGORICAL_COLS = [
    "airline_code", "airport", "direction", "route_type", "carrier_category"
]

NUMERIC_CONTINUOUS_COLS = [
    "airport_load", "precipitation", "wind_gusts_10m", "visibility",
    "weather_hazard_score", "load_x_weather", "hour_sin", "hour_cos",
    "dow_sin", "dow_cos"
]

PASSTHROUGH_BINARY_COLS = [
    "is_peak_hour", "is_heavy_traffic", "is_rain", "is_heavy_rain",
    "is_high_wind", "is_severe_wind", "is_low_vis", "is_storm",
    "is_golden_triangle", "is_holiday_flag", "is_weekend_flag",
    "is_domestic_flag", "is_departure", "peak_x_rain"
]


def split_by_date(
    df: pd.DataFrame, train_ratio: float = 0.80
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Perform chronological train-test split without shuffling (owned by Member 5)."""
    df_sorted = df.sort_values(["flight_date", "scheduled_time"]).reset_index(drop=True)
    split_idx = int(len(df_sorted) * train_ratio)
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    cutoff_date = str(train_df["flight_date"].max())
    return train_df, test_df, cutoff_date


class FlightDataPreprocessor:
    """Production-grade transformer packaging OneHotEncoder and StandardScaler."""

    def __init__(self, encoder=None, scaler=None, feature_names=None, encoded_cat_names=None):
        self.encoder = encoder or OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.scaler = scaler or StandardScaler()
        self.feature_names: List[str] = feature_names or []
        self.encoded_cat_names: List[str] = encoded_cat_names or []
        self.is_fitted = encoder is not None and scaler is not None

    def fit(self, X: pd.DataFrame) -> "FlightDataPreprocessor":
        # 1. Fit OneHotEncoder on categorical features
        cat_data = X[CATEGORICAL_COLS].astype(str).fillna("missing")
        self.encoder.fit(cat_data)
        self.encoded_cat_names = list(self.encoder.get_feature_names_out(CATEGORICAL_COLS))

        # 2. Fit StandardScaler on continuous numeric features
        num_data = X[NUMERIC_CONTINUOUS_COLS].fillna(0.0)
        self.scaler.fit(num_data)

        # 3. Store full feature column order
        self.feature_names = self.encoded_cat_names + NUMERIC_CONTINUOUS_COLS + PASSTHROUGH_BINARY_COLS
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("FlightDataPreprocessor must be fitted before transforming!")

        # 1. Transform categorical
        cat_data = X[CATEGORICAL_COLS].astype(str).fillna("missing")
        cat_encoded = self.encoder.transform(cat_data)

        # 2. Transform continuous
        num_data = X[NUMERIC_CONTINUOUS_COLS].fillna(0.0)
        num_scaled = self.scaler.transform(num_data)

        # 3. Get passthrough binaries
        bin_data = X[PASSTHROUGH_BINARY_COLS].fillna(0).astype(float).values

        # 4. Concatenate horizontally
        return np.hstack([cat_encoded, num_scaled, bin_data])

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    def to_dataframe(self, X_transformed: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(X_transformed, columns=self.feature_names)

    def save(self, target_dir: Path = PROCESSED_DIR):
        """Save standard scikit-learn transformers and metadata to avoid custom pickle dependencies."""
        target_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.encoder, target_dir / "encoder.joblib")
        joblib.dump(self.scaler, target_dir / "scaler.joblib")
        meta = {
            "feature_names": self.feature_names,
            "encoded_cat_names": self.encoded_cat_names,
            "categorical_columns": CATEGORICAL_COLS,
            "numeric_continuous_columns": NUMERIC_CONTINUOUS_COLS,
            "binary_columns": PASSTHROUGH_BINARY_COLS,
            "label_mapping": LABEL_MAP,
        }
        with open(target_dir / "preprocessing_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, target_dir: Path = PROCESSED_DIR) -> "FlightDataPreprocessor":
        """Load preprocessor from standard serialized artifacts."""
        enc = joblib.load(target_dir / "encoder.joblib")
        scl = joblib.load(target_dir / "scaler.joblib")
        meta_file = target_dir / "preprocessing_metadata.json"
        feature_names = []
        encoded_cat_names = []
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
                feature_names = meta.get("feature_names", [])
                encoded_cat_names = meta.get("encoded_cat_names", [])
        return cls(encoder=enc, scaler=scl, feature_names=feature_names, encoded_cat_names=encoded_cat_names)


def build_and_save_encoded_dataset(
    source_df: pd.DataFrame = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, FlightDataPreprocessor]:
    """Execute complete encoding and splitting pipeline and persist artifacts."""
    if source_df is None:
        features_path = PROCESSED_DIR / "flights_features.csv"
        if not features_path.exists():
            print("Feature dataset not found. Running feature engineering first...")
            from features.features import run_feature_pipeline
            source_df = run_feature_pipeline()
        else:
            source_df = pd.read_csv(features_path)

    # Filter to known labels
    clean_df = source_df[source_df["status_clean"].isin(LABEL_MAP.keys())].copy()

    # Chronological Split
    train_raw, test_raw, cutoff_date = split_by_date(clean_df, train_ratio=0.80)
    print(f"Chronological split at cutoff date: {cutoff_date}")
    print(f"Train set: {len(train_raw):,} flights | Test set: {len(test_raw):,} flights")

    # Target extraction
    y_train = train_raw["status_clean"].map(LABEL_MAP).astype(int)
    y_test = test_raw["status_clean"].map(LABEL_MAP).astype(int)

    # Preprocessor fit ONLY on Train
    preprocessor = FlightDataPreprocessor()
    preprocessor.fit(train_raw)

    # Transform Train and Test
    X_train_mat = preprocessor.transform(train_raw)
    X_test_mat = preprocessor.transform(test_raw)

    X_train = preprocessor.to_dataframe(X_train_mat)
    X_test = preprocessor.to_dataframe(X_test_mat)

    # Save artifacts using standard scikit-learn serializers
    preprocessor.save(PROCESSED_DIR)

    # Save metadata
    metadata = {
        "cutoff_date": cutoff_date,
        "train_rows": len(train_raw),
        "test_rows": len(test_raw),
        "total_features": len(preprocessor.feature_names),
        "feature_names": preprocessor.feature_names,
        "encoded_cat_names": preprocessor.encoded_cat_names,
        "categorical_columns": CATEGORICAL_COLS,
        "one_hot_columns": preprocessor.encoded_cat_names,
        "numeric_continuous_columns": NUMERIC_CONTINUOUS_COLS,
        "binary_columns": PASSTHROUGH_BINARY_COLS,
        "label_mapping": LABEL_MAP,
    }
    with open(FILE_METADATA, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Export CSVs for downstream models
    train_export = X_train.copy()
    train_export["target"] = y_train.values
    train_export["delay_min"] = train_raw["delay_min"].values
    train_export.to_csv(FILE_TRAIN, index=False)

    test_export = X_test.copy()
    test_export["target"] = y_test.values
    test_export["delay_min"] = test_raw["delay_min"].values
    test_export.to_csv(FILE_TEST, index=False)

    full_export = pd.concat([train_export, test_export], ignore_index=True)
    full_export.to_csv(FILE_PROCESSED, index=False)

    print(f"Saved artifacts to {PROCESSED_DIR}:")
    print(f"  - {FILE_TRAIN.name} ({train_export.shape[0]}x{train_export.shape[1]})")
    print(f"  - {FILE_TEST.name} ({test_export.shape[0]}x{test_export.shape[1]})")
    print(f"  - {FILE_METADATA.name} ({len(preprocessor.feature_names)} features)")
    print(f"  - scaler.joblib, encoder.joblib, preprocessor.joblib")

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    build_and_save_encoded_dataset()
