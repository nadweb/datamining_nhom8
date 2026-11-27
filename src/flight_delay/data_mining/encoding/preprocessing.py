# -*- coding: utf-8 -*-
"""[STEP 3] Fit-Once ML Preprocessor (OneHot + StandardScaler + CausalHistory Parity)."""
import json
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn

from flight_delay.common.constants import (
    ENCODERS_DIR,
    LABEL_MAP_3CLASS,
    LABEL_MAP_BINARY,
    PROCESSED_DIR,
    SCALERS_DIR,
)
from flight_delay.data_mining.encoding.categorical import SafeCategoricalEncoder
from flight_delay.data_mining.encoding.numerical import SafeNumericScaler
from flight_delay.data_mining.feature_engineering.registry import FEATURE_REGISTRY, assert_no_leakage


class FlightPreprocessor:
    """Encapsulates Train-only OneHotEncoder, Train-median StandardScaler, and Passthrough Binary cols."""

    def __init__(self, encoder=None, scaler=None, feature_names=None, encoded_cat_names=None, numeric_medians=None):
        self.cat_cols = FEATURE_REGISTRY.categorical_columns
        self.num_cols = FEATURE_REGISTRY.numeric_columns
        self.bin_cols = FEATURE_REGISTRY.binary_columns
        self.input_cols = FEATURE_REGISTRY.model_input_columns

        self.cat_enc = SafeCategoricalEncoder(self.cat_cols)
        if encoder is not None:
            self.cat_enc.encoder = encoder
            self.cat_enc.encoded_names = list(encoded_cat_names or encoder.get_feature_names_out(self.cat_cols))

        self.num_scl = SafeNumericScaler(self.num_cols, scaler=scaler, medians=numeric_medians)
        self.is_fitted = encoder is not None and scaler is not None
        self.encoded_cat_names: List[str] = list(self.cat_enc.encoded_names)
        self.feature_names: List[str] = list(feature_names or (self.encoded_cat_names + self.num_cols + self.bin_cols if self.is_fitted else []))

    def fit(self, X: pd.DataFrame) -> "FlightPreprocessor":
        assert_no_leakage(self.input_cols)
        missing = [c for c in self.input_cols if c not in X.columns]
        if missing:
            raise KeyError(f"Missing preprocessing input columns: {missing}")

        self.cat_enc.fit(X)
        self.num_scl.fit(X)
        self.encoded_cat_names = list(self.cat_enc.encoded_names)
        self.feature_names = self.encoded_cat_names + self.num_cols + self.bin_cols
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("FlightPreprocessor must be fitted before transform().")
        missing = [c for c in self.input_cols if c not in X.columns]
        if missing:
            raise KeyError(f"Missing preprocessing input columns: {missing}")
        c_mat = self.cat_enc.transform(X)
        n_mat = self.num_scl.transform(X)
        b_mat = X[self.bin_cols].fillna(0).astype(float).values
        return np.hstack([c_mat, n_mat, b_mat])

    def to_dataframe(self, X_transformed: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(X_transformed, columns=self.feature_names)

    def save(self, target_dir: Optional[Path] = None, extra_meta: Optional[dict] = None) -> dict:
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted FlightPreprocessor.")
        proc_dir = Path(target_dir or PROCESSED_DIR)
        proc_dir.mkdir(parents=True, exist_ok=True)
        ENCODERS_DIR.mkdir(parents=True, exist_ok=True)
        SCALERS_DIR.mkdir(parents=True, exist_ok=True)

        enc_path = proc_dir / "encoder.joblib"
        scl_path = proc_dir / "scaler.joblib"
        joblib.dump(self.cat_enc.encoder, enc_path)
        joblib.dump(self.num_scl.scaler, scl_path)
        joblib.dump(self.cat_enc.encoder, ENCODERS_DIR / "encoder.joblib")
        joblib.dump(self.num_scl.scaler, SCALERS_DIR / "scaler.joblib")

        meta = {
            "feature_names": self.feature_names,
            "encoded_cat_names": self.encoded_cat_names,
            "input_columns": self.input_cols,
            "categorical_columns": self.cat_cols,
            "numeric_continuous_columns": self.num_cols,
            "binary_columns": self.bin_cols,
            "label_mapping_3class": LABEL_MAP_3CLASS,
            "label_mapping_binary": LABEL_MAP_BINARY,
            "numeric_medians": self.num_scl._fill_values().to_dict(),
            "scaler_mean": dict(zip(self.num_cols, map(float, self.num_scl.scaler.mean_))),
            "scaler_scale": dict(zip(self.num_cols, map(float, self.num_scl.scaler.scale_))),
            "scaler_n_samples_seen": int(self.num_scl.scaler.n_samples_seen_),
            "sklearn_version": sklearn.__version__,
        }
        if extra_meta:
            meta.update(extra_meta)
        (proc_dir / "preprocessing_metadata.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return meta

    @classmethod
    def load(cls, target_dir: Optional[Path] = None) -> "FlightPreprocessor":
        proc_dir = Path(target_dir or PROCESSED_DIR)
        enc_path = proc_dir / "encoder.joblib"
        scl_path = proc_dir / "scaler.joblib"
        if not enc_path.exists():
            enc_path = ENCODERS_DIR / "encoder.joblib"
        if not scl_path.exists():
            scl_path = SCALERS_DIR / "scaler.joblib"
        enc = joblib.load(enc_path)
        scl = joblib.load(scl_path)
        meta_path = proc_dir / "preprocessing_metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return cls(
            encoder=enc,
            scaler=scl,
            feature_names=meta.get("feature_names"),
            encoded_cat_names=meta.get("encoded_cat_names"),
            numeric_medians=meta.get("numeric_medians"),
        )
