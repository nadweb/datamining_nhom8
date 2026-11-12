# -*- coding: utf-8 -*-
"""Mã hóa và chuẩn hóa đặc trưng (Part 4 - Thành viên 4).

Nhiệm vụ:
1. Cô lập rò rỉ: ma trận X không bao giờ chứa delay_min, actual_time, status, status_detail,
   is_delayed_for_analysis (bản sao nhị phân của nhãn, có trong flights_features mới)
   (kiểm tra ngay khi nạp module, khi fit và khi transform).
2. Mã hóa One-Hot cho biến phân loại (handle_unknown="ignore": danh mục lạ -> vector 0, không crash).
3. Chuẩn hóa StandardScaler cho biến liên tục. Giá trị thiếu được điền bằng TRUNG VỊ CỦA TẬP TRAIN
   (không điền 0: điền 0 sẽ biến "thiếu" thành "tầm nhìn 0 m" / "sân bay không có chuyến").
4. Giữ nguyên các cờ nhị phân 0/1.
5. Đóng gói FlightDataPreprocessor (fit / transform / fit_transform / save / load).
6. Xuất artifact: encoder.joblib, scaler.joblib, preprocessing_metadata.json.

Quy tắc: transformer CHỈ được fit trên tập Train (chia theo thời gian, không shuffle).
Việc huấn luyện mô hình thuộc Thành viên 5 (modeling/train.py).
"""
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import BASE_DIR

PROCESSED_DIR = BASE_DIR / "data" / "processed"

FILE_SCALER = PROCESSED_DIR / "scaler.joblib"
FILE_ENCODER = PROCESSED_DIR / "encoder.joblib"
FILE_METADATA = PROCESSED_DIR / "preprocessing_metadata.json"
FILE_TRAIN = PROCESSED_DIR / "train.csv"
FILE_TEST = PROCESSED_DIR / "test.csv"
FILE_PROCESSED = PROCESSED_DIR / "flights_processed.csv"

# Ánh xạ nhãn đích cho bài toán phân loại 3 lớp
LABEL_MAP = {"on_time": 0, "delayed": 1, "cancelled": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# CỘT RÒ RỈ: chứa kết quả của chuyến bay, KHÔNG ĐƯỢC nằm trong X
LEAKAGE_COLUMNS = ["delay_min", "actual_time", "status", "status_detail", "is_delayed_for_analysis"]

# Mã định danh / cột không mang tính dự báo, không đưa vào X
EXCLUDE_COLUMNS = LEAKAGE_COLUMNS + [
    "leg_key", "source", "status_clean", "flight_date", "flight_no",
    "scheduled_time", "origin", "destination", "other_code", "other_airport",
]

CATEGORICAL_COLS = ["airline_code", "airport", "direction", "route_type", "carrier_category"]

NUMERIC_CONTINUOUS_COLS = [
    "airport_load", "precipitation", "wind_gusts_10m", "visibility",
    "weather_hazard_score", "load_x_weather", "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
]

PASSTHROUGH_BINARY_COLS = [
    "is_peak_hour", "is_heavy_traffic", "is_rain", "is_heavy_rain",
    "is_high_wind", "is_severe_wind", "is_low_vis", "is_storm",
    "is_golden_triangle", "is_holiday_flag", "is_weekend_flag",
    "is_domestic_flag", "is_departure", "peak_x_rain",
]

FEATURE_INPUT_COLS = CATEGORICAL_COLS + NUMERIC_CONTINUOUS_COLS + PASSTHROUGH_BINARY_COLS


def assert_no_leakage(columns) -> None:
    """Báo lỗi nếu danh sách cột đặc trưng chứa cột rò rỉ hoặc nhãn đích."""
    bad = sorted(set(columns) & (set(LEAKAGE_COLUMNS) | {"status_clean", "target"}))
    if bad:
        raise ValueError(f"Rò rỉ dữ liệu: cột {bad} không được nằm trong ma trận đặc trưng X")


# Bảo vệ ngay khi nạp module: sửa nhầm danh sách cột ở trên sẽ báo lỗi ngay
assert_no_leakage(FEATURE_INPUT_COLS)


def split_by_date(df: pd.DataFrame, train_ratio: float = 0.80) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Chia Train/Test theo thời gian, KHÔNG shuffle (Lookahead Bias).
    Sắp xếp theo (flight_date, scheduled_time); 80% đầu là Train, 20% sau là Test."""
    df_sorted = df.sort_values(["flight_date", "scheduled_time"]).reset_index(drop=True)
    split_idx = int(len(df_sorted) * train_ratio)
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    cutoff_date = str(train_df["flight_date"].max())
    return train_df, test_df, cutoff_date


class FlightDataPreprocessor:
    """Bộ tiền xử lý đóng gói OneHotEncoder + StandardScaler (+ trung vị điền thiếu của tập Train)."""

    def __init__(self, encoder=None, scaler=None, feature_names=None, encoded_cat_names=None,
                 numeric_medians: Optional[Dict[str, float]] = None):
        self.encoder = encoder if encoder is not None else OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.feature_names: List[str] = list(feature_names or [])
        self.encoded_cat_names: List[str] = list(encoded_cat_names or [])
        # None = artifact cũ (chưa có trung vị) -> dùng trung bình của scaler để điền thiếu
        self.numeric_medians_: Optional[pd.Series] = (
            pd.Series(numeric_medians, dtype=float) if numeric_medians else None
        )
        self.is_fitted = encoder is not None and scaler is not None
        if self.is_fitted and not self.encoded_cat_names:
            self.encoded_cat_names = list(self.encoder.get_feature_names_out(CATEGORICAL_COLS))
        if self.is_fitted and not self.feature_names:
            self.feature_names = self.encoded_cat_names + NUMERIC_CONTINUOUS_COLS + PASSTHROUGH_BINARY_COLS

    # ---------- các bước chuẩn bị dữ liệu dùng chung cho fit và transform ----------
    @staticmethod
    def _check_columns(X: pd.DataFrame) -> None:
        missing = [c for c in FEATURE_INPUT_COLS if c not in X.columns]
        if missing:
            raise KeyError(f"Thiếu cột đặc trưng đầu vào: {missing}")

    @staticmethod
    def _cat(X: pd.DataFrame) -> pd.DataFrame:
        return X[CATEGORICAL_COLS].astype(object).fillna("missing").astype(str)

    @staticmethod
    def _num(X: pd.DataFrame) -> pd.DataFrame:
        return X[NUMERIC_CONTINUOUS_COLS].apply(pd.to_numeric, errors="coerce").astype(float)

    def _impute_values(self) -> pd.Series:
        if self.numeric_medians_ is not None:
            return self.numeric_medians_.reindex(NUMERIC_CONTINUOUS_COLS)
        return pd.Series(self.scaler.mean_, index=NUMERIC_CONTINUOUS_COLS)

    # ---------- giao diện scikit-learn ----------
    def fit(self, X: pd.DataFrame) -> "FlightDataPreprocessor":
        """Fit CHỈ trên dữ liệu Train."""
        assert_no_leakage(FEATURE_INPUT_COLS)
        self._check_columns(X)

        self.encoder.fit(self._cat(X))
        self.encoded_cat_names = list(self.encoder.get_feature_names_out(CATEGORICAL_COLS))

        num = self._num(X)
        self.numeric_medians_ = num.median().fillna(0.0)          # cột trống hoàn toàn -> 0
        self.scaler.fit(num.fillna(self.numeric_medians_))

        self.feature_names = self.encoded_cat_names + NUMERIC_CONTINUOUS_COLS + PASSTHROUGH_BINARY_COLS
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("FlightDataPreprocessor phải được fit trước khi transform!")
        self._check_columns(X)

        cat_encoded = self.encoder.transform(self._cat(X))
        num_scaled = self.scaler.transform(self._num(X).fillna(self._impute_values()))
        bin_data = X[PASSTHROUGH_BINARY_COLS].fillna(0).astype(float).values
        return np.hstack([cat_encoded, num_scaled, bin_data])

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    def to_dataframe(self, X_transformed: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(X_transformed, columns=self.feature_names)

    # ---------- lưu / nạp artifact ----------
    def save(self, target_dir: Path = PROCESSED_DIR, extra_meta: Optional[dict] = None) -> None:
        """Lưu encoder.joblib, scaler.joblib, preprocessing_metadata.json (chỉ dùng đối tượng chuẩn scikit-learn)."""
        if not self.is_fitted:
            raise ValueError("Chưa fit, không có gì để lưu.")
        target_dir = Path(target_dir)
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
            "leakage_columns": LEAKAGE_COLUMNS,
            "numeric_medians": self._impute_values().to_dict(),
            "scaler_mean": dict(zip(NUMERIC_CONTINUOUS_COLS, map(float, self.scaler.mean_))),
            "scaler_scale": dict(zip(NUMERIC_CONTINUOUS_COLS, map(float, self.scaler.scale_))),
            "scaler_n_samples_seen": int(self.scaler.n_samples_seen_),
            "sklearn_version": sklearn.__version__,
        }
        if extra_meta:
            meta.update(extra_meta)
        with open(target_dir / "preprocessing_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, target_dir: Path = PROCESSED_DIR) -> "FlightDataPreprocessor":
        """Nạp bộ tiền xử lý từ artifact. Artifact cũ (chưa có trung vị) vẫn nạp được."""
        target_dir = Path(target_dir)
        enc = joblib.load(target_dir / "encoder.joblib")
        scl = joblib.load(target_dir / "scaler.joblib")
        meta, meta_file = {}, target_dir / "preprocessing_metadata.json"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
        saved_ver = meta.get("sklearn_version")
        if saved_ver and saved_ver != sklearn.__version__:
            warnings.warn(
                f"Artifact được tạo bằng scikit-learn {saved_ver}, máy hiện dùng {sklearn.__version__}. "
                f"Hãy cài đúng phiên bản ghi trong requirements-ml.txt.", RuntimeWarning)
        return cls(encoder=enc, scaler=scl, feature_names=meta.get("feature_names"),
                   encoded_cat_names=meta.get("encoded_cat_names"), numeric_medians=meta.get("numeric_medians"))


def build_and_save_encoded_dataset(
    source_df: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, FlightDataPreprocessor]:
    """Chia Train/Test theo ngày, fit preprocessor CHỈ trên Train, lưu artifact và các tệp train/test."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if source_df is None:
        features_path = PROCESSED_DIR / "flights_features.csv"
        if not features_path.exists():
            print("Chưa có flights_features.csv. Chạy Part 3 (python main.py features) trước...")
            from features.features import run_feature_pipeline
            source_df = run_feature_pipeline()
        else:
            source_df = pd.read_csv(features_path)

    clean_df = source_df[source_df["status_clean"].isin(LABEL_MAP.keys())].copy()

    train_raw, test_raw, cutoff_date = split_by_date(clean_df, train_ratio=0.80)
    print(f"Mốc cắt theo thời gian: {cutoff_date}")
    print(f"Train: {len(train_raw):,} chuyến | Test: {len(test_raw):,} chuyến")

    y_train = train_raw["status_clean"].map(LABEL_MAP).astype(int)
    y_test = test_raw["status_clean"].map(LABEL_MAP).astype(int)

    preprocessor = FlightDataPreprocessor()
    preprocessor.fit(train_raw)                                  # CHỈ fit trên Train
    X_train = preprocessor.to_dataframe(preprocessor.transform(train_raw))
    X_test = preprocessor.to_dataframe(preprocessor.transform(test_raw))

    preprocessor.save(PROCESSED_DIR, extra_meta={
        "cutoff_date": cutoff_date,
        "train_rows": len(train_raw),
        "test_rows": len(test_raw),
        "total_features": len(preprocessor.feature_names),
        "one_hot_columns": preprocessor.encoded_cat_names,
    })

    # Tệp cho Thành viên 5. Lưu ý: cột delay_min chỉ dành cho mô hình HỒI QUY (làm nhãn y),
    # tuyệt đối không đưa vào X khi huấn luyện phân loại.
    train_export = X_train.copy()
    train_export["target"] = y_train.values
    train_export["delay_min"] = train_raw["delay_min"].values
    train_export.to_csv(FILE_TRAIN, index=False)

    test_export = X_test.copy()
    test_export["target"] = y_test.values
    test_export["delay_min"] = test_raw["delay_min"].values
    test_export.to_csv(FILE_TEST, index=False)

    pd.concat([train_export, test_export], ignore_index=True).to_csv(FILE_PROCESSED, index=False)

    print(f"Đã lưu vào {PROCESSED_DIR}:")
    print(f"  - {FILE_TRAIN.name} ({train_export.shape[0]}x{train_export.shape[1]})")
    print(f"  - {FILE_TEST.name} ({test_export.shape[0]}x{test_export.shape[1]})")
    print(f"  - encoder.joblib, scaler.joblib, {FILE_METADATA.name} ({len(preprocessor.feature_names)} đặc trưng)")
    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    build_and_save_encoded_dataset()
