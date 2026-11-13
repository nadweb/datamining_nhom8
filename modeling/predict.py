# -*- coding: utf-8 -*-
"""Single, pre-flight inference interface shared by the web API and clients.

Artifacts are loaded once, without fitting or training during a request.  Their
feature schemas and available training metadata must agree before serving.
"""
import json
import math
import re
import sys
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import holidays
import joblib
import numpy as np
import pandas as pd
import sklearn

from config import AIRPORT_COORD, BASE_DIR, FILE_RULES, VN_AIRLINES
from features.features import extract_single_flight_features
from modeling.encoding import (
    CATEGORICAL_COLS, NUMERIC_CONTINUOUS_COLS, PASSTHROUGH_BINARY_COLS,
    FlightDataPreprocessor, LABEL_MAP, assert_no_leakage,
)

MODELS_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FILE_CLASSIFIER = MODELS_DIR / "best_classifier.joblib"
FILE_REGRESSOR = MODELS_DIR / "best_regressor.joblib"
FILE_KMEANS = MODELS_DIR / "kmeans_model.joblib"
FILE_PROFILES = MODELS_DIR / "cluster_profiles.json"
FILE_SUPERVISED_METRICS = MODELS_DIR / "supervised_metrics.json"
FILE_UNSUPERVISED_METRICS = MODELS_DIR / "unsupervised_metrics.json"

_CACHE: Dict[str, Any] = {}
_CACHE_LOCK = Lock()


class ArtifactError(RuntimeError):
    """A required model artifact is missing, invalid, or inconsistent."""


def _read_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as f:
            result = json.load(f)
        if not isinstance(result, dict):
            raise ValueError("expected a JSON object")
        return result
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"Không đọc được artifact {path.name}: {exc}") from exc


def _check_training_lineage(metadata: dict, supervised: dict, unsupervised: dict) -> None:
    """Reject explicit contradictions; matching metadata is not proof of provenance."""
    split = supervised.get("split", {})
    environment = unsupervised.get("environment", {})
    for key in ("train_rows", "test_rows", "cutoff_date"):
        left, right = metadata.get(key), split.get(key)
        if left is not None and right is not None and left != right:
            raise ArtifactError(
                f"Metadata tiền xử lý và model không khớp {key}: {left} != {right}. "
                "Cần bộ encoder, scaler và model cùng lần huấn luyện."
            )
    for meta_key, model_key in (("train_rows", "n_train"), ("test_rows", "n_test")):
        left, right = metadata.get(meta_key), environment.get(model_key)
        if left is not None and right is not None and left != right:
            raise ArtifactError(
                f"Metadata tiền xử lý và mô hình phân cụm không khớp {meta_key}: {left} != {right}."
            )
    # Older reports may store train_rows under metadata or at the top level.
    for source in (supervised.get("metadata", {}), supervised):
        left, right = metadata.get("train_rows"), source.get("train_rows")
        if left is not None and right is not None and left != right:
            raise ArtifactError(f"Metadata tiền xử lý và model không khớp train_rows: {left} != {right}.")
    # Follow the project policy in encoding.py: patch releases may differ,
    # while different major.minor versions must be reviewed before serving.
    runtime_mm = ".".join(str(sklearn.__version__).split(".")[:2])
    for saved in (metadata.get("sklearn_version"), supervised.get("environment", {}).get("sklearn"),
                  environment.get("sklearn")):
        if saved and ".".join(str(saved).split(".")[:2]) != runtime_mm:
            raise ArtifactError(f"Artifact dùng scikit-learn {saved}, môi trường hiện tại là {sklearn.__version__}.")


def _read_rules(path: Path) -> List[dict]:
    try:
        frame = pd.read_csv(path)
        required = {"Antecedent", "Consequent", "Confidence (%)", "Lift"}
        if not required.issubset(frame.columns):
            raise ValueError(f"thiếu cột: {sorted(required - set(frame.columns))}")
        frame = frame.replace([np.inf, -np.inf], np.nan).astype(object)
        return frame.where(pd.notna(frame), None).to_dict("records")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ArtifactError(f"Không đọc được luật kết hợp {path.name}: {exc}") from exc


def _check_feature_schema(preprocessor, models: dict) -> None:
    names = list(preprocessor.feature_names)
    if not names or len(names) != len(set(names)):
        raise ArtifactError("Danh sách đặc trưng tiền xử lý trống hoặc bị trùng.")
    try:
        encoded = list(preprocessor.encoder.get_feature_names_out(CATEGORICAL_COLS))
    except Exception as exc:
        raise ArtifactError(f"Không đọc được danh mục của encoder: {exc}") from exc
    if encoded != list(preprocessor.encoded_cat_names):
        raise ArtifactError("Danh mục encoder không khớp metadata tiền xử lý.")
    if names != encoded + NUMERIC_CONTINUOUS_COLS + PASSTHROUGH_BINARY_COLS:
        raise ArtifactError("Thứ tự đặc trưng trong metadata không khớp cấu hình tiền xử lý.")
    scaler_n = getattr(preprocessor.scaler, "n_features_in_", None)
    if scaler_n is not None and scaler_n != len(NUMERIC_CONTINUOUS_COLS):
        raise ArtifactError("Số biến số trong scaler không khớp cấu hình tiền xử lý.")
    try:
        assert_no_leakage(names)
    except ValueError as exc:
        raise ArtifactError(str(exc)) from exc
    for name, model in models.items():
        n_features = getattr(model, "n_features_in_", None)
        if n_features is not None and n_features != len(names):
            raise ArtifactError(f"Model {name} yêu cầu {n_features} đặc trưng; preprocessor có {len(names)}.")
        saved_names = getattr(model, "feature_names_in_", None)
        if saved_names is not None and list(saved_names) != names:
            raise ArtifactError(f"Thứ tự/tên đặc trưng của model {name} không khớp preprocessor.")
    classes = set(getattr(models["classifier"], "classes_", []))
    if not classes or not classes.issubset(set(LABEL_MAP) | set(LABEL_MAP.values())):
        raise ArtifactError("Model phân loại có nhãn lớp không hợp lệ.")


def _load_artifacts() -> dict:
    """Load an all-or-nothing cached artifact bundle; never refit or retrain."""
    if _CACHE:
        return _CACHE
    with _CACHE_LOCK:
        if _CACHE:
            return _CACHE
        paths = [PROCESSED_DIR / "encoder.joblib", PROCESSED_DIR / "scaler.joblib",
                 PROCESSED_DIR / "preprocessing_metadata.json", FILE_CLASSIFIER,
                 FILE_REGRESSOR, FILE_KMEANS, FILE_PROFILES, FILE_SUPERVISED_METRICS,
                 FILE_UNSUPERVISED_METRICS, FILE_RULES]
        missing = [p.name for p in paths if not p.is_file()]
        if missing:
            raise ArtifactError("Thiếu artifact: " + ", ".join(missing))
        metadata = _read_json(PROCESSED_DIR / "preprocessing_metadata.json")
        supervised, unsupervised = _read_json(FILE_SUPERVISED_METRICS), _read_json(FILE_UNSUPERVISED_METRICS)
        _check_training_lineage(metadata, supervised, unsupervised)
        try:
            preprocessor = FlightDataPreprocessor.load(PROCESSED_DIR)
            n_seen = getattr(preprocessor.scaler, "n_samples_seen_", None)
            train_rows = metadata.get("train_rows")
            if n_seen is not None and train_rows is not None and not np.all(np.asarray(n_seen) == train_rows):
                raise ArtifactError("Số dòng scaler đã fit không khớp metadata tiền xử lý.")
            models = {"classifier": joblib.load(FILE_CLASSIFIER),
                      "regressor": joblib.load(FILE_REGRESSOR), "kmeans": joblib.load(FILE_KMEANS)}
            _check_feature_schema(preprocessor, models)
            # A single prediction does not benefit from creating a joblib thread
            # pool for every request. This only changes the in-memory setting.
            if hasattr(models["classifier"], "n_jobs"):
                models["classifier"].n_jobs = 1
            profiles = _read_json(FILE_PROFILES)
            rules = _read_rules(FILE_RULES)
        except ArtifactError:
            raise
        except Exception as exc:
            raise ArtifactError(f"Không thể nạp artifact model: {exc}") from exc
        # Publish only after the complete bundle was validated successfully.
        _CACHE.update({"preprocessor": preprocessor, **models, "profiles": profiles, "rules": rules})
        return _CACHE


@lru_cache(maxsize=32)
def _vn_holidays(year: int):
    return holidays.country_holidays("VN", years=[year])


def _text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} phải là chuỗi không rỗng.")
    return value.strip()


def _number(value, name: str, *, integer: bool = False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} phải là số{' nguyên' if integer else ''} không âm.")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < 0 or (integer and value != int(value)):
        raise ValueError(f"{name} phải là số{' nguyên' if integer else ''} hữu hạn, không âm.")
    return int(value) if integer else float(value)


def _validate_inputs(airline_code, airport, direction, route_type, scheduled_time, flight_date,
                     precipitation, wind_gusts_10m, visibility, weather_code, airport_load,
                     origin, destination) -> dict:
    airline = _text(airline_code, "airline_code").upper()
    airport = _text(airport, "airport").upper()
    direction = _text(direction, "direction").upper()
    route_type = _text(route_type, "route_type").lower()
    origin, destination = _text(origin, "origin").upper(), _text(destination, "destination").upper()
    if airline not in VN_AIRLINES:
        raise ValueError("airline_code không thuộc danh sách hãng của dự án.")
    if airport not in AIRPORT_COORD:
        raise ValueError("airport phải là SGN, HAN, DAD, CXR hoặc PQC.")
    if direction not in {"ARRIVALS", "DEPARTURES"}:
        raise ValueError("direction phải là ARRIVALS hoặc DEPARTURES.")
    if route_type not in {"domestic", "international"}:
        raise ValueError("route_type phải là domestic hoặc international.")
    if not re.fullmatch(r"[A-Z]{3}", origin) or not re.fullmatch(r"[A-Z]{3}", destination):
        raise ValueError("origin và destination phải là mã IATA gồm 3 chữ cái.")
    scheduled_time, flight_date = _text(scheduled_time, "scheduled_time"), _text(flight_date, "flight_date")
    try:
        if not re.fullmatch(r"\d{2}:\d{2}", scheduled_time):
            raise ValueError
        datetime.strptime(scheduled_time, "%H:%M")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", flight_date):
            raise ValueError
        flight_day = date.fromisoformat(flight_date)
    except ValueError as exc:
        raise ValueError("flight_date phải là YYYY-MM-DD và scheduled_time phải là HH:MM hợp lệ.") from exc
    values = {"airline_code": airline, "airport": airport, "direction": direction,
              "route_type": route_type, "scheduled_time": scheduled_time, "flight_date": flight_date,
              "precipitation": _number(precipitation, "precipitation"),
              "wind_gusts_10m": _number(wind_gusts_10m, "wind_gusts_10m"),
              "visibility": _number(visibility, "visibility"),
              "weather_code": _number(weather_code, "weather_code", integer=True),
              "airport_load": _number(airport_load, "airport_load", integer=True),
              "origin": origin, "destination": destination}
    if values["weather_code"] > 99:
        raise ValueError("weather_code phải nằm trong khoảng 0–99.")
    values["flight_day"] = flight_day
    return values


def _rule_items(values: dict) -> set:
    """Build items using the same right-open bins as the mining pipeline.

    Load=low/mid/high is intentionally omitted: it depends on the airport's
    empirical rank in the dataset used to mine rules, which is not stored in
    the rules artifact. An unknown condition must never count as a match.
    """
    hour = int(values["scheduled_time"][:2])
    slot = ("00h-06h", "06h-12h", "12h-18h", "18h-24h")[hour // 6]
    rain, wind, vis = values["precipitation"], values["wind_gusts_10m"], values["visibility"]
    items = {f"Airline={values['airline_code']}", f"Airport={values['airport']}",
             f"Dir={values['direction'].lower()}", f"Route={values['route_type']}", f"Slot={slot}",
             f"Day={'weekend' if values['flight_day'].weekday() >= 5 else 'weekday'}"}
    # The mining pipeline uses 1e9 as an exclusive upper bin edge.
    if rain < 1e9:
        items.add(f"Rain={'none' if rain < 0.1 else 'light' if rain < 2.5 else 'heavy'}")
    if wind < 1e9:
        items.add(f"Gust={'<30' if wind < 30 else '30-50' if wind < 50 else '>50'}")
    if vis < 1e9:
        items.add(f"Vis={'<5km' if vis < 5000 else '>=5km'}")
    if values["flight_day"] in _vn_holidays(values["flight_day"].year):
        items.add("Holiday=yes")
    if values["weather_code"] >= 95:
        items.add("Storm=yes")
    return items


def match_association_rules(airline: str, airport: str, direction: str, rules: List[dict],
                            context: dict = None) -> List[dict]:
    """Match every antecedent item exactly; unobserved items never match."""
    items = context if context is not None else {f"Airline={airline.upper()}",
                                                  f"Airport={airport.upper()}", f"Dir={direction.lower()}"}
    matched = []
    for rule in rules:
        antecedent = rule.get("Antecedent")
        if not isinstance(antecedent, str):
            continue
        required = {item.strip() for item in antecedent.split(" & ") if item.strip()}
        if required and required.issubset(items):
            confidence = rule.get("Confidence (%)")
            matched.append({"antecedent": antecedent, "consequent": rule.get("Consequent"),
                            "confidence": confidence, "confidence_pct": confidence,
                            "support_pct": rule.get("Support (%)"), "lift": rule.get("Lift")})
            if len(matched) == 3:
                break
    return matched


def predict_risk(airline_code: str, airport: str, direction: str, route_type: str = "domestic",
                 scheduled_time: str = "12:00", flight_date: str = "2026-11-06",
                 precipitation: float = 0.0, wind_gusts_10m: float = 10.0,
                 visibility: float = 10000.0, weather_code: int = 0, airport_load: int = 20,
                 origin: str = "SGN", destination: str = "HAN") -> Dict[str, Any]:
    """Return the schema in docs/schema.md, plus existing frontend fields."""
    v = _validate_inputs(airline_code, airport, direction, route_type, scheduled_time, flight_date,
                         precipitation, wind_gusts_10m, visibility, weather_code, airport_load, origin, destination)
    artifacts = _load_artifacts()
    inputs = {k: value for k, value in v.items() if k != "flight_day"}
    frame = extract_single_flight_features(**inputs)
    # The shared single-flight helper currently hardcodes weekday/non-holiday.
    # Correct its derived flags here without changing another member's module.
    flight_day = v["flight_day"]
    frame["is_weekend_flag"] = int(flight_day.weekday() >= 5)
    frame["is_holiday_flag"] = int(flight_day in _vn_holidays(flight_day.year))
    prep = artifacts["preprocessor"]
    classifier = artifacts["classifier"]
    try:
        X = prep.to_dataframe(prep.transform(frame))
        class_probs = classifier.predict_proba(X)[0]
        delay = float(artifacts["regressor"].predict(X)[0])
        cluster_id = int(artifacts["kmeans"].predict(X)[0])
    except Exception as exc:
        raise ArtifactError(f"Không thể suy luận với bộ artifact hiện tại: {exc}") from exc
    if len(class_probs) != len(classifier.classes_) or not np.isfinite(class_probs).all():
        raise ArtifactError("Model trả về xác suất không hợp lệ.")
    probs = {name: 0.0 for name in LABEL_MAP}
    for label, prob in zip(classifier.classes_, class_probs):
        name = label if isinstance(label, str) else next((n for n, i in LABEL_MAP.items() if i == label), None)
        if name not in probs or not 0 <= float(prob) <= 1:
            raise ArtifactError("Model trả về lớp hoặc xác suất không hợp lệ.")
        probs[name] = float(prob)
    if not math.isclose(sum(probs.values()), 1.0, abs_tol=1e-6):
        raise ArtifactError("Tổng xác suất model không bằng 1.")
    prediction = max(probs, key=probs.get)
    risk_score = round(probs["delayed"] + probs["cancelled"], 4)
    risk_level = "HIGH" if risk_score >= 0.45 else "MEDIUM" if risk_score >= 0.25 else "LOW"

    if not math.isfinite(delay):
        raise ArtifactError("Model hồi quy trả về số phút không hợp lệ.")
    profile = artifacts["profiles"].get(str(cluster_id))
    if (not isinstance(profile, dict) or "name" not in profile or "delay_rate_pct" not in profile
            or profile.get("cluster_id") != cluster_id):
        raise ArtifactError(f"Không có hồ sơ đầy đủ cho cụm {cluster_id}.")
    cluster = dict(profile)
    cluster["cluster_id"] = cluster_id
    cluster["cluster_name"] = profile["name"]
    try:
        historical_risk = float(profile["delay_rate_pct"]) + float(profile.get("cancellation_rate_pct", 0))
    except (ValueError, TypeError) as exc:
        raise ArtifactError(f"Tỷ lệ rủi ro của cụm {cluster_id} không hợp lệ.") from exc
    if not math.isfinite(historical_risk) or not 0 <= historical_risk <= 100:
        raise ArtifactError(f"Tỷ lệ rủi ro của cụm {cluster_id} không hợp lệ.")
    cluster["cluster_risk_avg"] = f"{historical_risk:.2f}%"
    return {
        "prediction": prediction, "risk_score": risk_score, "risk_level": risk_level,
        "estimated_delay_min": max(0, int(round(delay))),
        "probabilities": {name: round(prob, 4) for name, prob in probs.items()},
        "cluster_profile": cluster,
        "matched_rules": match_association_rules(v["airline_code"], v["airport"], v["direction"],
                                                  artifacts["rules"], _rule_items(v)),
        "flight_summary": {"airline": v["airline_code"], "route": f"{v['origin']} -> {v['destination']}",
                           "airport": v["airport"], "direction": v["direction"],
                           "scheduled_time": v["scheduled_time"], "flight_date": v["flight_date"],
                           "weather": {"precipitation_mm": v["precipitation"], "wind_kmh": v["wind_gusts_10m"],
                                       "visibility_m": v["visibility"], "weather_code": v["weather_code"]},
                           "airport_load": v["airport_load"]},
    }


if __name__ == "__main__":
    try:
        print(json.dumps(predict_risk(airline_code="VJ", airport="SGN", direction="DEPARTURES",
                                      scheduled_time="19:30", flight_date="2026-11-06"),
                         indent=2, ensure_ascii=False))
    except ArtifactError as exc:
        print(f"Không thể dự báo: {exc}", file=sys.stderr)
        sys.exit(1)
