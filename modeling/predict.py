# -*- coding: utf-8 -*-
"""Prediction & Inference Engine (Contracted API for Web & Clients)
Implements predict_risk() conforming strictly to docs/schema.md.
Combines:
1. Supervised Classification (Probabilities for on_time, delayed, cancelled)
2. Supervised Regression (Estimated delay minutes)
3. Unsupervised K-Means Cluster Assignment (Operational Risk Profile)
4. Association Rule Matching (Explainable insights based on FP-Growth rules)
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd

from config import BASE_DIR, FILE_RULES
from features.features import extract_single_flight_features
from modeling.encoding import FlightDataPreprocessor

MODELS_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

FILE_CLASSIFIER = MODELS_DIR / "best_classifier.joblib"
FILE_REGRESSOR = MODELS_DIR / "best_regressor.joblib"
FILE_KMEANS = MODELS_DIR / "kmeans_model.joblib"
FILE_PREPROCESSOR = PROCESSED_DIR / "preprocessor.joblib"
FILE_PROFILES = MODELS_DIR / "cluster_profiles.json"

_CACHE = {}


def _load_artifacts():
    """Load model artifacts with in-memory caching."""
    if not _CACHE:
        # Preprocessor
        if (PROCESSED_DIR / "encoder.joblib").exists() and (PROCESSED_DIR / "scaler.joblib").exists():
            _CACHE["preprocessor"] = FlightDataPreprocessor.load(PROCESSED_DIR)
        else:
            from modeling.encoding import build_and_save_encoded_dataset
            _, _, _, _, prep = build_and_save_encoded_dataset()
            _CACHE["preprocessor"] = prep

        # Classifier
        if FILE_CLASSIFIER.exists():
            _CACHE["classifier"] = joblib.load(FILE_CLASSIFIER)
        else:
            from modeling.train import run_supervised_training
            run_supervised_training()
            _CACHE["classifier"] = joblib.load(FILE_CLASSIFIER)

        # Regressor
        if FILE_REGRESSOR.exists():
            _CACHE["regressor"] = joblib.load(FILE_REGRESSOR)

        # KMeans
        if FILE_KMEANS.exists():
            _CACHE["kmeans"] = joblib.load(FILE_KMEANS)
        else:
            from modeling.unsupervised import run_unsupervised_pipeline
            run_unsupervised_pipeline()
            _CACHE["kmeans"] = joblib.load(FILE_KMEANS)

        # Profiles
        if FILE_PROFILES.exists():
            with open(FILE_PROFILES, "r", encoding="utf-8") as f:
                _CACHE["profiles"] = json.load(f)
        else:
            _CACHE["profiles"] = {}

        # Association Rules
        if FILE_RULES.exists():
            try:
                _CACHE["rules"] = pd.read_csv(FILE_RULES).to_dict(orient="records")
            except Exception:
                _CACHE["rules"] = []
        else:
            _CACHE["rules"] = []

    return _CACHE


def match_association_rules(airline: str, airport: str, direction: str, rules: List[dict]) -> List[dict]:
    """Find top association rules matching current flight characteristics."""
    matched = []
    airline_str = f"Airline={airline.upper()}"
    airport_str = f"Airport={airport.upper()}"
    dir_str = f"Dir={direction.lower()}"

    for r in rules:
        ante = r.get("Antecedent", "")
        # If any matching item is in antecedent
        if airline_str in ante or airport_str in ante or dir_str in ante:
            matched.append({
                "antecedent": r.get("Antecedent"),
                "consequent": r.get("Consequent"),
                "support_pct": r.get("Support (%)"),
                "confidence_pct": r.get("Confidence (%)"),
                "lift": r.get("Lift"),
            })
            if len(matched) >= 3:
                break
    return matched


def predict_risk(
    airline_code: str,
    airport: str,
    direction: str,
    route_type: str = "domestic",
    scheduled_time: str = "12:00",
    flight_date: str = "2026-11-06",
    precipitation: float = 0.0,
    wind_gusts_10m: float = 10.0,
    visibility: float = 10000.0,
    weather_code: int = 0,
    airport_load: int = 20,
    origin: str = "SGN",
    destination: str = "HAN"
) -> Dict[str, Any]:
    """Contracted Flight Delay Risk Prediction Interface.

    Returns structured dictionary with:
    - prediction: 'on_time' | 'delayed' | 'cancelled'
    - risk_score: float in [0.0, 1.0]
    - risk_level: 'LOW' | 'MEDIUM' | 'HIGH'
    - estimated_delay_min: int (predicted delay duration in minutes)
    - probabilities: dict of class probabilities
    - cluster_profile: unsupervised operational cluster profile
    - matched_rules: relevant mined association rules
    """
    artifacts = _load_artifacts()
    preprocessor = artifacts["preprocessor"]
    classifier = artifacts["classifier"]
    regressor = artifacts.get("regressor")
    kmeans = artifacts.get("kmeans")
    profiles = artifacts.get("profiles", {})
    rules = artifacts.get("rules", [])

    # 1. Feature Engineering for single flight
    df_single = extract_single_flight_features(
        airline_code=airline_code,
        airport=airport,
        direction=direction,
        route_type=route_type,
        scheduled_time=scheduled_time,
        flight_date=flight_date,
        precipitation=precipitation,
        wind_gusts_10m=wind_gusts_10m,
        visibility=visibility,
        weather_code=weather_code,
        airport_load=airport_load,
        origin=origin,
        destination=destination,
    )

    # 2. Transform into model matrix
    X_mat = preprocessor.transform(df_single)
    X_df = preprocessor.to_dataframe(X_mat)

    # 3. Supervised Classification
    class_probs = classifier.predict_proba(X_df)[0]
    classes = ["on_time", "delayed", "cancelled"]
    prob_dict = {
        cls_name: round(float(class_probs[i]), 4) if i < len(class_probs) else 0.0
        for i, cls_name in enumerate(classes)
    }

    pred_idx = int(np.argmax(class_probs))
    pred_label = classes[pred_idx] if pred_idx < len(classes) else "on_time"

    # Risk score is combined probability of delay or cancellation
    risk_score = round(float(prob_dict.get("delayed", 0.0) + prob_dict.get("cancelled", 0.0)), 4)
    if risk_score >= 0.45:
        risk_level = "HIGH"
    elif risk_score >= 0.25:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # 4. Supervised Regression (Delay Minutes)
    if regressor is not None:
        delay_est = float(regressor.predict(X_df)[0])
        # If predicted on-time, keep delay estimated at 0 or non-negative
        delay_est_min = max(0, int(round(delay_est)))
    else:
        delay_est_min = 15 if pred_label == "delayed" else 0

    # 5. Unsupervised K-Means Cluster Assignment
    if kmeans is not None:
        cid = int(kmeans.predict(X_df)[0])
        cluster_info = profiles.get(str(cid), {
            "cluster_id": cid,
            "name": f"Cluster {cid}",
            "risk_tier": "UNKNOWN",
        })
    else:
        cluster_info = {"cluster_id": 0, "name": "Default Cluster", "risk_tier": "LOW"}

    # 6. Association Rules
    matched_rules = match_association_rules(airline_code, airport, direction, rules)

    return {
        "prediction": pred_label,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "estimated_delay_min": delay_est_min,
        "probabilities": prob_dict,
        "cluster_profile": cluster_info,
        "matched_rules": matched_rules,
        "flight_summary": {
            "airline": airline_code.upper(),
            "route": f"{origin.upper()} -> {destination.upper()}",
            "airport": airport.upper(),
            "direction": direction.upper(),
            "scheduled_time": scheduled_time,
            "flight_date": flight_date,
            "weather": {
                "precipitation_mm": precipitation,
                "wind_kmh": wind_gusts_10m,
                "visibility_m": visibility,
                "weather_code": weather_code,
            },
            "airport_load": airport_load,
        }
    }


if __name__ == "__main__":
    print("Testing predict_risk() interface with sample flight:")
    res = predict_risk(
        airline_code="VJ",
        airport="SGN",
        direction="DEPARTURES",
        scheduled_time="19:30",
        flight_date="2026-11-06",
        precipitation=3.5,
        wind_gusts_10m=35.0,
        visibility=4000.0,
        weather_code=95,
        airport_load=34,
        origin="SGN",
        destination="HAN",
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))
