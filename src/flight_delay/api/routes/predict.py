# -*- coding: utf-8 -*-
"""[STEP 7] Real-Time Prediction, FIDS Feed & Holdout Audit Routes."""
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException

from flight_delay.api.dependencies import get_bundle, safe_read_json
from flight_delay.api.schemas.predict import PredictRequest, PredictResponse
from flight_delay.common.constants import (
    ENCODERS_DIR,
    FILE_GOLD,
    FILE_RULES,
    PROCESSED_DIR,
    REPORTS_DIR,
    ROOT_DIR,
)
from flight_delay.common.exceptions import (
    ApprovalError,
    ArtifactError,
    InvalidPredictionTimeError,
    LeakageError,
    SchemaValidationError,
)
from flight_delay.data_mining.feature_engineering.builder import FeatureBuilder
from flight_delay.inference.predict import predict_risk
from flight_delay.modeling.unsupervised.clustering import CLUSTER_FEATURES

router = APIRouter(prefix="/api", tags=["Inference & Operational Feed"])


def _match_cluster_context(payload: dict, bundle) -> dict:
    try:
        scaler = joblib.load(ENCODERS_DIR / "cluster_scaler.joblib")
        km = joblib.load(ROOT_DIR / "models" / "kmeans_model.joblib")
        fb = FeatureBuilder(horizon_hours=bundle.cls_meta.get("prediction_horizon_hours", 3), history_store=bundle.history_store)
        gold_1row = fb.build_single(**payload)
        avail = [c for c in CLUSTER_FEATURES if c in gold_1row.columns]
        cid = int(km.predict(scaler.transform(gold_1row[avail].fillna(0.0)))[0])
        unsup = safe_read_json(REPORTS_DIR / "experiments" / "unsupervised_results.json")
        for prof in unsup.get("kmeans", {}).get("cluster_profiles", []):
            if int(prof.get("cluster_id", -1)) == cid:
                return {
                    "cluster_id": cid,
                    "name": prof.get("name"),
                    "cluster_name": prof.get("name"),
                    "delay_rate_pct": prof.get("delay_rate_pct"),
                    "disruption_rate_pct": prof.get("disruption_rate_pct"),
                    "recommended_action": prof.get("recommended_action"),
                }
        return {"cluster_id": cid, "cluster_name": f"Cụm #{cid}", "delay_rate_pct": None}
    except Exception:
        return {"cluster_id": 0, "cluster_name": "Ngữ cảnh Vận hành Tiêu chuẩn", "delay_rate_pct": None}


def _match_association_rules(payload: dict) -> list:
    if not FILE_RULES.exists():
        return []
    rules_df = pd.read_csv(FILE_RULES)
    if rules_df.empty:
        return []

    hr = int(str(payload.get("scheduled_time", "12:00")).split(":")[0])
    slot = "00h-06h" if hr < 6 else "06h-12h" if hr < 12 else "12h-18h" if hr < 18 else "18h-24h"
    is_peak = 6 <= hr <= 9 or 17 <= hr <= 20
    active_tokens = {
        f"Airline={payload.get('airline_code', '').upper()}",
        f"Airport={payload.get('airport', '').upper()}",
        f"Slot={slot}",
    }
    if is_peak:
        active_tokens.add("Peak=Yes")
    if float(payload.get("airport_load", 20)) > 28:
        active_tokens.add("HeavyTraffic=Yes")
    if float(payload.get("precipitation", 0.0)) > 0.1:
        active_tokens.add("Rain=Yes")
    if float(payload.get("wind_gusts_10m", 10.0)) > 30.0:
        active_tokens.add("HighWind=Yes")
    if str(payload.get("route_type", "domestic")).lower() == "domestic":
        active_tokens.add("Domestic=Yes")

    matched = []
    for _, r in rules_df.iterrows():
        ants = [tok.strip() for tok in str(r.get("antecedents", "")).split("+") if tok.strip()]
        if ants and all(t in active_tokens for t in ants):
            matched.append({
                "antecedent": str(r.get("antecedents")),
                "consequent": str(r.get("consequent")),
                "support_pct": float(r.get("support_pct", 0.0)),
                "confidence_pct": float(r.get("confidence_pct", 0.0)),
                "lift": float(r.get("lift", 1.0)),
                "holdout_validated": bool(r.get("holdout_validated", True)),
            })
    matched.sort(key=lambda x: (x["lift"], x["confidence_pct"]), reverse=True)
    return matched[:6]


@router.post("/predict", response_model=PredictResponse)
def api_predict(req: PredictRequest):
    raw_payload = req.model_dump()
    try:
        bundle = get_bundle()
        res = predict_risk(raw_payload, prediction_time=req.prediction_time, bundle=bundle)
    except (InvalidPredictionTimeError, SchemaValidationError, LeakageError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (ApprovalError, ArtifactError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    clean_for_ctx = {k: v for k, v in raw_payload.items() if v is not None}
    clean_for_ctx.setdefault("flight_date", res["prediction_time"].split(" ")[0])
    clean_for_ctx.setdefault("origin", req.airport if req.direction == "DEPARTURES" else "HAN")
    clean_for_ctx.setdefault("destination", "HAN" if req.direction == "DEPARTURES" else req.airport)

    cluster_prof = _match_cluster_context(clean_for_ctx, bundle)
    matched_rules = _match_association_rules(clean_for_ctx)

    return {
        **res,
        "prediction": res["predicted_class"],
        "risk_level": res["risk_tier"],
        "risk_score": res["probabilities"]["disrupted"],
        "estimated_delay_min": res["predicted_delay_minutes"],
        "cluster_profile": cluster_prof,
        "matched_rules": matched_rules,
    }


@router.get("/fids-flights")
def api_fids_flights():
    if not FILE_GOLD.exists():
        return []
    df = pd.read_csv(FILE_GOLD, low_memory=False)
    sample = df[df["airport"].isin(["SGN", "HAN", "DAD", "CXR", "PQC"])].tail(80).iloc[::-1]
    rows = []
    for _, r in sample.iterrows():
        rows.append({
            "flight_no": str(r.get("flight_no", "")),
            "airline": str(r.get("airline_code", "")),
            "airline_code": str(r.get("airline_code", "")),
            "airport": str(r.get("airport", "")),
            "origin": str(r.get("origin", "")),
            "destination": str(r.get("destination", "")),
            "direction": str(r.get("direction", "DEPARTURES")),
            "route_type": str(r.get("route_type", "domestic")),
            "flight_date": str(r.get("flight_date", "")),
            "scheduled_time": str(r.get("scheduled_time", ""))[:5],
            "status": str(r.get("status_clean", "")),
            "precipitation": round(float(r.get("precipitation", 0.0) or 0.0), 1),
            "wind_gusts_10m": round(float(r.get("wind_gusts_10m", 10.0) or 10.0), 1),
            "visibility": round(float(r.get("visibility", 10000.0) or 10000.0), 0),
            "weather_code": int(float(r.get("weather_code", 0) or 0)),
            "airport_load": int(float(r.get("airport_load", 20) or 20)),
        })
    return rows


@router.get("/audit-flights")
def api_audit_flights():
    audit_path = PROCESSED_DIR / "audit_benchmark_flights.json"
    if audit_path.exists():
        return safe_read_json(audit_path, [])
    return []
