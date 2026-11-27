# -*- coding: utf-8 -*-
"""[STEP 7] Public Pre-Flight Inference API (`predict_risk`)."""
import time
from typing import Any, Dict, Optional

import numpy as np

from flight_delay.common.constants import PRIMARY_HORIZON_HOURS
from flight_delay.data_mining.feature_engineering.builder import FeatureBuilder
from flight_delay.inference.feature_validation import validate_flight_input
from flight_delay.inference.model_loader import ModelBundle, load_production_artifacts
from flight_delay.inference.risk_policy import classify_risk_tier, explain_operational_factors


def predict_risk(
    flight_data: Dict[str, Any],
    prediction_time: Optional[str] = None,
    bundle: Optional[ModelBundle] = None,
) -> Dict[str, Any]:
    """Run fail-closed, leakage-free pre-flight risk & delay prediction."""
    t0 = time.perf_counter()
    art = bundle or load_production_artifacts()
    horizon_h = int(art.cls_meta.get("prediction_horizon_hours", PRIMARY_HORIZON_HOURS))
    clean_in = validate_flight_input(flight_data, prediction_time=prediction_time, horizon_hours=horizon_h)

    fb = FeatureBuilder(horizon_hours=horizon_h, history_store=art.history_store)
    gold_df = fb.build_single(**clean_in)
    X_mat = art.preprocessor.to_dataframe(art.preprocessor.transform(gold_df))

    probs = np.asarray(art.cls_model.predict_proba(X_mat)[0], dtype=np.float64)
    probs = probs / (float(probs.sum()) or 1.0)
    p_ontime = float(probs[0])
    p_delayed = float(probs[1])
    p_cancelled = float(probs[2]) if len(probs) > 2 else 0.0
    p_disrupted = p_delayed + p_cancelled

    class_names = list(art.label_mapping.get("class_names", ["on_time", "delayed", "cancelled"]))
    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx] if pred_idx < len(class_names) else "on_time"

    est_delay = max(0.0, float(art.reg_model.predict(X_mat)[0]))
    tier = classify_risk_tier(p_disrupted)
    row_s = gold_df.iloc[0]

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        "predicted_class": pred_class,
        "probabilities": {
            "on_time": round(p_ontime, 4),
            "delayed": round(p_delayed, 4),
            "cancelled": round(p_cancelled, 4),
            "disrupted": round(p_disrupted, 4),
        },
        "probabilities_pct": {
            "on_time": round(p_ontime * 100.0, 1),
            "delayed": round(p_delayed * 100.0, 1),
            "cancelled": round(p_cancelled * 100.0, 1),
            "disrupted": round(p_disrupted * 100.0, 1),
        },
        "risk_tier": tier["risk_tier"],
        "risk_level": tier["risk_level"],
        "risk_color": tier["risk_color"],
        "recommendation": tier["recommendation"],
        "predicted_delay_minutes": round(est_delay, 1),
        "estimated_delay_minutes": round(est_delay, 1),
        "model_version": f"{art.cls_meta.get('model_name')}:{art.cls_meta.get('version')}",
        "regressor_version": f"{art.reg_meta.get('model_name')}:{art.reg_meta.get('version')}",
        "dataset_version": art.cls_meta.get("dataset_version"),
        "prediction_time": clean_in["prediction_time"],
        "horizon_hours": horizon_h,
        "latency_ms": latency_ms,
        "factors": explain_operational_factors(row_s),
        "causal_rates": {
            "historical_airline_delay_rate": round(float(row_s["historical_airline_delay_rate"]), 4),
            "historical_airport_delay_rate": round(float(row_s["historical_airport_delay_rate"]), 4),
            "weather_hazard_score": round(float(row_s["weather_hazard_score"]), 4),
        },
    }
