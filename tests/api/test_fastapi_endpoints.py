# -*- coding: utf-8 -*-
"""[API] FastAPI Route Contracts, Predict 3-Class Schema & Fail-Closed Guardrails."""
import pytest
from fastapi import HTTPException

from flight_delay.api.app import app
from flight_delay.api.routes.drift import api_drift, api_manifest
from flight_delay.api.routes.eda import api_clusters, api_eda, api_rules
from flight_delay.api.routes.model import (
    api_backtest,
    api_evaluation,
    api_experiments,
    api_governance,
    api_stats,
)
from flight_delay.api.routes.predict import api_audit_flights, api_fids_flights, api_predict
from flight_delay.api.schemas.predict import PredictRequest


def _find_route(path: str):
    for route in app.routes:
        if getattr(route, "path", None) == path:
            return route.endpoint
    raise KeyError(f"Missing FastAPI route {path!r}")


def test_health_and_html_pages():
    health_fn = _find_route("/health")
    h = health_fn()
    assert h.status == "ok" and h.artifact_integrity is True

    dash = _find_route("/")()
    assert dash.status_code == 200
    assert b"FlightRisk VN" in dash.body

    pres = _find_route("/presentation")()
    assert pres.status_code == 200
    assert b"FlightRisk VN 2.0" in pres.body


def test_predict_success_and_guardrail_rejections(sample_flight_payload):
    req = PredictRequest(**sample_flight_payload)
    data = api_predict(req)
    assert data["prediction"] in ("on_time", "delayed", "cancelled")
    assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    probs = data["probabilities"]
    assert abs(probs["on_time"] + probs["delayed"] + probs["cancelled"] - 1.0) < 0.01
    assert abs(probs["disrupted"] - (probs["delayed"] + probs["cancelled"])) < 0.01
    assert isinstance(data["cluster_profile"], dict)
    assert isinstance(data["matched_rules"], list)

    # Future prediction_time (> scheduled_dt) fails closed with HTTP 422
    bad_req = PredictRequest(**{**sample_flight_payload, "prediction_time": "2026-11-06 20:00:00"})
    with pytest.raises(HTTPException) as exc_info:
        api_predict(bad_req)
    assert exc_info.value.status_code == 422


def test_analytics_and_governance_endpoints():
    assert "supervised" in api_stats()
    assert "eda" in api_eda()
    assert "backtest_by_airport" in api_backtest()
    assert "profiles" in api_clusters()
    rules = api_rules()
    assert rules["total_rules"] >= 15
    assert "overall_status" in api_drift()
    assert "model_registry" in api_manifest()
    assert "data_lineage" in api_governance()
    assert api_experiments()["total_latest_runs"] >= 5
    assert len(api_fids_flights()) >= 10
    assert len(api_audit_flights()) >= 20
    assert "classification_3class" in api_evaluation()
