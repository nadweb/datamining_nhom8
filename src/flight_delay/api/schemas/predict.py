# -*- coding: utf-8 -*-
"""[STEP 7] Pydantic Request & Response Schemas for Prediction API."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    airline_code: str = Field(..., min_length=2, max_length=4, examples=["VJ"])
    airport: str = Field(..., min_length=3, max_length=4, examples=["SGN"])
    direction: str = Field(default="DEPARTURES", examples=["DEPARTURES", "ARRIVALS"])
    route_type: str = Field(default="domestic", examples=["domestic", "international"])
    scheduled_time: str = Field(..., examples=["19:30"])
    flight_date: Optional[str] = Field(default=None, examples=["2026-11-06"])
    origin: Optional[str] = Field(default=None, examples=["SGN"])
    destination: Optional[str] = Field(default=None, examples=["HAN"])
    precipitation: float = Field(default=0.0, ge=0.0, le=500.0)
    wind_gusts_10m: float = Field(default=10.0, ge=0.0, le=250.0)
    visibility: float = Field(default=10000.0, ge=0.0, le=100000.0)
    weather_code: int = Field(default=0, ge=0, le=99)
    airport_load: int = Field(default=20, ge=1, le=300)
    prediction_time: Optional[str] = Field(default=None, examples=["2026-11-06 16:30:00"])


class PredictResponse(BaseModel):
    prediction: str
    predicted_class: str
    probabilities: Dict[str, float]
    probabilities_pct: Dict[str, float]
    risk_tier: str
    risk_level: str
    risk_color: str
    risk_score: float
    recommendation: str
    predicted_delay_minutes: float
    estimated_delay_min: float
    model_version: str
    regressor_version: str
    dataset_version: Optional[str]
    prediction_time: str
    horizon_hours: int
    latency_ms: float
    factors: List[str]
    causal_rates: Dict[str, float]
    cluster_profile: Dict[str, Any]
    matched_rules: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    approved_classifier: Optional[str] = None
    approved_regressor: Optional[str] = None
    dataset_version: Optional[str] = None
    artifact_integrity: bool = False
    error: Optional[str] = None
