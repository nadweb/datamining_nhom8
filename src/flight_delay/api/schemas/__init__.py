# -*- coding: utf-8 -*-
"""Pydantic Schemas for API Contracts."""
from flight_delay.api.schemas.predict import HealthResponse, PredictRequest, PredictResponse

__all__ = ["PredictRequest", "PredictResponse", "HealthResponse"]
