# -*- coding: utf-8 -*-
"""FastAPI API Route Registration Package."""
from flight_delay.api.routes import drift, eda, model, predict

__all__ = ["predict", "eda", "model", "drift"]
