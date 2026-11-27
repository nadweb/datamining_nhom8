# -*- coding: utf-8 -*-
"""Shared Pytest Fixtures for Unit, Leakage, Parity, API & E2E Suites."""
import shutil
from pathlib import Path

import pytest


def find_chromium_binary():
    candidates = [
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    return next((str(Path(p)) for p in candidates if p and Path(p).is_file()), None)


@pytest.fixture
def sample_flight_payload() -> dict:
    return {
        "airline_code": "VJ",
        "airport": "SGN",
        "direction": "DEPARTURES",
        "route_type": "domestic",
        "scheduled_time": "19:30",
        "flight_date": "2026-11-06",
        "origin": "SGN",
        "destination": "HAN",
        "precipitation": 12.5,
        "wind_gusts_10m": 42.0,
        "visibility": 3500.0,
        "weather_code": 95,
        "airport_load": 34,
        "prediction_time": "2026-11-06 16:30:00",
    }
