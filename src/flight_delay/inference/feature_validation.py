# -*- coding: utf-8 -*-
"""[STEP 7] Fail-Closed Input Schema, Leakage & Prediction Horizon Validation."""
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import pandas as pd

from flight_delay.common.constants import FORBIDDEN_POST_FLIGHT_COLUMNS, PRIMARY_HORIZON_HOURS
from flight_delay.common.exceptions import InvalidPredictionTimeError, LeakageError, SchemaValidationError


def validate_flight_input(
    flight_data: Dict[str, Any],
    prediction_time: Optional[str] = None,
    horizon_hours: int = PRIMARY_HORIZON_HOURS,
) -> Dict[str, Any]:
    """Validate and normalize inference payload, enforcing prediction_time <= scheduled_dt and zero leakage."""
    if not isinstance(flight_data, dict):
        raise SchemaValidationError("flight_data must be a JSON object / dict.")

    leaked = sorted(set(flight_data.keys()) & set(FORBIDDEN_POST_FLIGHT_COLUMNS))
    if leaked:
        raise LeakageError(f"Post-flight columns are forbidden in inference payload: {leaked}")

    airline = str(flight_data.get("airline_code", "")).strip().upper()
    airport = str(flight_data.get("airport", "")).strip().upper()
    sched_time = str(flight_data.get("scheduled_time", "")).strip()
    if not airline or not airport or not sched_time:
        raise SchemaValidationError("airline_code, airport, and scheduled_time are required.")

    flight_date = str(flight_data.get("flight_date") or datetime.now().strftime("%Y-%m-%d")).strip()
    sched_dt = pd.to_datetime(f"{flight_date} {sched_time}", errors="coerce")
    if pd.isna(sched_dt):
        raise InvalidPredictionTimeError(f"Invalid flight_date/scheduled_time: {flight_date!r} {sched_time!r}")

    raw_ptime = prediction_time if prediction_time is not None else flight_data.get("prediction_time")
    if raw_ptime is None or str(raw_ptime).strip() == "":
        pred_dt = sched_dt - timedelta(hours=int(horizon_hours))
    else:
        pred_dt = pd.to_datetime(raw_ptime, errors="coerce")
        if pd.isna(pred_dt):
            raise InvalidPredictionTimeError(f"Unparseable prediction_time: {raw_ptime!r}")
        if pred_dt > sched_dt:
            raise InvalidPredictionTimeError(
                f"Invalid prediction_time {pred_dt} > scheduled departure {sched_dt} (future leakage)."
            )

    def _nonneg_float(key: str, default: float, max_val: float) -> float:
        try:
            v = float(flight_data.get(key, default))
        except (TypeError, ValueError) as exc:
            raise SchemaValidationError(f"Field {key!r} must be numeric.") from exc
        if pd.isna(v) or v < 0.0 or v > max_val:
            raise SchemaValidationError(f"Field {key!r}={v} out of valid bounds [0, {max_val}].")
        return v

    precip = _nonneg_float("precipitation", 0.0, 500.0)
    wind = _nonneg_float("wind_gusts_10m", 10.0, 250.0)
    vis = _nonneg_float("visibility", 10000.0, 100000.0)
    load = int(_nonneg_float("airport_load", 20.0, 300.0))
    try:
        wmo = int(flight_data.get("weather_code", 0))
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError("weather_code must be an integer WMO code.") from exc
    if wmo < 0 or wmo > 99:
        raise SchemaValidationError(f"weather_code={wmo} out of WMO [0, 99] bounds.")

    direction = str(flight_data.get("direction", "DEPARTURES")).strip().upper()
    if direction not in ("DEPARTURES", "ARRIVALS"):
        raise SchemaValidationError(f"Invalid direction {direction!r}; expected DEPARTURES or ARRIVALS.")

    route_type = str(flight_data.get("route_type", "domestic")).strip().lower()
    if route_type not in ("domestic", "international"):
        raise SchemaValidationError(f"Invalid route_type {route_type!r}; expected domestic or international.")

    origin = str(flight_data.get("origin") or (airport if direction == "DEPARTURES" else "HAN")).strip().upper()
    destination = str(flight_data.get("destination") or ("HAN" if direction == "DEPARTURES" else airport)).strip().upper()

    return {
        "airline_code": airline,
        "airport": airport,
        "direction": direction,
        "route_type": route_type,
        "scheduled_time": sched_dt.strftime("%H:%M"),
        "flight_date": sched_dt.strftime("%Y-%m-%d"),
        "origin": origin,
        "destination": destination,
        "precipitation": precip,
        "wind_gusts_10m": wind,
        "visibility": vis,
        "weather_code": wmo,
        "airport_load": load,
        "prediction_time": pred_dt.strftime("%Y-%m-%d %H:%M:%S"),
    }
