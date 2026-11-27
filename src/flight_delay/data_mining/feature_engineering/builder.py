# -*- coding: utf-8 -*-
"""[STEP 3] Single Source-of-Truth FeatureBuilder Guaranteeing Training-Serving Parity."""
from datetime import timedelta
from typing import Optional

import holidays
import pandas as pd

from flight_delay.common.constants import PRIMARY_HORIZON_HOURS, VN_AIRPORTS
from flight_delay.common.exceptions import InvalidPredictionTimeError
from flight_delay.data_mining.feature_engineering.airline import compute_airline_features
from flight_delay.data_mining.feature_engineering.airport import compute_airport_features
from flight_delay.data_mining.feature_engineering.historical import CausalHistoryStore
from flight_delay.data_mining.feature_engineering.operational import compute_operational_features
from flight_delay.data_mining.feature_engineering.registry import FEATURE_REGISTRY, assert_no_leakage
from flight_delay.data_mining.feature_engineering.temporal import compute_temporal_features
from flight_delay.data_mining.feature_engineering.weather import compute_weather_features


def _compute_scheduled_dt(df: pd.DataFrame) -> pd.Series:
    hm = df["scheduled_time"].astype(str).str.extract(r"(\d{1,2}):(\d{2})").astype(float)
    return pd.to_datetime(df["flight_date"], errors="coerce") + pd.to_timedelta(
        hm[0].fillna(12.0) * 60 + hm[1].fillna(0.0), unit="m"
    )


class FeatureBuilder:
    """Shared feature engineering transformer used identically in batch training and single-flight serving."""

    def __init__(
        self,
        horizon_hours: int = PRIMARY_HORIZON_HOURS,
        history_store: Optional[CausalHistoryStore] = None,
    ):
        self.horizon_hours = int(horizon_hours)
        self.history_store = history_store

    def build_dataframe(
        self,
        df: pd.DataFrame,
        history_store: Optional[CausalHistoryStore] = None,
    ) -> pd.DataFrame:
        """Transform Silver rows (or synthetic pre-flight inputs) into Gold feature matrix rows."""
        work = df.copy()
        work["scheduled_dt"] = _compute_scheduled_dt(work)

        if "prediction_time" not in work.columns or work["prediction_time"].isna().any():
            pred_dt = work["scheduled_dt"] - pd.to_timedelta(self.horizon_hours, unit="h")
            work["prediction_time"] = pred_dt.dt.strftime("%Y-%m-%d %H:%M:%S")
        work["horizon_hours"] = self.horizon_hours

        store = history_store or self.history_store
        if store is None:
            # Causal self-history fallback: build strictly point-in-time index from rows that have outcome columns
            if {"status_clean", "scheduled_time", "flight_date"}.issubset(work.columns):
                store = CausalHistoryStore.from_observations(work)
            else:
                store = CausalHistoryStore.from_observations(pd.DataFrame())
            self.history_store = store

        work = compute_temporal_features(work)
        work = compute_airport_features(
            work,
            airport_baseline_loads=store.airport_median_load,
            default_median_load=store.global_median_load,
        )
        work = compute_weather_features(work)
        work = compute_airline_features(work)
        work = compute_operational_features(work)
        work = store.enrich_dataframe(work)

        missing = [c for c in FEATURE_REGISTRY.model_input_columns if c not in work.columns]
        if missing:
            raise RuntimeError(f"FeatureBuilder contract violated; missing columns: {missing}")
        assert_no_leakage(FEATURE_REGISTRY.model_input_columns)
        return work

    def build_single(
        self,
        airline_code: str,
        airport: str,
        direction: str,
        route_type: str,
        scheduled_time: str,
        flight_date: str,
        precipitation: float = 0.0,
        wind_gusts_10m: float = 10.0,
        visibility: float = 10000.0,
        weather_code: int = 0,
        airport_load: int = 20,
        origin: str = "SGN",
        destination: str = "HAN",
        prediction_time: Optional[str] = None,
        history_store: Optional[CausalHistoryStore] = None,
    ) -> pd.DataFrame:
        """Build 1-row DataFrame using the EXACT same pipeline as build_dataframe."""
        sched_dt = pd.to_datetime(f"{flight_date} {scheduled_time}", errors="coerce")
        if pd.isna(sched_dt):
            raise InvalidPredictionTimeError(f"Invalid flight_date/scheduled_time: {flight_date} {scheduled_time}")

        if prediction_time is None:
            pred_dt = sched_dt - timedelta(hours=self.horizon_hours)
        else:
            pred_dt = pd.to_datetime(prediction_time, errors="coerce")
            if pd.isna(pred_dt):
                raise InvalidPredictionTimeError(f"Invalid prediction_time format: {prediction_time!r}")
            if pred_dt > sched_dt:
                raise InvalidPredictionTimeError(
                    f"prediction_time ({pred_dt}) must not be later than scheduled departure ({sched_dt})."
                )

        # Compute identical holiday/day_type/time_slot context matching Step 2 Silver transformation
        parsed_date = pd.to_datetime(flight_date).date()
        is_wknd = parsed_date.weekday() >= 5
        vn_hol = holidays.country_holidays("VN", years=[parsed_date.year])
        hr = sched_dt.hour
        slot = "00h-06h" if hr < 6 else "06h-12h" if hr < 12 else "12h-18h" if hr < 18 else "18h-24h"
        resolved_route = (
            "domestic"
            if (origin.upper() in VN_AIRPORTS and destination.upper() in VN_AIRPORTS)
            else route_type.lower()
        )

        row = {
            "leg_key": f"{airline_code.upper()}0000|{flight_date}|{origin.upper()}>{destination.upper()}",
            "flight_date": flight_date,
            "flight_no": f"{airline_code.upper()}0000",
            "airline_code": airline_code.upper(),
            "origin": origin.upper(),
            "destination": destination.upper(),
            "route_type": resolved_route,
            "airport": airport.upper(),
            "direction": direction.upper(),
            "scheduled_time": scheduled_time,
            "prediction_time": pred_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "time_slot": slot,
            "day_type": "weekend" if is_wknd else "weekday",
            "is_holiday": parsed_date in vn_hol,
            "airport_load": float(airport_load),
            "precipitation": float(precipitation),
            "wind_gusts_10m": float(wind_gusts_10m),
            "visibility": float(visibility),
            "weather_code": int(weather_code),
            "source": "inference_input",
        }
        return self.build_dataframe(pd.DataFrame([row]), history_store=history_store)
