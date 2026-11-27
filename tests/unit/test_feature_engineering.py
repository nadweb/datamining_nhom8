# -*- coding: utf-8 -*-
"""[STAGE 03 TESTS] Domain Feature Engineering & Preprocessing Determinism."""
import numpy as np
import pandas as pd

from flight_delay.data_mining.encoding.preprocessing import FlightPreprocessor
from flight_delay.data_mining.feature_engineering.builder import FeatureBuilder
from flight_delay.data_mining.feature_engineering.registry import FEATURE_REGISTRY


def test_feature_builder_cyclical_weather_and_golden_triangle():
    fb = FeatureBuilder(horizon_hours=3)
    df = fb.build_single(
        airline_code="VN",
        airport="SGN",
        direction="DEPARTURES",
        route_type="domestic",
        scheduled_time="18:30",
        flight_date="2026-11-06",
        precipitation=12.0,
        wind_gusts_10m=45.0,
        visibility=2000.0,
        weather_code=95,
        airport_load=34,
        origin="SGN",
        destination="HAN",
    )
    row = df.iloc[0]
    assert int(row["is_golden_triangle"]) == 1
    assert int(row["is_peak_hour"]) == 1
    assert int(row["is_rain"]) == 1
    assert int(row["is_heavy_rain"]) == 1
    assert int(row["is_storm"]) == 1
    assert 0.0 < float(row["weather_hazard_score"]) <= 1.0
    assert np.isclose(row["hour_sin"] ** 2 + row["hour_cos"] ** 2, 1.0, atol=1e-5)
    assert np.isclose(row["dow_sin"] ** 2 + row["dow_cos"] ** 2, 1.0, atol=1e-5)


def test_flight_preprocessor_handles_unseen_categories_deterministically():
    fb = FeatureBuilder(horizon_hours=3)
    df_tr = pd.concat([
        fb.build_single("VN", "SGN", "DEPARTURES", "domestic", "08:00", "2026-10-15"),
        fb.build_single("VJ", "HAN", "ARRIVALS", "international", "20:00", "2026-10-16"),
    ], ignore_index=True)

    prep = FlightPreprocessor().fit(df_tr)
    df_unseen = fb.build_single("XX", "VVV", "DEPARTURES", "domestic", "12:00", "2026-10-17")
    mat = prep.transform(df_unseen)
    assert mat.shape == (1, len(prep.feature_names))
    assert set(FEATURE_REGISTRY.model_input_columns).issubset(df_tr.columns)
