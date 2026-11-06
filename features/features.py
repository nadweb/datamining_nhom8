# -*- coding: utf-8 -*-
"""Feature Engineering Module (Part 3)
Builds domain-specific, machine learning-ready features without data leakage.

Key feature groups:
1. Cyclical time transformations (hour_sin/cos, dow_sin/cos)
2. Airport congestion & peak hour flags
3. Weather hazard index & individual severe weather indicators
4. Route classification (Golden Triangle routes, domestic/international)
5. Carrier category (Full-service vs LCC vs Regional)
6. Congestion-Weather interaction terms
"""
import sys
from pathlib import Path
from typing import Dict, Union

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from config import BASE_DIR, FILE_CLEANED

PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FILE_FEATURES = PROCESSED_DIR / "flights_features.csv"

# Golden triangle of Vietnam aviation
GOLDEN_TRIANGLE = {
    ("SGN", "HAN"), ("HAN", "SGN"),
    ("SGN", "DAD"), ("DAD", "SGN"),
    ("HAN", "DAD"), ("DAD", "HAN"),
}

# Carrier classification
CARRIER_TYPE = {
    "VN": "FSC",        # Vietnam Airlines (Full Service Carrier)
    "VJ": "LCC",        # VietJet Air (Low Cost Carrier)
    "QH": "HYBRID",     # Bamboo Airways
    "BL": "LCC",        # Pacific Airlines
    "VU": "TOURIST",    # Vietravel Airlines
    "0V": "REGIONAL",   # VASCO
    "9G": "SEAPLANE",   # Sun PhuQuoc
}


def compute_cyclical_time(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cyclical sine and cosine representations for hour and day of week."""
    df = df.copy()
    hm = df["scheduled_time"].astype(str).str.extract(r"(\d{1,2}):(\d{2})").astype(float)
    hours = hm[0].fillna(12.0) + hm[1].fillna(0.0) / 60.0
    df["scheduled_hour_num"] = hours

    # 24-hour cycle
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)

    # 7-day week cycle
    flight_dates = pd.to_datetime(df["flight_date"], errors="coerce")
    dow = flight_dates.dt.dayofweek.fillna(2).astype(float)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    return df


def compute_airport_congestion(df: pd.DataFrame) -> pd.DataFrame:
    """Add congestion indicators based on airport_load and rush hours."""
    df = df.copy()
    load = df["airport_load"].fillna(df["airport_load"].median() if df["airport_load"].notna().any() else 20.0)

    # Peak operational hours in Vietnam airports: morning (06:00 - 09:30) and evening (17:00 - 20:30)
    h = df["scheduled_hour_num"]
    morning_peak = (h >= 6.0) & (h <= 9.5)
    evening_peak = (h >= 17.0) & (h <= 20.5)
    df["is_peak_hour"] = (morning_peak | evening_peak).astype(int)

    # High load flag (top 25% percentile per airport or load > 30)
    df["is_heavy_traffic"] = (load > 28).astype(int)
    return df


def compute_weather_hazard(df: pd.DataFrame) -> pd.DataFrame:
    """Compute multi-factor weather hazard scores and threshold flags."""
    df = df.copy()
    precip = df["precipitation"].fillna(0.0)
    wind = df["wind_gusts_10m"].fillna(10.0)
    vis = df["visibility"].fillna(10000.0)
    wcode = df["weather_code"].fillna(0).astype(int)

    df["is_rain"] = (precip > 0.1).astype(int)
    df["is_heavy_rain"] = (precip > 2.5).astype(int)
    df["is_high_wind"] = (wind > 30.0).astype(int)
    df["is_severe_wind"] = (wind > 50.0).astype(int)
    df["is_low_vis"] = (vis < 5000.0).astype(int)
    df["is_storm"] = (wcode >= 95).astype(int)

    # Composite continuous hazard index (0.0 to 1.0)
    # Scaled components: rain (max 10mm), wind (max 60km/h), low visibility (inverted from 10km), storm bonus
    score = (
        np.clip(precip / 10.0, 0, 1) * 0.35 +
        np.clip((wind - 15.0) / 45.0, 0, 1) * 0.25 +
        np.clip((10000.0 - vis) / 10000.0, 0, 1) * 0.20 +
        df["is_storm"] * 0.20
    )
    df["weather_hazard_score"] = np.round(score, 4)
    return df


def compute_route_and_carrier_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive route importance, carrier business model, and operational context."""
    df = df.copy()

    # Carrier category
    df["carrier_category"] = df["airline_code"].map(CARRIER_TYPE).fillna("OTHER")

    # Golden Triangle flag (SGN - HAN - DAD)
    pairs = list(zip(df["origin"], df["destination"]))
    df["is_golden_triangle"] = [1 if p in GOLDEN_TRIANGLE else 0 for p in pairs]

    # Holiday & Weekend flags
    df["is_holiday_flag"] = df["is_holiday"].fillna(False).astype(int)
    df["is_weekend_flag"] = (df["day_type"] == "weekend").astype(int)

    # Route type flag
    df["is_domestic_flag"] = (df["route_type"] == "domestic").astype(int)

    # Departure direction flag
    df["is_departure"] = (df["direction"].str.upper() == "DEPARTURES").astype(int)

    return df


def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct multi-factor interaction terms."""
    df = df.copy()
    # High load compound with weather hazard
    df["load_x_weather"] = np.round(df["airport_load"].fillna(20.0) * df["weather_hazard_score"], 4)
    # Peak hour and heavy rain compound
    df["peak_x_rain"] = df["is_peak_hour"] * df["is_rain"]
    return df


def build_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline entry point: generate all engineered features for modeling.

    Ensures zero data leakage: does NOT use delay_min, actual_time, status, status_detail.
    """
    df_feat = df.copy()
    df_feat = compute_cyclical_time(df_feat)
    df_feat = compute_airport_congestion(df_feat)
    df_feat = compute_weather_hazard(df_feat)
    df_feat = compute_route_and_carrier_features(df_feat)
    df_feat = compute_interaction_features(df_feat)
    return df_feat


def extract_single_flight_features(
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
    destination: str = "HAN"
) -> pd.DataFrame:
    """Helper for web inference: generate 1-row feature dataframe from raw flight request."""
    row = {
        "leg_key": f"{airline_code}0000|{flight_date}|{origin}>{destination}",
        "flight_date": flight_date,
        "flight_no": f"{airline_code}0000",
        "airline_code": airline_code.upper(),
        "origin": origin.upper(),
        "destination": destination.upper(),
        "route_type": route_type.lower(),
        "airport": airport.upper(),
        "direction": direction.upper(),
        "scheduled_time": scheduled_time,
        "time_slot": "06h-12h",
        "day_type": "weekday",
        "is_holiday": False,
        "airport_load": airport_load,
        "precipitation": float(precipitation),
        "wind_gusts_10m": float(wind_gusts_10m),
        "visibility": float(visibility),
        "weather_code": int(weather_code),
        "status_clean": "on_time",  # dummy for format
        "source": "web_input",
    }
    df = pd.DataFrame([row])
    return build_engineered_features(df)


def run_feature_pipeline(source_path: Path = FILE_CLEANED) -> pd.DataFrame:
    """CLI runner to create features dataset."""
    print("=" * 65)
    print("             PART 3: FEATURE ENGINEERING PIPELINE")
    print("=" * 65)
    if not source_path.exists():
        fallback = source_path.parent / "flights_sample.csv"
        if fallback.exists():
            print(f"(!) {source_path.name} not found. Using sample: {fallback.name}")
            source_path = fallback
        else:
            raise FileNotFoundError(f"Neither {source_path} nor {fallback} exists!")

    raw_df = pd.read_csv(source_path)
    print(f"Loaded source data: {len(raw_df):,} rows, {len(raw_df.columns)} columns")

    features_df = build_engineered_features(raw_df)
    features_df.to_csv(FILE_FEATURES, index=False, encoding="utf-8-sig")

    print(f"Successfully engineered features: {features_df.shape[0]:,} rows x {features_df.shape[1]} columns")
    print(f"Saved features dataset to: {FILE_FEATURES}")

    new_cols = [c for c in features_df.columns if c not in raw_df.columns]
    print(f"\nGenerated {len(new_cols)} new feature columns:")
    print("  " + ", ".join(new_cols))
    print("=" * 65)
    return features_df


if __name__ == "__main__":
    run_feature_pipeline()

