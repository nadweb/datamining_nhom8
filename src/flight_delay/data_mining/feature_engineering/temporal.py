# -*- coding: utf-8 -*-
"""[STEP 3] Temporal & Calendar Cyclical Feature Builders."""
import numpy as np
import pandas as pd


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hm = df["scheduled_time"].astype(str).str.extract(r"(\d{1,2}):(\d{2})").astype(float)
    hours = hm[0].fillna(12.0) + hm[1].fillna(0.0) / 60.0
    df["scheduled_hour_num"] = hours
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)

    flight_dates = pd.to_datetime(df["flight_date"], errors="coerce")
    dow = flight_dates.dt.dayofweek.fillna(2).astype(float)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    morning_peak = (hours >= 6.0) & (hours <= 9.5)
    evening_peak = (hours >= 17.0) & (hours <= 20.5)
    df["is_peak_hour"] = (morning_peak | evening_peak).astype(int)

    if "is_holiday" in df.columns:
        df["is_holiday_flag"] = df["is_holiday"].fillna(False).astype(bool).astype(int)
    else:
        try:
            import holidays
            years = sorted(flight_dates.dt.year.dropna().unique().astype(int))
            vn = holidays.country_holidays("VN", years=years if years else [2026])
            df["is_holiday_flag"] = flight_dates.dt.date.map(lambda d: int(d in vn) if pd.notna(d) else 0)
        except Exception:
            df["is_holiday_flag"] = 0

    if "day_type" in df.columns:
        df["is_weekend_flag"] = (df["day_type"].astype(str).str.lower() == "weekend").astype(int)
    else:
        df["is_weekend_flag"] = (dow >= 5).astype(int)
    return df
