# -*- coding: utf-8 -*-
"""[STEP 3] Aviation Weather Flags & Continuous Hazard Composite Index."""
import numpy as np
import pandas as pd


def compute_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    precip = pd.to_numeric(df.get("precipitation", 0.0), errors="coerce").fillna(0.0)
    wind = pd.to_numeric(df.get("wind_gusts_10m", 10.0), errors="coerce").fillna(10.0)
    vis = pd.to_numeric(df.get("visibility", 10000.0), errors="coerce").fillna(10000.0)
    wcode = pd.to_numeric(df.get("weather_code", 0), errors="coerce").fillna(0).astype(int)

    df["is_rain"] = (precip > 0.1).astype(int)
    df["is_heavy_rain"] = (precip > 2.5).astype(int)
    df["is_high_wind"] = (wind > 30.0).astype(int)
    df["is_severe_wind"] = (wind > 50.0).astype(int)
    df["is_low_vis"] = (vis < 5000.0).astype(int)
    df["is_storm"] = (wcode >= 95).astype(int)

    score = (
        np.clip(precip / 10.0, 0, 1) * 0.35
        + np.clip((wind - 15.0) / 45.0, 0, 1) * 0.25
        + np.clip((10000.0 - vis) / 10000.0, 0, 1) * 0.20
        + df["is_storm"] * 0.20
    )
    df["weather_hazard_score"] = np.round(score, 4)
    return df
