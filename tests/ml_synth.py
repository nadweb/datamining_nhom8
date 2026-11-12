# -*- coding: utf-8 -*-
"""Dữ liệu giả lập có tín hiệu thật (mưa lớn, giờ tối, tải cao -> trễ/hủy) cho kiểm thử Part 5."""
import numpy as np
import pandas as pd

CARRIER = {"VN": "FSC", "VJ": "LCC", "QH": "HYBRID", "BL": "LCC", "9G": "REGIONAL"}


def make_ml_features(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2026-10-08", periods=26).strftime("%Y-%m-%d")
    hour = rng.choice(np.arange(5, 24), n, p=np.r_[np.full(8, 0.07), np.full(11, 0.0404)][:19] / np.r_[np.full(8, 0.07), np.full(11, 0.0404)][:19].sum())
    minute = rng.choice([0, 15, 30, 45], n)
    df = pd.DataFrame({
        "flight_date": rng.choice(days, n),
        "scheduled_time": [f"{h:02d}:{m:02d}" for h, m in zip(hour, minute)],
        "airline_code": rng.choice(list(CARRIER), n, p=[.35, .35, .1, .1, .1]),
        "airport": rng.choice(["SGN", "HAN", "DAD", "CXR", "PQC"], n, p=[.35, .3, .15, .1, .1]),
        "direction": rng.choice(["ARRIVALS", "DEPARTURES"], n),
        "route_type": rng.choice(["domestic", "international"], n, p=[.85, .15]),
    })
    df["leg_key"] = [f"X{i}|{d}|A>B" for i, d in enumerate(df["flight_date"])]
    df["carrier_category"] = df["airline_code"].map(CARRIER)
    rain = rng.random(n) < 0.06                                    # nhóm "mưa lớn/dông"
    df["precipitation"] = np.where(rain, rng.uniform(2.5, 8, n), np.where(rng.random(n) < 0.15, rng.uniform(0.1, 1.0, n), 0.0))
    df["wind_gusts_10m"] = np.where(rain, rng.uniform(28, 50, n), rng.uniform(5, 30, n))
    df["visibility"] = np.where(rain, rng.uniform(2000, 5000, n), rng.choice([24000., 15000., 9000.], n))
    df["airport_load"] = rng.integers(8, 42, n).astype(float)
    df["weather_hazard_score"] = np.clip(df["precipitation"] / 8 * 0.6 + df["wind_gusts_10m"] / 60 * 0.4, 0, 1)
    df["load_x_weather"] = df["airport_load"] * df["weather_hazard_score"]
    ang = 2 * np.pi * (hour + minute / 60) / 24
    df["hour_sin"], df["hour_cos"] = np.sin(ang), np.cos(ang)
    dow = pd.to_datetime(df["flight_date"]).dt.dayofweek.values
    df["dow_sin"], df["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    df["is_peak_hour"] = ((hour >= 17) & (hour <= 21)).astype(int)
    df["is_heavy_traffic"] = (df["airport_load"] >= 30).astype(int)
    df["is_rain"] = (df["precipitation"] > 0.1).astype(int)
    df["is_heavy_rain"] = (df["precipitation"] >= 2.5).astype(int)
    df["is_high_wind"] = (df["wind_gusts_10m"] >= 30).astype(int)
    df["is_severe_wind"] = (df["wind_gusts_10m"] >= 50).astype(int)
    df["is_low_vis"] = (df["visibility"] < 5000).astype(int)
    df["is_storm"] = (rain & (rng.random(n) < 0.6)).astype(int)
    df["is_golden_triangle"] = rng.integers(0, 2, n)
    df["is_holiday_flag"] = 0
    df["is_weekend_flag"] = (dow >= 5).astype(int)
    df["is_domestic_flag"] = (df["route_type"] == "domestic").astype(int)
    df["is_departure"] = (df["direction"] == "DEPARTURES").astype(int)
    df["peak_x_rain"] = df["is_peak_hour"] * df["is_rain"]
    # nhãn có tín hiệu thật
    p_del = 0.17 + 0.32 * df["is_heavy_rain"] + 0.14 * df["is_peak_hour"] + 0.08 * df["is_heavy_traffic"]
    p_can = 0.008 + 0.15 * df["is_storm"]
    u = rng.random(n)
    status = np.where(u < p_can, "cancelled", np.where(u < p_can + p_del, "delayed", "on_time"))
    df["status_clean"] = status
    df["delay_min"] = np.where(status == "delayed", 15 + rng.exponential(25, n) + 12 * df["is_heavy_rain"],
                               np.where(status == "on_time", rng.normal(3, 5, n), np.nan))
    df["actual_time"], df["status"], df["status_detail"] = "10:00", "Departed On-Time", "departed_on_time"
    df["is_delayed_for_analysis"] = (status == "delayed").astype(int)
    df["source"], df["origin"], df["destination"] = "guide_sgn", "SGN", "HAN"
    return df

