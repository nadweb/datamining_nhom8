# -*- coding: utf-8 -*-
"""[STEP 2] Multi-Source Weather Merging & Cross-Source Agreement Evaluation."""
from itertools import combinations

import numpy as np
import pandas as pd

from flight_delay.common.constants import OFFICIAL_STATS, VALID_OUTCOME_LABELS, WX_COLS
from flight_delay.data_engineering.deduplication import K1
from flight_delay.data_engineering.transformation import _cut, standardize_status
from flight_delay.ingestion.weather_client import load_weather


def attach_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Attach Open-Meteo hourly weather at (airport, hour_floor) preserving fetched_at metadata."""
    df = df.copy()
    derived = ["rain_level", "wind_level", "vis_level"]
    wx = load_weather()
    if wx.empty:
        for c in WX_COLS + derived:
            df[c] = np.nan
        df["weather_fetched_at"] = ""
        df["is_weather_matched"] = 0
        return df

    base = wx[["airport", "time"] + WX_COLS + ["fetched_at"]].rename(
        columns={"time": "hour_floor", "fetched_at": "weather_fetched_at"}
    )
    df = df.merge(base, on=["airport", "hour_floor"], how="left")
    df["rain_level"] = _cut(df["precipitation"], [0, 0.1, 2.5, 1e9], ["none", "light", "heavy"])
    df["wind_level"] = _cut(df["wind_gusts_10m"], [0, 30, 50, 1e9], ["<30", "30-50", ">50"])
    df["vis_level"] = _cut(df["visibility"], [0, 5000, 1e9], ["<5km", ">=5km"])
    df["is_weather_matched"] = df["precipitation"].notna().astype(int)
    return df


def cohen_kappa(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in set(a) | set(b))
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def source_agreement(per_src: pd.DataFrame, delay_threshold: int) -> pd.DataFrame:
    ps = per_src.copy()
    ps["lab"] = ps.apply(standardize_status, axis=1, delay_threshold=delay_threshold)
    ps = ps[ps["lab"].isin(VALID_OUTCOME_LABELS)]
    if ps["source"].nunique() < 2:
        return pd.DataFrame()
    lab = ps.pivot(index=K1, columns="source", values="lab")
    dly = ps.pivot(index=K1, columns="source", values="delay_min")
    rows = []
    for a, b in combinations(lab.columns, 2):
        both = lab[[a, b]].dropna()
        if len(both) < 20:
            continue
        d = (dly.loc[both.index, a] - dly.loc[both.index, b]).abs().dropna()
        rows.append({
            "source_a": a,
            "source_b": b,
            "shared_events": int(len(both)),
            "label_agreement_pct": round(float((both[a] == both[b]).mean() * 100), 2),
            "cohen_kappa": round(cohen_kappa(both[a], both[b]), 4),
            "median_delay_diff_min": round(float(d.median()), 1) if len(d) else np.nan,
        })
    return pd.DataFrame(rows)


def airline_benchmark_table(valid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for al, g in list(valid.groupby("airline_code")) + [("ALL", valid)]:
        n = len(g)
        vc = g["status_clean"].value_counts()
        off = OFFICIAL_STATS.get(al, {})
        rows.append({
            "airline_code": al,
            "flights": int(n),
            "delay_pct": round(vc.get("delayed", 0) / n * 100, 2),
            "cancel_pct": round(vc.get("cancelled", 0) / n * 100, 2),
            "otp_pct": round(vc.get("on_time", 0) / n * 100, 2),
            "caav_delay_pct": off.get("delay"),
            "caav_cancel_pct": off.get("cancel"),
            "caav_otp_pct": off.get("otp"),
        })
    return pd.DataFrame(rows).sort_values("flights", ascending=False)
