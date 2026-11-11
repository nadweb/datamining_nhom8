# -*- coding: utf-8 -*-
"""Data Transformation Module (Part 2: Feature & Schema Transformations)
Responsible for:
1. IATA code mapping & flight number normalization
2. Flight leg key derivation and domestic/international classification
3. Delay calculation (including midnight rollover & text parsing)
4. Target labeling (on_time, delayed, cancelled)
5. Time features (time_slot, day_type, is_holiday, airport_load)
6. Weather merging and hazard level discretization
"""
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from config import AIRPORT_COORD, CITY_IATA, VN_AIRPORTS
from pipeline.collector import norm
from pipeline.weather import WX_COLS, load_weather

EARLY_LIMIT = -90.0  # Flights departing >90 min early treated as anomalies


def parse_time_min(t_str: Union[str, float]) -> Tuple[Optional[int], int]:
    """Parse 'HH:MM', 'HH:MM+', or 'HH:MM-' into (minutes_from_midnight, day_offset).

    Returns:
        (total_minutes, day_offset): e.g., '01:30' -> (90, 0); '00:15+' -> (15, 1)
    """
    if t_str is None or pd.isna(t_str):
        return None, 0
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*([+-]?)\s*", str(t_str))
    if not m:
        return None, 0
    h, mins, sign = int(m.group(1)), int(m.group(2)), m.group(3)
    if h > 23 or mins > 59:
        return None, 0
    return h * 60 + mins, {"-": -1, "+": 1, "": 0}[sign]


def calculate_delay(sched: str, est: str, actual: str, status_str: str) -> float:
    """Calculate flight delay in minutes.

    Priority:
    1. Direct delay regex in status text ('X minutes late', 'delayed X minutes')
    2. Early flight regex ('X minutes early')
    3. Difference between actual (or estimated) and scheduled time with midnight rollover adjustment.
    """
    st = "" if status_str is None or pd.isna(status_str) else str(status_str)
    m_late = re.search(r"(?:(\d+)\s*minutes?\s*late|delayed\s*(\d+)\s*minutes?)", st, re.IGNORECASE)
    if m_late:
        return float(m_late.group(1) or m_late.group(2))
    m_early = re.search(r"(\d+)\s*minutes?\s*early", st, re.IGNORECASE)
    if m_early:
        return -float(m_early.group(1))

    def _valid_time(value):
        return value is not None and not pd.isna(value) and str(value).strip().lower() not in {"", "nan", "none", "nat"}

    target = actual if _valid_time(actual) else est
    s_min, _ = parse_time_min(sched)
    t_min, off = parse_time_min(target)
    if s_min is None or t_min is None:
        return np.nan
    diff = (t_min + off * 1440) - s_min
    if off == 0:  # Midnight rollover handling (e.g. scheduled 23:55, departed 00:15)
        if diff < -720:
            diff += 1440
        elif diff > 720:
            diff -= 1440
    return float(diff)


def _s(v) -> str:
    return "" if v is None or pd.isna(v) else str(v).strip()


def label_status(row: Union[pd.Series, dict], delay_threshold: int = 15) -> Tuple[str, str, str]:
    """Assign standardized status tuple: (status, status_detail, status_clean).

    Returns:
        status: Text summary (e.g. 'Departed On-Time', 'Delayed 25 minutes', 'Cancelled')
        status_detail: Fine-grained status category
        status_clean: Target label in {'on_time', 'delayed', 'cancelled', 'unknown'}
    """
    thr = delay_threshold
    st_raw = _s(row.get("status"))
    st = st_raw.lower()
    direction = _s(row.get("direction")).upper()
    delay = row.get("delay_min", np.nan)
    has_delay = pd.notna(delay)
    n = int(round(delay)) if has_delay else 0
    has_act = bool(_s(row.get("actual_time")))
    late_txt = bool(re.search(r"late|delay", st))
    ontime_txt = bool(re.search(r"on-?time", st))

    # 1. Cancelled / Diverted
    if re.search(r"cancel|huy|hủy", st):
        return "Cancelled", "cancelled", "cancelled"
    if "divert" in st:
        return "Diverted", "diverted", "unknown"

    # 2. Arrived / Departed (Completed flights)
    if re.search(r"arrive|land|hacan", st):
        prefix = "Arrived"
    elif re.search(r"depart|catcan", st):
        prefix = "Departed"
    elif has_act:
        prefix = "Departed" if direction == "DEPARTURES" else "Arrived" if direction == "ARRIVALS" else ""
    else:
        prefix = ""

    if prefix:
        tag = prefix.lower()
        if has_delay:
            if delay < EARLY_LIMIT:
                return st_raw, "unknown", "unknown"
            if delay >= thr:
                return f"{prefix} {n} minutes late", f"{tag}_late", "delayed"
            return f"{prefix} On-Time", f"{tag}_on_time", "on_time"
        if late_txt:
            return st_raw, f"{tag}_late", "delayed"
        if ontime_txt:
            return f"{prefix} On-Time", f"{tag}_on_time", "on_time"
        return st_raw, "unknown", "unknown"

    # 3. Scheduled / Upcoming flights
    if st in ("", "scheduled"):
        if has_delay and delay >= thr:
            return f"Delayed {n} minutes", "delayed", "delayed"
        return "Scheduled", "scheduled", "unknown"
    if has_delay:
        if delay >= thr:
            return f"Delayed {n} minutes", "delayed", "delayed"
        if delay >= EARLY_LIMIT:
            return "On-Time", "on_time", "on_time"
        return st_raw, "unknown", "unknown"
    if late_txt:
        return st_raw, "delayed", "delayed"
    if ontime_txt:
        return "On-Time", "on_time", "on_time"
    return st_raw, "unknown", "unknown"


def standardize_status(row: Union[pd.Series, dict], delay_threshold: int = 15) -> str:
    """Convenience helper returning only the status_clean label."""
    return label_status(row, delay_threshold)[2]


def apply_labels(df: pd.DataFrame, thr: int = 15) -> pd.DataFrame:
    """Apply labeling rule to entire dataframe and enforce cancelled delay_min is NaN."""
    df = df.copy()
    res = df.apply(label_status, axis=1, delay_threshold=thr)
    df["status"] = [r[0] for r in res]
    df["status_detail"] = [r[1] for r in res]
    df["status_clean"] = [r[2] for r in res]
    df.loc[df["status_clean"] == "cancelled", "delay_min"] = np.nan
    return df


def to_iata(s: Union[str, float]) -> str:
    """Map destination/origin text or raw string to 3-letter IATA code."""
    u = str(s or "").upper()
    m = re.search(r"\(([A-Z]{3})\)", u)
    if m:
        return m.group(1)
    if re.fullmatch(r"\s*[A-Z]{3}\s*", u):
        return u.strip()
    n = norm(s)
    for k in sorted(CITY_IATA, key=len, reverse=True):
        if n == k or (len(k) >= 5 and k in n):
            return CITY_IATA[k]
    return ""


def normalize_flight_no(s: pd.Series) -> pd.Series:
    """Normalize flight numbers: remove whitespace, strip leading zeros after airline code."""
    s = s.str.replace(r"\s+", "", regex=True)
    return s.str.replace(r"^([A-Z0-9]{2})0+(\d)", r"\1\2", regex=True)


def add_leg_key(df: pd.DataFrame) -> pd.DataFrame:
    """Derive origin, destination, route_type, and unique flight leg_key."""
    df = df.copy()
    cache = {u: to_iata(u) for u in df["other_airport"].unique()}
    df["other_code"] = df["other_airport"].map(cache)
    dep = df["direction"] == "DEPARTURES"
    df["origin"] = np.where(dep, df["airport"], df["other_code"])
    df["destination"] = np.where(dep, df["other_code"], df["airport"])
    ok = df["other_code"] != ""
    df["leg_key"] = np.where(
        ok,
        df["flight_no"] + "|" + df["flight_date"] + "|" + df["origin"] + ">" + df["destination"],
        df["flight_no"] + "|" + df["flight_date"] + "|" + df["airport"] + "|" + df["direction"],
    )
    df["leg_matched"] = ok
    domestic = df["origin"].isin(VN_AIRPORTS) & df["destination"].isin(VN_AIRPORTS)
    df["route_type"] = np.where(domestic, "domestic", np.where(ok, "international", ""))
    return df


def _cut(s: pd.Series, bins: list, labels: list) -> pd.Series:
    return pd.cut(s, bins=bins, labels=labels, right=False).astype(object)


def sched_datetime(df: pd.DataFrame) -> pd.Series:
    """Combine flight_date and scheduled_time into datetime."""
    hm = df["scheduled_time"].str.extract(r"(\d{1,2}):(\d{2})").astype(float)
    return pd.to_datetime(df["flight_date"], errors="coerce") + pd.to_timedelta(hm[0] * 60 + hm[1], unit="m")


def compute_airport_load(df: pd.DataFrame) -> pd.DataFrame:
    """Count flights per airport per hour floor across all airlines."""
    K1 = ["airport", "direction", "flight_date", "flight_no", "scheduled_time"]
    u = df.drop_duplicates(K1)
    u = u[u["airport"].isin(AIRPORT_COORD) & u["hour_floor"].notna()]
    return u.groupby(["airport", "hour_floor"]).size().rename("airport_load").reset_index()


def add_time_features(df: pd.DataFrame, load: pd.DataFrame) -> pd.DataFrame:
    """Generate time_slot, day_type, is_holiday, and join airport_load."""
    df = df.copy()
    dt = df["sched_dt"]
    df["time_slot"] = _cut(dt.dt.hour, [0, 6, 12, 18, 24], ["00h-06h", "06h-12h", "12h-18h", "18h-24h"])
    wd = dt.dt.dayofweek
    df["day_type"] = np.where(wd.isna(), None, np.where(wd >= 5, "weekend", "weekday"))
    try:
        import holidays
        years = sorted(dt.dt.year.dropna().unique().astype(int))
        vn = holidays.country_holidays("VN", years=years if years else [2026])
        df["is_holiday"] = dt.dt.date.map(lambda d: d in vn if pd.notna(d) else False)
    except Exception:
        df["is_holiday"] = False
    return df.merge(load, on=["airport", "hour_floor"], how="left")


def attach_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Merge hourly weather observation by (airport, hour_floor) and compute hazard categories."""
    df = df.copy()
    derived = ["rain_level", "wind_level", "vis_level"]
    wx = load_weather()
    if wx.empty:
        for c in WX_COLS + derived:
            df[c] = np.nan
        return df

    base = wx[["airport", "time"] + WX_COLS].rename(columns={"time": "hour_floor"})
    df = df.merge(base, on=["airport", "hour_floor"], how="left")
    df["rain_level"] = _cut(df["precipitation"], [0, 0.1, 2.5, 1e9], ["none", "light", "heavy"])
    df["wind_level"] = _cut(df["wind_gusts_10m"], [0, 30, 50, 1e9], ["<30", "30-50", ">50"])
    df["vis_level"] = _cut(df["visibility"], [0, 5000, 1e9], ["<5km", ">=5km"])
    return df

