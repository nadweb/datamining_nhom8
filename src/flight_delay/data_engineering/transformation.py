# -*- coding: utf-8 -*-
"""[STEP 2] Modulo 1440 Delay Calculation, IATA/ICAO Resolution & Outcome Policies."""
import re
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from flight_delay.common.constants import (
    AIRPORT_COORD,
    CITY_IATA,
    DELAY_THRESHOLD_MIN,
    EARLY_LIMIT,
    VN_AIRPORTS,
)
from flight_delay.common.utils import norm_text


def parse_time_min(t_str: Union[str, float]) -> Tuple[Optional[int], int]:
    """Parse 'HH:MM', 'HH:MM+', or 'HH:MM-' into (minutes_from_midnight, day_offset)."""
    if t_str is None or pd.isna(t_str):
        return None, 0
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*([+-]?)\s*", str(t_str))
    if not m:
        return None, 0
    h, mins, sign = int(m.group(1)), int(m.group(2)), m.group(3)
    if h > 23 or mins > 59:
        return None, 0
    return h * 60 + mins, {"-": -1, "+": 1, "": 0}[sign]


def _valid_time_str(value) -> bool:
    return value is not None and not pd.isna(value) and str(value).strip().lower() not in {"", "nan", "none", "nat"}


def calculate_delay(sched: str, est: str, actual: str, status_str: str) -> float:
    """Calculate delay duration in minutes using Modulo 1440 rollover correction."""
    st = "" if status_str is None or pd.isna(status_str) else str(status_str)
    m_late = re.search(r"(?:(\d+)\s*minutes?\s*late|delayed\s*(\d+)\s*minutes?)", st, re.IGNORECASE)
    if m_late:
        return float(m_late.group(1) or m_late.group(2))
    m_early = re.search(r"(\d+)\s*minutes?\s*early", st, re.IGNORECASE)
    if m_early:
        return -float(m_early.group(1))

    target = actual if _valid_time_str(actual) else est
    s_min, _ = parse_time_min(sched)
    t_min, off = parse_time_min(target)
    if s_min is None or t_min is None:
        return np.nan
    diff = (t_min + off * 1440) - s_min
    if off == 0:
        if diff < -720:
            diff += 1440
        elif diff > 720:
            diff -= 1440
    return float(diff)


def _s(v) -> str:
    return "" if v is None or pd.isna(v) else str(v).strip()


def label_status(row: Union[pd.Series, dict], delay_threshold: int = DELAY_THRESHOLD_MIN) -> Tuple[str, str, str]:
    """Assign standardized outcome tuple: (status, status_detail, status_clean).

    Policy (Section 3.3):
    - Diverted, incident, scheduled, or early outliers (< EARLY_LIMIT) map cleanly to
      'diverted' / 'scheduled' / 'unknown' and are NEVER coerced into 0/1/2.
    - Cancelled strictly preserves delay_min = NaN.
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

    if re.search(r"cancel|huy|hủy", st):
        return "Cancelled", "cancelled", "cancelled"
    if "divert" in st:
        return "Diverted", "diverted", "diverted"

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
                return st_raw, "early_outlier", "unknown"
            if delay >= thr:
                return f"{prefix} {n} minutes late", f"{tag}_late", "delayed"
            return f"{prefix} On-Time", f"{tag}_on_time", "on_time"
        if late_txt:
            return st_raw, f"{tag}_late", "delayed"
        if ontime_txt:
            return f"{prefix} On-Time", f"{tag}_on_time", "on_time"
        return st_raw, "unknown", "unknown"

    if st in ("", "scheduled"):
        if has_delay and delay >= thr:
            return f"Delayed {n} minutes", "delayed", "delayed"
        return "Scheduled", "scheduled", "unknown"
    if has_delay:
        if delay >= thr:
            return f"Delayed {n} minutes", "delayed", "delayed"
        if delay >= EARLY_LIMIT:
            return "On-Time", "on_time", "on_time"
        return st_raw, "early_outlier", "unknown"
    if late_txt:
        return st_raw, "delayed", "delayed"
    if ontime_txt:
        return "On-Time", "on_time", "on_time"
    return st_raw, "unknown", "unknown"


def standardize_status(row: Union[pd.Series, dict], delay_threshold: int = DELAY_THRESHOLD_MIN) -> str:
    return label_status(row, delay_threshold)[2]


def impute_actual_time(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing actual_time strictly for completed flights; cancelled keeps NaN."""
    df = df.copy()
    if "actual_time" not in df.columns or "scheduled_time" not in df.columns:
        return df

    def _fill_time(row):
        act = row.get("actual_time")
        if _valid_time_str(act):
            return str(act).strip()
        st_clean = str(row.get("status_clean", "")).strip().lower()
        if st_clean in ("cancelled", "diverted", "unknown"):
            return np.nan
        s_min, _ = parse_time_min(str(row.get("scheduled_time", "")).strip())
        if s_min is None:
            return np.nan
        est = str(row.get("estimated_time", "")).strip() if "estimated_time" in row else ""
        e_min, _ = parse_time_min(est)
        if st_clean == "delayed":
            delay = row.get("delay_min")
            if pd.notna(delay):
                tot_min = (s_min + int(round(float(delay)))) % 1440
                return f"{(tot_min // 60) % 24:02d}:{tot_min % 60:02d}"
        if e_min is not None:
            return f"{(e_min // 60) % 24:02d}:{e_min % 60:02d}"
        return f"{(s_min // 60) % 24:02d}:{s_min % 60:02d}"

    df["actual_time"] = df.apply(_fill_time, axis=1)
    return df


def apply_labels(df: pd.DataFrame, thr: int = DELAY_THRESHOLD_MIN) -> pd.DataFrame:
    df = df.copy()
    res = df.apply(label_status, axis=1, delay_threshold=thr)
    df["status"] = [r[0] for r in res]
    df["status_detail"] = [r[1] for r in res]
    df["status_clean"] = [r[2] for r in res]
    df.loc[df["status_clean"] == "cancelled", "delay_min"] = np.nan
    return impute_actual_time(df)


def to_iata(s: Union[str, float]) -> str:
    u = str(s or "").upper()
    m = re.search(r"\(([A-Z]{3})\)", u)
    if m:
        return m.group(1)
    if re.fullmatch(r"\s*[A-Z]{3}\s*", u):
        return u.strip()
    n = norm_text(s)
    for k in sorted(CITY_IATA, key=len, reverse=True):
        if n == k or (len(k) >= 5 and k in n):
            return CITY_IATA[k]
    return ""


def normalize_flight_no(s: pd.Series) -> pd.Series:
    s = s.str.replace(r"\s+", "", regex=True)
    return s.str.replace(r"^([A-Z0-9]{2})0+(\d)", r"\1\2", regex=True)


def add_leg_key(df: pd.DataFrame) -> pd.DataFrame:
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


def sched_datetime(df: pd.DataFrame) -> pd.Series:
    hm = df["scheduled_time"].str.extract(r"(\d{1,2}):(\d{2})").astype(float)
    return pd.to_datetime(df["flight_date"], errors="coerce") + pd.to_timedelta(hm[0] * 60 + hm[1], unit="m")


def compute_airport_load(df: pd.DataFrame) -> pd.DataFrame:
    """Count distinct scheduled operations across ALL airlines per (airport, hour_floor)."""
    key = ["airport", "direction", "flight_date", "flight_no", "scheduled_time"]
    u = df.drop_duplicates(key)
    u = u[u["airport"].isin(AIRPORT_COORD) & u["hour_floor"].notna()]
    return u.groupby(["airport", "hour_floor"]).size().rename("airport_load").reset_index()


def _cut(s: pd.Series, bins: list, labels: list) -> pd.Series:
    return pd.cut(s, bins=bins, labels=labels, right=False).astype(object)


def add_time_features(df: pd.DataFrame, load: pd.DataFrame) -> pd.DataFrame:
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
