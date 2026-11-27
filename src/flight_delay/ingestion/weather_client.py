# -*- coding: utf-8 -*-
"""[STEP 1] Open-Meteo Hourly Weather Collector & Bronze Cache Loader."""
import json
import logging
from datetime import date, datetime, timedelta

import pandas as pd
import requests

from flight_delay.common.constants import (
    AIRPORT_COORD,
    DEBUG_DIR,
    FILE_WEATHER,
    FILE_WEATHER_LEGACY,
    SOURCES_CFG,
    WX_COLS,
)

log = logging.getLogger(__name__)
HOURLY = ",".join(SOURCES_CFG["open_meteo"]["hourly_fields"])
ARCHIVE = SOURCES_CFG["open_meteo"]["archive_url"]
FORECAST = SOURCES_CFG["open_meteo"]["forecast_url"]
FORECAST_PAST_DAYS = int(SOURCES_CFG["open_meteo"]["forecast_past_days"])
ARCHIVE_LAG_DAYS = int(SOURCES_CFG["open_meteo"]["archive_lag_days"])


def _payload_to_df(payload: dict, ap: str) -> pd.DataFrame:
    df = pd.DataFrame(payload["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df["airport"] = ap
    df["fetched_at"] = (payload.get("_meta") or {}).get("scraped_at", "")
    return df


def _save_weather_raw_json(payload: dict, ap: str, kind: str, start: date, end: date) -> None:
    target_dir = DEBUG_DIR / "open_meteo"
    target_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"open_meteo_{ap}_{kind}_{start.isoformat()}_{end.isoformat()}_{ts}.json"
    (target_dir / fname).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _get(url: str, ap: str, start: date, end: date, hourly: str, kind: str = "archive") -> pd.DataFrame:
    lat, lon = AIRPORT_COORD[ap]
    r = requests.get(
        url,
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": hourly,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "timezone": "Asia/Bangkok",
        },
        timeout=60,
    )
    r.raise_for_status()
    raw_json = r.json()
    raw_json["_meta"] = {
        "airport": ap,
        "kind": kind,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "scraped_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_weather_raw_json(raw_json, ap, kind, start, end)
    return _payload_to_df(raw_json, ap)


def fetch_airport(ap: str, start: date, end: date) -> pd.DataFrame:
    today = date.today()
    end = min(end, today + timedelta(days=1))
    recent_start = today - timedelta(days=FORECAST_PAST_DAYS)
    frames = []
    if start < recent_start:
        arch_end = min(end, recent_start - timedelta(days=1))
        frames.append(_get(ARCHIVE, ap, start, arch_end, HOURLY, kind="archive"))
    fc_start = max(start, recent_start)
    if fc_start <= end:
        try:
            frames.append(_get(FORECAST, ap, fc_start, end, HOURLY + ",visibility", kind="forecast"))
        except requests.HTTPError:
            arch_end = min(end, today - timedelta(days=ARCHIVE_LAG_DAYS))
            if fc_start <= arch_end:
                frames.append(_get(ARCHIVE, ap, fc_start, arch_end, HOURLY, kind="archive"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _merge_frames(frames: list) -> pd.DataFrame:
    wx = pd.concat(frames, ignore_index=True)
    wx["time"] = pd.to_datetime(wx["time"])
    wx["fetched_at"] = wx.get("fetched_at", pd.Series("", index=wx.index)).fillna("").astype(str)
    for c in WX_COLS:
        if c not in wx.columns:
            wx[c] = float("nan")
    wx = wx.sort_values("fetched_at", kind="stable")
    out = wx.groupby(["airport", "time"], as_index=False)[WX_COLS + ["fetched_at"]].last()
    return out.sort_values(["airport", "time"]).reset_index(drop=True)


def load_weather_from_debug_json() -> pd.DataFrame:
    wx_dir = DEBUG_DIR / "open_meteo"
    if not wx_dir.exists():
        return pd.DataFrame()
    frames = []
    for p in sorted(wx_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if "hourly" not in data:
                continue
            ap = (data.get("_meta") or {}).get("airport")
            if not ap:
                parts = p.stem.split("_")
                ap = parts[2] if len(parts) > 2 else ""
            frames.append(_payload_to_df(data, ap))
        except Exception as exc:
            log.warning("Weather JSON read error %s: %s", p.name, exc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_weather() -> pd.DataFrame:
    """Load deduplicated hourly weather preferring Bronze data/raw/weather and debug JSON."""
    frames = []
    wx_json = load_weather_from_debug_json()
    if not wx_json.empty:
        frames.append(wx_json)
    for csv_path in (FILE_WEATHER, FILE_WEATHER_LEGACY):
        if csv_path.exists():
            cache = pd.read_csv(csv_path, parse_dates=["time"])
            if "fetched_at" not in cache.columns:
                cache["fetched_at"] = ""
            frames.append(cache)
    return _merge_frames(frames) if frames else pd.DataFrame()


def update_weather(start: date, end: date, airports=None):
    got = False
    for ap in (airports or AIRPORT_COORD):
        try:
            df = fetch_airport(ap, start, end)
            got = got or not df.empty
        except Exception as exc:
            log.error("[weather %s] failed: %s", ap, exc)
    if not got:
        return None
    wx = load_weather()
    FILE_WEATHER.parent.mkdir(parents=True, exist_ok=True)
    wx.to_csv(FILE_WEATHER, index=False)
    return wx


def update_from_raw() -> None:
    from flight_delay.ingestion.collector import load_standard_records

    df = load_standard_records()
    if df.empty:
        log.warning("No raw records available to update weather range.")
        return
    d = pd.to_datetime(df["flight_date"], errors="coerce").dropna()
    if not d.empty:
        update_weather(d.min().date(), d.max().date())
