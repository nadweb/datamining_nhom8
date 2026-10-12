# -*- coding: utf-8 -*-
"""Tải thời tiết theo giờ từ Open-Meteo (miễn phí, không cần key) và lưu cache CSV.
- Ngày cũ: archive-api (ERA5, KHÔNG có tầm nhìn/visibility).
- 6 ngày gần đây: forecast-api (có visibility).
Giờ trả về theo múi giờ Asia/Bangkok = giờ địa phương của lịch bay.
"""
import glob
import logging
from datetime import date, timedelta
import pandas as pd
import requests
from config import AIRPORT_COORD, FILE_WEATHER, RAW_DIR

log = logging.getLogger("weather")
HOURLY = "precipitation,wind_gusts_10m,cloud_cover,weather_code"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"


def _get(url, ap, start, end, hourly):
    lat, lon = AIRPORT_COORD[ap]
    r = requests.get(url, params={"latitude": lat, "longitude": lon, "hourly": hourly,
                                  "start_date": start.isoformat(), "end_date": end.isoformat(),
                                  "timezone": "Asia/Bangkok"}, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df["airport"] = ap
    return df


def fetch_airport(ap, start, end):
    today = date.today()
    end = min(end, today + timedelta(days=1))
    arch_end = min(end, today - timedelta(days=6))
    frames = []
    if start <= arch_end:
        frames.append(_get(ARCHIVE, ap, start, arch_end, HOURLY))
    fc_start = max(start, arch_end + timedelta(days=1))
    if fc_start <= end:
        frames.append(_get(FORECAST, ap, fc_start, end, HOURLY + ",visibility"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def update_weather(start, end, airports=None):
    frames = []
    for ap in (airports or AIRPORT_COORD):
        try:
            df = fetch_airport(ap, start, end)
            frames.append(df)
            log.info("[weather %s] %d giờ (%s -> %s)", ap, len(df), start, end)
        except Exception as e:
            log.error("[weather %s] lỗi: %s", ap, e)
    if not frames:
        return None
    new = pd.concat(frames, ignore_index=True)
    if FILE_WEATHER.exists():
        old = pd.read_csv(FILE_WEATHER, parse_dates=["time"])
        new = pd.concat([old, new], ignore_index=True)
    new = new.drop_duplicates(["airport", "time"], keep="last").sort_values(["airport", "time"])
    FILE_WEATHER.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(FILE_WEATHER, index=False)
    log.info("Đã lưu %d dòng thời tiết -> %s", len(new), FILE_WEATHER)
    return new


def update_from_raw():
    """Đọc ngày bay nhỏ nhất/lớn nhất trong data/raw rồi tải thời tiết phủ khoảng đó."""
    files = glob.glob(str(RAW_DIR / "*.csv"))
    if not files:
        print("Chưa có dữ liệu thô.")
        return
    d = pd.concat([pd.read_csv(f, dtype=str, usecols=["flight_date"], keep_default_na=False) for f in files])
    d = pd.to_datetime(d["flight_date"], errors="coerce").dropna()
    if d.empty:
        print("Không có flight_date hợp lệ.")
        return
    update_weather(d.min().date(), d.max().date())
