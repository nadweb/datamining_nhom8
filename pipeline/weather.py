# -*- coding: utf-8 -*-
import sys
from pathlib import Path

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

"""Tải thời tiết theo giờ từ Open-Meteo (miễn phí, không cần key).

- Phản hồi JSON gốc lưu ở data/debug_data/open_meteo/ (nguồn chính cho processor).
- data/weather_hourly.csv chỉ là bản cache gộp từ các JSON đó.

Lưu ý: Archive API (ERA5) KHÔNG có `visibility`; chỉ Forecast API có và nó hỗ trợ
lùi tối đa ~92 ngày. Vì vậy 90 ngày gần nhất lấy từ Forecast API (đủ cột),
chỉ phần cũ hơn mới dùng Archive (visibility sẽ trống).
"""
import json
import logging
from datetime import date, datetime, timedelta
import pandas as pd
import requests
from config import AIRPORT_COORD, DEBUG_DIR, FILE_WEATHER

log = logging.getLogger("weather")
HOURLY = "precipitation,wind_gusts_10m,weather_code"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"
FORECAST_PAST_DAYS = 90      # Forecast API cho phép lùi tối đa 92 ngày
ARCHIVE_LAG_DAYS = 6         # Archive API trễ vài ngày so với hiện tại
WX_COLS = ["precipitation", "wind_gusts_10m", "visibility", "weather_code"]


def save_weather_raw_json(payload, ap, kind, start, end):
    """Lưu phản hồi JSON gốc từ Open-Meteo vào data/debug_data/open_meteo/."""
    try:
        target_dir = DEBUG_DIR / "open_meteo"
        target_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"open_meteo_{ap}_{kind}_{start.isoformat()}_{end.isoformat()}_{ts}.json"
        with open(target_dir / fname, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        log.info("Đã lưu debug JSON thời tiết: open_meteo/%s", fname)
    except Exception as e:
        log.warning("Không thể lưu debug JSON thời tiết (%s): %s", ap, e)


def _get(url, ap, start, end, hourly, kind="archive"):
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
    save_weather_raw_json(raw_json, ap, kind, start, end)
    return _payload_to_df(raw_json, ap)


def _payload_to_df(payload, ap):
    df = pd.DataFrame(payload["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df["airport"] = ap
    df["fetched_at"] = (payload.get("_meta") or {}).get("scraped_at", "")
    return df


def fetch_airport(ap, start, end):
    """Forecast API cho 90 ngày gần nhất (có visibility); Archive cho phần cũ hơn."""
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
        except requests.HTTPError as e:
            # Dự phòng: Forecast API từ chối khoảng quá khứ -> dùng Archive (không có visibility)
            log.warning("[weather %s] Forecast API lỗi (%s) -> dùng Archive dự phòng", ap, e)
            arch_end = min(end, today - timedelta(days=ARCHIVE_LAG_DAYS))
            if fc_start <= arch_end:
                frames.append(_get(ARCHIVE, ap, fc_start, arch_end, HOURLY, kind="archive"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _merge_frames(frames):
    """Gộp nhiều bản tải: với mỗi (sân bay, giờ) lấy giá trị KHÔNG rỗng của bản tải mới nhất
    (nhờ vậy cột visibility từ Forecast không bị bản Archive cũ ghi đè thành rỗng)."""
    wx = pd.concat(frames, ignore_index=True)
    wx["time"] = pd.to_datetime(wx["time"])
    if "fetched_at" not in wx.columns:
        wx["fetched_at"] = ""
    wx["fetched_at"] = wx["fetched_at"].fillna("").astype(str)
    for c in WX_COLS:
        if c not in wx.columns:
            wx[c] = float("nan")
    wx = wx.sort_values("fetched_at", kind="stable")
    out = wx.groupby(["airport", "time"], as_index=False)[WX_COLS + ["fetched_at"]].last()
    return out.sort_values(["airport", "time"]).reset_index(drop=True)


def load_weather_from_debug_json():
    """Đọc toàn bộ file JSON trong data/debug_data/open_meteo/ thành DataFrame."""
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
                parts = p.stem.split("_")      # open_meteo_<AP>_<kind>_...
                ap = parts[2] if len(parts) > 2 else ""
            frames.append(_payload_to_df(data, ap))
        except Exception as e:
            log.warning("Lỗi đọc JSON thời tiết %s: %s", p.name, e)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_weather():
    """Thời tiết theo giờ cho processor: JSON gốc + cache CSV, đã khử trùng."""
    frames = []
    wx_json = load_weather_from_debug_json()
    if not wx_json.empty:
        frames.append(wx_json)
    if FILE_WEATHER.exists():
        cache = pd.read_csv(FILE_WEATHER, parse_dates=["time"])
        if "fetched_at" not in cache.columns:
            cache["fetched_at"] = ""          # cache cũ -> coi là cũ nhất
        frames.append(cache)
    return _merge_frames(frames) if frames else pd.DataFrame()


def update_weather(start, end, airports=None):
    got = False
    for ap in (airports or AIRPORT_COORD):
        try:
            df = fetch_airport(ap, start, end)
            got = got or not df.empty
            log.info("[weather %s] %d giờ (%s -> %s)", ap, len(df), start, end)
        except Exception as e:
            log.error("[weather %s] lỗi: %s", ap, e)
    if not got:
        return None
    wx = load_weather()      # JSON vừa lưu đã nằm trong debug_data -> gộp lại luôn
    FILE_WEATHER.parent.mkdir(parents=True, exist_ok=True)
    wx.to_csv(FILE_WEATHER, index=False)
    log.info("Đã lưu %d dòng thời tiết -> %s", len(wx), FILE_WEATHER)
    return wx


def update_from_raw():
    """Đọc ngày bay nhỏ nhất/lớn nhất trong data/raw rồi tải thời tiết phủ khoảng đó."""
    from pipeline.collector import load_standard_records      # import muộn để tránh phụ thuộc vòng
    df = load_standard_records()
    if df.empty:
        print("Chưa có dữ liệu thô (chạy: python main.py build-raw).")
        return
    d = pd.to_datetime(df["flight_date"], errors="coerce").dropna()
    if d.empty:
        print("Không có flight_date hợp lệ.")
        return
    update_weather(d.min().date(), d.max().date())
