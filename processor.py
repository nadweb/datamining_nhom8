# -*- coding: utf-8 -*-
"""Tích hợp -> khử trùng -> gắn thời tiết -> gán nhãn -> khai phá luật kết hợp (FP-Growth).

Đầu vào : data/debug_data/{playwright,aviationstack,open_meteo}/*.json  -> data/raw/*.csv (mỗi nguồn giữ cột riêng, dựng lại từ JSON)
Đầu ra  : data/flights_cleaned_global.csv  (schema = config.CLEANED_FIELDS, 1 dòng = 1 chuyến)
          data/mined_association_rules*.csv
"""
import glob
import re
from itertools import combinations
import numpy as np
import pandas as pd
from config import (
    AIRPORT_COORD, CLEANED_FIELDS, DEBUG_DIR, FIELDS, FILE_CLEANED, FILE_RULES,
    CITY_IATA, OFFICIAL_STATS, RAW_DIR, VN_AIRLINES, VN_AIRPORTS,
)
from collector import load_standard_records, norm, rebuild_raw_csv
from weather import WX_COLS, load_weather

TARGETS = {
    "Status=delayed":   dict(min_sup=0.01,  min_conf=0.25, min_lift=1.2),
    "Status=cancelled": dict(min_sup=0.002, min_conf=0.03, min_lift=1.5),
}
MAX_LEN = 4
LABELS = ["on_time", "delayed", "cancelled"]
K1 = ["airport", "direction", "flight_date", "flight_no", "scheduled_time"]
EARLY_LIMIT = -90          # bay sớm hơn 90' so với lịch -> coi là dữ liệu bất thường
MISSING_WARN = 40.0        # % thiếu vượt ngưỡng này -> cảnh báo cân nhắc bỏ cột


# ============================================================ THỜI GIAN / TRỄ / TRẠNG THÁI
def parse_time_min(t_str):
    if not t_str or pd.isna(t_str):
        return None, 0
    m = re.search(r"(\d{1,2}):(\d{2})\s*([+-]?)", str(t_str))
    if not m:
        return None, 0
    h, mins, sign = int(m.group(1)), int(m.group(2)), m.group(3)
    return h * 60 + mins, {"-": -1, "+": 1, "": 0}[sign]


def calculate_delay(sched, est, actual, status_str):
    st = str(status_str or "")
    # Bắt cả "X minutes late" lẫn "Delayed X minutes"
    m_late = re.search(r"(?:(\d+)\s*minutes?\s*late|delayed\s*(\d+)\s*minutes?)", st, re.IGNORECASE)
    if m_late:
        return float(m_late.group(1) or m_late.group(2))
    m_early = re.search(r"(\d+)\s*minutes?\s*early", st, re.IGNORECASE)
    if m_early:
        return -float(m_early.group(1))

    target = actual if (actual and str(actual).strip()) else est
    s_min, _ = parse_time_min(sched)
    t_min, off = parse_time_min(target)
    if s_min is None or t_min is None:
        return np.nan
    diff = (t_min + off * 1440) - s_min
    if off == 0:                       # xử lý qua nửa đêm
        if diff < -720:
            diff += 1440
        elif diff > 720:
            diff -= 1440
    return float(diff)


def _s(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip()


def label_status(row, delay_threshold=15):
    """Trả về (status, status_detail, status_clean).

    status_clean  : on_time | delayed | cancelled | unknown   (nhãn đích; unknown bị loại khỏi khai phá)
    status_detail : departed_on_time | departed_late | arrived_on_time | arrived_late   (đã bay/hạ cánh)
                    on_time | delayed   (chưa bay, theo giờ dự kiến) | cancelled | diverted | scheduled | unknown
    status        : chuỗi chuẩn, nhất quán với nhãn:
                    "Departed On-Time", "Departed N minutes late", "Arrived On-Time", "Arrived N minutes late",
                    "Delayed N minutes", "On-Time", "Cancelled", ...
    Số phút trễ chính xác luôn nằm ở delay_min; ngưỡng quyết định on_time/delayed là delay_threshold.
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

    # 1. Hủy / chuyển hướng
    if re.search(r"cancel|huy|hủy", st):
        return "Cancelled", "cancelled", "cancelled"
    if "divert" in st:
        return "Diverted", "diverted", "unknown"

    # 2. Đã bay / đã hạ cánh (ưu tiên chữ trong status, sau đó mới dựa vào actual_time + hướng)
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
        if late_txt:                                   # có chữ trễ nhưng không có số phút
            return st_raw, f"{tag}_late", "delayed"
        if ontime_txt:
            return f"{prefix} On-Time", f"{tag}_on_time", "on_time"
        return st_raw, "unknown", "unknown"            # "Departed" trơn, không có giờ -> không đoán nhãn

    # 3. Chưa bay
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


def standardize_status(row, delay_threshold=15):
    return label_status(row, delay_threshold)[2]


def apply_labels(df, thr):
    df = df.copy()
    res = df.apply(label_status, axis=1, delay_threshold=thr)
    df["status"] = [r[0] for r in res]
    df["status_detail"] = [r[1] for r in res]
    df["status_clean"] = [r[2] for r in res]
    df.loc[df["status_clean"] == "cancelled", "delay_min"] = np.nan    # chuyến hủy không có số phút trễ
    return df


# ============================================================ KHÓA CẤP CHUYẾN BAY
def to_iata(s):
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


def add_leg_key(df):
    """Suy ra origin/destination/route_type và khóa chuyến bay (flight_no|ngày|đi>đến)."""
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


def normalize_flight_no(s):
    s = s.str.replace(r"\s+", "", regex=True)
    return s.str.replace(r"^([A-Z0-9]{2})0+(\d)", r"\1\2", regex=True)


# ============================================================ ĐẶC TRƯNG THỜI GIAN & THỜI TIẾT
def _cut(s, bins, labels):
    return pd.cut(s, bins=bins, labels=labels, right=False).astype(object)


def sched_datetime(df):
    hm = df["scheduled_time"].str.extract(r"(\d{1,2}):(\d{2})").astype(float)
    return pd.to_datetime(df["flight_date"], errors="coerce") + pd.to_timedelta(hm[0] * 60 + hm[1], unit="m")


def compute_airport_load(df):
    """Số chuyến (MỌI hãng, cả đi & đến) có giờ dự kiến trong cùng giờ tại sân bay.
    Chỉ tính cho 5 sân bay được cào đầy đủ; sân bay khác để trống thay vì cho số sai."""
    u = df.drop_duplicates(K1)
    u = u[u["airport"].isin(AIRPORT_COORD) & u["hour_floor"].notna()]
    return u.groupby(["airport", "hour_floor"]).size().rename("airport_load").reset_index()


def add_time_features(df, load):
    dt = df["sched_dt"]
    df["time_slot"] = _cut(dt.dt.hour, [0, 6, 12, 18, 24], ["00h-06h", "06h-12h", "12h-18h", "18h-24h"])
    wd = dt.dt.dayofweek
    df["day_type"] = np.where(wd.isna(), None, np.where(wd >= 5, "weekend", "weekday"))
    try:
        import holidays
        vn = holidays.country_holidays("VN", years=sorted(dt.dt.year.dropna().unique().astype(int)))
        df["is_holiday"] = dt.dt.date.map(lambda d: d in vn if pd.notna(d) else False)
    except ImportError:
        print("(!) Chưa cài thư viện 'holidays' -> is_holiday = False toàn bộ. Chạy: pip install holidays")
        df["is_holiday"] = False
    return df.merge(load, on=["airport", "hour_floor"], how="left")


def attach_weather(df):
    """Gộp thời tiết (từ JSON gốc open_meteo + cache CSV) vào từng chuyến theo (sân bay quan sát, giờ bay dự kiến)."""
    derived = ["rain_level", "wind_level", "vis_level"]
    wx = load_weather()
    if wx.empty:
        print("(!) Chưa có dữ liệu thời tiết -> các cột thời tiết để trống. Chạy: python main.py weather")
        for c in WX_COLS + derived:
            df[c] = np.nan
        return df

    base = wx[["airport", "time"] + WX_COLS].rename(columns={"time": "hour_floor"})
    df = df.merge(base, on=["airport", "hour_floor"], how="left")
    df["rain_level"] = _cut(df["precipitation"], [0, 0.1, 2.5, 1e9], ["none", "light", "heavy"])
    df["wind_level"] = _cut(df["wind_gusts_10m"], [0, 30, 50, 1e9], ["<30", "30-50", ">50"])
    df["vis_level"] = _cut(df["visibility"], [0, 5000, 1e9], ["<5km", ">=5km"])
    return df


# ============================================================ CHUẨN BỊ DỮ LIỆU
def prepare_dataset(prefer_side="DEPARTURES", min_age_hours=0, unit="leg"):
    rebuild_raw_csv()                            # data/raw luôn được dựng lại từ JSON gốc
    std = load_standard_records()                # tích hợp: cột riêng của từng nguồn -> schema chuẩn FIELDS
    frames = [std] if not std.empty else []

    if not frames:
        print("Chưa có dữ liệu trong data/debug_data hay data/raw!")
        return None

    df = pd.concat(frames, ignore_index=True)
    stats = {"snapshots": len(df), "from_debug_json": len(df)}

    for c in ["airport", "direction", "flight_date", "flight_no", "scheduled_time", "airline_code"]:
        df[c] = df[c].astype(str).str.strip().str.upper()
    df["status"] = df["status"].astype(str).str.strip()

    # Chuyến đã hoàn thành mà actual_time trống -> lấy estimated_time
    done = df["status"].str.contains(r"arrive|depart|land|hacan|catcan", case=False, regex=True, na=False)
    empty_act = df["actual_time"].astype(str).str.strip().isin(["", "nan", "None"])
    has_est = ~df["estimated_time"].astype(str).str.strip().isin(["", "nan", "None"])
    df.loc[done & empty_act & has_est, "actual_time"] = df.loc[done & empty_act & has_est, "estimated_time"]

    df["flight_no"] = normalize_flight_no(df["flight_no"])
    df["sched_dt"] = sched_datetime(df)
    df["hour_floor"] = df["sched_dt"].dt.floor("h")

    # Tải sân bay tính trên MỌI hãng (trước khi lọc hãng Việt Nam)
    load = compute_airport_load(df)

    df = df[df["airline_code"].isin(VN_AIRLINES)].copy()
    df["delay_min"] = df.apply(
        lambda r: calculate_delay(r["scheduled_time"], r["estimated_time"], r["actual_time"], r["status"]), axis=1
    )
    df = add_leg_key(df)

    if min_age_hours:
        cutoff = pd.to_datetime(df["scraped_at"], errors="coerce").max() - pd.Timedelta(hours=min_age_hours)
        keep = df["sched_dt"].isna() | (df["sched_dt"] <= cutoff)
        stats["dropped_too_recent"] = int((~keep).sum())
        df = df[keep].copy()
    stats["min_age_hours"] = min_age_hours

    df["has_done_act"] = (df["actual_time"].astype(str).str.strip() != "").astype(int)
    df["has_info"] = (~df["status"].str.upper().isin(["", "SCHEDULED"])).astype(int)
    df["is_pref_side"] = (df["direction"] == prefer_side).astype(int)
    df["at_monitored"] = df["airport"].isin(AIRPORT_COORD).astype(int)   # phía có dữ liệu thời tiết/tải sân bay

    # Bước 1: khử trùng cùng phía (giữ bản ghi nhiều thông tin & mới nhất)
    df = df.sort_values(K1 + ["has_info", "has_done_act", "scraped_at"])
    per_src = df.drop_duplicates(K1 + ["source"], keep="last")
    src1 = df.groupby(K1)["source"].nunique().rename("source_recorded_count")
    side = df.drop_duplicates(K1, keep="last").merge(src1, on=K1, how="left")
    stats["after_side_dedup"] = len(side)

    # Bước 2: khử trùng cấp chuyến bay (1 chuyến thấy ở cả bảng đi và bảng đến -> 1 dòng)
    # Ưu tiên: có thông tin > có giờ thực tế > phía ở sân bay được theo dõi (có thời tiết) > phía khởi hành > mới nhất
    side = side.sort_values(["leg_key", "has_info", "has_done_act", "at_monitored", "is_pref_side", "scraped_at"])
    legs_cnt = side.groupby("leg_key")["direction"].nunique().rename("n_sides_seen")
    if unit == "event":
        leg = side.merge(legs_cnt, on="leg_key", how="left")
    else:
        leg = side.drop_duplicates("leg_key", keep="last").merge(legs_cnt, on="leg_key", how="left")
    stats["unit"] = unit
    stats["after_leg_dedup"] = len(leg)
    stats["legs_seen_both_sides"] = int((leg["n_sides_seen"] > 1).sum())

    unm = side[~side["leg_matched"]]
    stats["unmatched_side"] = len(unm)
    stats["unmapped_top"] = unm["other_airport"].str.strip().replace("", "(trống)").value_counts().head(20)

    leg = add_time_features(leg, load)
    leg = attach_weather(leg)
    return side, leg, stats, per_src


# ============================================================ RỜI RẠC HÓA -> ITEM
def _load_level(v):
    """Mức tải tương đối so với chính sân bay đó (tertile theo từng sân bay)."""
    pct = v.groupby("airport")["airport_load"].rank(pct=True)
    return pd.Series(np.select([pct <= 1 / 3, pct <= 2 / 3, pct > 2 / 3], ["low", "mid", "high"], None), index=v.index)


def build_items(v):
    cols = {
        "Airline": "Airline=" + v["airline_code"],
        "Airport": "Airport=" + v["airport"],
        "Dir": "Dir=" + v["direction"].str.lower(),
        "Route": "Route=" + v["route_type"].replace("", np.nan),
        "Slot": "Slot=" + v["time_slot"].astype(object),
        "Day": "Day=" + v["day_type"].astype(object),
        "Holiday": pd.Series(np.where(v["is_holiday"], "Holiday=yes", None), index=v.index),
        "Load": "Load=" + _load_level(v),
        "Status": "Status=" + v["status_clean"],
        "Rain": "Rain=" + v["rain_level"].astype(object),
        "Gust": "Gust=" + v["wind_level"].astype(object),
        "Vis": "Vis=" + v["vis_level"].astype(object),
        "Storm": pd.Series(np.where(v["weather_code"] >= 95, "Storm=yes", None), index=v.index),
    }
    parts = [pd.get_dummies(s.where(s.notna(), None)).astype(bool) for s in cols.values()]
    return pd.concat(parts, axis=1)


# ============================================================ KHAI PHÁ LUẬT
def mine_rules(items, target, min_sup, min_conf, min_lift):
    try:
        from mlxtend.frequent_patterns import fpgrowth
    except ImportError as e:
        raise SystemExit("Thiếu thư viện mlxtend (cần cho FP-Growth). Chạy: pip install mlxtend") from e

    others = [c for c in items.columns if c.startswith("Status=") and c != target]
    it = items.drop(columns=others)
    if target not in it.columns or it[target].mean() < min_sup:
        return pd.DataFrame()
    fi = fpgrowth(it, min_support=min_sup, use_colnames=True, max_len=MAX_LEN)
    sup = dict(zip(fi["itemsets"], fi["support"]))
    n, sup_c = len(it), sup[frozenset([target])]
    conf_of = {
        fs - {target}: s / sup[fs - {target}]
        for fs, s in sup.items()
        if target in fs and len(fs) > 1 and (fs - {target}) in sup
    }
    rows = []
    for ante, conf in conf_of.items():
        s = conf * sup[ante]
        lift = conf / sup_c
        if not (s >= min_sup and conf >= min_conf and lift >= min_lift):
            continue
        if any(                                           # bỏ luật dư thừa (luật con đã đủ tốt)
            conf_of.get(frozenset(sub), 0) >= conf - 0.01
            for r in range(1, len(ante))
            for sub in combinations(ante, r)
        ):
            continue
        rows.append({
            "Antecedent": " & ".join(sorted(ante)),
            "Consequent": target,
            "Count": int(round(s * n)),
            "Support (%)": round(s * 100, 2),
            "Confidence (%)": round(conf * 100, 2),
            "Lift": round(lift, 2),
            "Leverage": round(s - sup[ante] * sup_c, 4),
            "Conviction": round((1 - sup_c) / (1 - conf), 2) if conf < 1 else float("inf"),
        })
    return pd.DataFrame(rows).sort_values(["Lift", "Support (%)"], ascending=False) if rows else pd.DataFrame()


# ============================================================ ĐỐI CHIẾU NGUỒN / HÃNG
def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def source_agreement(per_src, thr):
    ps = per_src.copy()
    ps["lab"] = ps.apply(standardize_status, axis=1, delay_threshold=thr)
    ps = ps[ps["lab"].isin(LABELS)]
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
            "Nguồn A": a,
            "Nguồn B": b,
            "Số sự kiện chung": len(both),
            "Trùng nhãn (%)": round((both[a] == both[b]).mean() * 100, 1),
            "Kappa": round(cohen_kappa(both[a], both[b]), 3),
            "Lệch trễ trung vị (phút)": round(float(d.median()), 1) if len(d) else np.nan,
        })
    return pd.DataFrame(rows)


def airline_table(valid):
    rows = []
    for al, g in list(valid.groupby("airline_code")) + [("ALL", valid)]:
        n = len(g)
        vc = g["status_clean"].value_counts()
        off = OFFICIAL_STATS.get(al, {})
        rows.append({
            "Hãng": al,
            "Số chuyến": n,
            "Trễ (%)": round(vc.get("delayed", 0) / n * 100, 1),
            "Hủy (%)": round(vc.get("cancelled", 0) / n * 100, 2),
            "OTP (%)": round(vc.get("on_time", 0) / n * 100, 1),
            "Chính thức: trễ": off.get("delay", ""),
            "Chính thức: hủy": off.get("cancel", ""),
            "Chính thức: OTP": off.get("otp", ""),
        })
    return pd.DataFrame(rows).sort_values("Số chuyến", ascending=False)


def print_diagnostics(stats):
    print("-" * 62)
    print("CHẨN ĐOÁN SÂN BAY & NGUỒN DỮ LIỆU")
    print(f"Bản ghi trong data/raw (dựng từ JSON gốc): {stats.get('from_debug_json', 0):,}")
    print(f"Số bản ghi cùng phía không ghép được khóa cấp chuyến: {stats['unmatched_side']:,} / {stats['after_side_dedup']:,}")
    if stats["unmatched_side"]:
        print("Giá trị hay gặp nhất:")
        print(stats["unmapped_top"].to_string())
    if stats.get("min_age_hours"):
        print(f"Đã loại {stats.get('dropped_too_recent', 0):,} dòng thô có giờ bay cách lần cào cuối < {stats['min_age_hours']}h")
    print("-" * 62)


def print_missing_report(valid):
    """% thiếu theo cột của bộ dữ liệu cuối - dùng để quyết định cột nào nên bỏ khỏi schema."""
    miss = valid.replace("", np.nan).isna().mean().mul(100).round(1)
    miss = miss[miss > 0].sort_values(ascending=False)
    print("-" * 62)
    print("TỶ LỆ THIẾU THEO CỘT (bộ dữ liệu cuối)")
    if miss.empty:
        print("Không cột nào thiếu dữ liệu.")
    for c, p in miss.items():
        print(f"  {c:<22}{p:>6.1f}%" + ("   <-- thiếu nhiều, cân nhắc bỏ" if p > MISSING_WARN else ""))


# ============================================================ PIPELINE
def label_and_report(side, leg, stats, thr, per_src=None):
    side, leg = apply_labels(side, thr), apply_labels(leg, thr)
    valid = leg[leg["status_clean"].isin(LABELS)].copy()

    # Nhất quán "hủy / không hủy" giữa các bản ghi của cùng một chuyến (phía đi vs phía đến, nguồn này vs nguồn kia)
    sv = side[side["status_clean"].isin(LABELS)].assign(is_cx=lambda d: d["status_clean"].eq("cancelled"))
    g = sv.groupby("leg_key")["is_cx"].agg(["nunique", "size"])
    multi = g[g["size"] > 1]
    agree = (multi["nunique"] == 1).mean() * 100 if len(multi) else float("nan")

    print("=" * 62)
    print(f"BÁO CÁO TÍCH HỢP DỮ LIỆU (ngưỡng trễ = {thr} phút)")
    print("=" * 62)
    print(f"Snapshot thô:                               {stats['snapshots']:,}")
    print(f"Sau khử trùng cùng phía (hãng Việt Nam):    {stats['after_side_dedup']:,}")
    unit_txt = "1 sự kiện sân bay=1" if stats.get("unit") == "event" else "1 chuyến=1"
    print(f"Sau khử trùng ({unit_txt}):".ljust(44) + f"{stats['after_leg_dedup']:,}")
    print(f"  - chuyến thấy ở cả bảng đi và bảng đến:   {stats['legs_seen_both_sides']:,}")
    print(f"  - sự kiện có >=2 nguồn (cùng phía):       {(leg['source_recorded_count'] > 1).sum():,}")
    print(f"Mẫu hợp lệ cho khai phá:                    {len(valid):,} / 10,000")
    print(f"Nhất quán hủy/không hủy giữa các bản ghi:   {agree:.1f}% ({len(multi):,} chuyến có >=2 bản ghi)")
    print("-" * 62)
    print("PHÂN BỐ NHÃN ĐÍCH (status_clean):")
    print(valid["status_clean"].value_counts().to_string())
    print("\nPHÂN BỐ TRẠNG THÁI CHI TIẾT (status_detail):")
    print(valid["status_detail"].value_counts().to_string())
    print(f"\nChuyến bị loại vì chưa có nhãn cuối (unknown): {len(leg) - len(valid):,} / {len(leg):,}")
    print("=" * 62)
    print("TỶ LỆ THEO HÃNG (so với số liệu chính thức)")
    print(airline_table(valid).to_string(index=False))
    if per_src is not None:
        sa = source_agreement(per_src, thr)
        print("-" * 62)
        print("NHẤT QUÁN GIỮA CÁC NGUỒN KHÁC NHAU (cùng sân bay + hướng + chuyến + giờ)")
        print(sa.to_string(index=False) if not sa.empty else "Chưa có cặp nguồn nào đủ >=20 sự kiện chung.")
    print("=" * 62)

    items = build_items(valid)
    print(f"Số item: {items.shape[1]} | Giao dịch: {len(items):,} | Thuật toán: FP-Growth (mlxtend)")
    res = [mine_rules(items, t, **p) for t, p in TARGETS.items()]
    res = [r for r in res if not r.empty]

    # Xuất đúng schema chuẩn: bỏ các cột tạm (sched_dt, hour_floor, other_code, ...)
    out = valid.reindex(columns=CLEANED_FIELDS).replace("", np.nan)
    print_missing_report(out)
    return out, (pd.concat(res, ignore_index=True) if res else pd.DataFrame())


def process_and_mine(threshold=15, compare=None, min_age_hours=0, unit="leg"):
    prep = prepare_dataset(min_age_hours=min_age_hours, unit=unit)
    if prep is None:
        return None
    side, leg, stats, per_src = prep
    print_diagnostics(stats)
    valid, rules = label_and_report(side, leg, stats, threshold, per_src)
    valid.to_csv(FILE_CLEANED, index=False, encoding="utf-8-sig")
    print(f"\n=> Đã xuất bộ dữ liệu sạch ({len(valid):,} dòng, {len(valid.columns)} cột): {FILE_CLEANED}")
    if not rules.empty:
        rules.to_csv(FILE_RULES, index=False, encoding="utf-8-sig")
        print("\nTOP 10 LUẬT KẾT HỢP (sắp theo Lift):")
        print(rules.head(10).to_string(index=False))
        print(f"\n=> Đã xuất tập luật: {FILE_RULES}")
    else:
        print("\nKhông có luật nào qua ngưỡng. Thử giảm min_sup trong TARGETS.")
    if compare:
        _, r2 = label_and_report(side, leg, stats, compare, per_src)
        if not r2.empty:
            p = FILE_RULES.with_name(f"mined_association_rules_{compare}min.csv")
            r2.to_csv(p, index=False, encoding="utf-8-sig")
            print(f"=> Luật ở ngưỡng {compare} phút: {p}")
    return valid, rules
