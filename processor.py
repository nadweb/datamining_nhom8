# -*- coding: utf-8 -*-
import glob
import re
import unicodedata
from itertools import combinations
import numpy as np
import pandas as pd
from config import (RAW_DIR, FILE_CLEANED, FILE_RULES, FILE_WEATHER, VN_AIRLINES, CITY_IATA,
                    OFFICIAL_STATS)

try:
    from mlxtend.frequent_patterns import fpgrowth
except ImportError:  # chưa cài mlxtend -> dùng Apriori tự viết bên dưới (kết quả tương đương)
    fpgrowth = None

# Ngưỡng khai phá cho từng lớp đích. Lớp "hủy" hiếm nên support thấp hơn.
TARGETS = {
    "Status=delayed":   dict(min_sup=0.01,  min_conf=0.25, min_lift=1.2),
    "Status=cancelled": dict(min_sup=0.002, min_conf=0.03, min_lift=1.5),
}
MAX_LEN = 4  # tối đa 4 item (3 điều kiện + 1 nhãn)
LABELS = ["on_time", "delayed", "cancelled"]
K1 = ["airport", "direction", "flight_date", "flight_no", "scheduled_time"]


# ============================================================ THỜI GIAN / TRỄ
def parse_time_min(t_str):
    if not t_str or pd.isna(t_str):
        return None, 0
    m = re.search(r"(\d{1,2}):(\d{2})\s*([+-]?)", str(t_str))
    if not m:
        return None, 0
    h, mins, sign = int(m.group(1)), int(m.group(2)), m.group(3)
    return h * 60 + mins, {"-": -1, "+": 1, "": 0}[sign]


def calculate_delay(sched, est, actual, status_str):
    m = re.search(r"(\d+)\s*minutes?\s*late", str(status_str), re.IGNORECASE)
    if m:
        return float(m.group(1))
    target = actual if (actual and str(actual).strip()) else est
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


def standardize_status(row, delay_threshold=15):
    st_raw = str(row.get("status", "")).strip().lower()
    delay = row.get("delay_min", np.nan)
    if re.search(r"cancel|huy|hủy", st_raw):
        return "cancelled"
    if not st_raw or st_raw in {"scheduled", "(trống)"}:
        return "unknown"
    is_done = bool(re.search(r"arrive|depart|land|hacan|catcan|on-time|ontime", st_raw))
    if pd.notna(delay):
        if delay >= delay_threshold:
            return "delayed"
        elif delay >= -60:
            return "on_time"
        return "unknown"
    if is_done:
        return "delayed" if "late" in st_raw else "on_time"
    return "unknown"


# ============================================================ KHÓA CẤP CHUYẾN BAY
def _strip(s):
    s = unicodedata.normalize("NFD", str(s or ""))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").replace("đ", "d").replace("Đ", "D")
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def to_iata(s):
    """'Ho Chi Minh City (SGN)' -> SGN ; 'SGN' -> SGN ; 'TP. Hồ Chí Minh' -> SGN ; không nhận ra -> ''."""
    u = str(s or "").upper()
    m = re.search(r"\(([A-Z]{3})\)", u)
    if m:
        return m.group(1)
    if re.fullmatch(r"\s*[A-Z]{3}\s*", u):
        return u.strip()
    n = _strip(s)
    for k in sorted(CITY_IATA, key=len, reverse=True):
        if n == k or (len(k) >= 5 and k in n):
            return CITY_IATA[k]
    return ""


def add_leg_key(df):
    """Mỗi chuyến bay thật (origin->dest) chỉ có 1 khóa, dù xuất hiện ở bảng đi và bảng đến.
    Nếu không suy ra được sân bay còn lại thì rơi về khóa cũ (cùng phía) để không gộp nhầm."""
    df["other_code"] = df["other_airport"].map(to_iata)
    dep = df["direction"] == "DEPARTURES"
    origin = np.where(dep, df["airport"], df["other_code"])
    dest = np.where(dep, df["other_code"], df["airport"])
    ok = df["other_code"] != ""
    df["leg_key"] = np.where(ok,
                             df["flight_no"] + "|" + df["flight_date"] + "|" + origin + ">" + dest,
                             df["flight_no"] + "|" + df["flight_date"] + "|" + df["airport"] + "|" + df["direction"])
    df["leg_matched"] = ok
    return df


def normalize_flight_no(s):
    s = s.str.replace(r"\s+", "", regex=True)
    return s.str.replace(r"^([A-Z0-9]{2})0+(\d)", r"\1\2", regex=True)


# ============================================================ ĐẶC TRƯNG PHỤ
def add_time_features(df):
    hm = df["scheduled_time"].str.extract(r"(\d{1,2}):(\d{2})").astype(float)
    dt = pd.to_datetime(df["flight_date"], errors="coerce") + pd.to_timedelta(hm[0] * 60 + hm[1], unit="m")
    df["sched_dt"] = dt
    df["hour_floor"] = dt.dt.floor("h")
    df["hour"] = dt.dt.hour

    def slot(h):
        if pd.isna(h):
            return None
        h = int(h)
        return "00h-06h" if h < 6 else "06h-12h" if h < 12 else "12h-18h" if h < 18 else "18h-24h"

    df["time_slot"] = df["hour"].map(slot)
    wd = dt.dt.dayofweek
    df["day_type"] = np.where(wd.isna(), None, np.where(wd >= 5, "weekend", "weekday"))
    try:
        import holidays
        vn = holidays.country_holidays("VN", years=sorted(dt.dt.year.dropna().unique().astype(int)))
        df["is_holiday"] = dt.dt.date.map(lambda d: d in vn if pd.notna(d) else False)
    except Exception:
        df["is_holiday"] = False
    # Mật độ: số chuyến cùng sân bay + cùng giờ (đếm trên dữ liệu đã khử trùng)
    df["airport_load"] = df.groupby(["airport", "hour_floor"])["flight_no"].transform("count")
    return df


def attach_weather(df):
    if not FILE_WEATHER.exists():
        print("(!) Chưa có data/weather_hourly.csv -> bỏ qua thời tiết. Chạy: python main.py weather")
        return df
    wx = pd.read_csv(FILE_WEATHER, parse_dates=["time"])
    for c in ["precipitation", "wind_gusts_10m", "visibility", "cloud_cover", "weather_code"]:
        if c not in wx.columns:
            wx[c] = np.nan
    base = wx.rename(columns={"time": "hour_floor"})
    df = df.merge(base, on=["airport", "hour_floor"], how="left")
    oth = base.add_prefix("dst_").rename(columns={"dst_hour_floor": "hour_floor", "dst_airport": "other_code"})
    return df.merge(oth, on=["other_code", "hour_floor"], how="left")


def _cut(s, bins, labels):
    return pd.cut(s, bins=bins, labels=labels, right=False).astype(object)


# ============================================================ CHUẨN BỊ DỮ LIỆU
def prepare_dataset(prefer_side="DEPARTURES", min_age_hours=0, unit="leg"):
    files = glob.glob(str(RAW_DIR / "*.csv"))
    if not files:
        print("Chưa có file CSV thô nào trong thư mục data/raw!")
        return None
    df = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in files], ignore_index=True)
    stats = {"snapshots": len(df)}

    for c in ["airport", "direction", "flight_date", "flight_no", "scheduled_time", "status", "airline_code"]:
        df[c] = df[c].astype(str).str.strip().str.upper()
    df["flight_no"] = normalize_flight_no(df["flight_no"])
    df = df[df["airline_code"].isin(VN_AIRLINES)].copy()
    df["delay_min"] = df.apply(lambda r: calculate_delay(r["scheduled_time"], r.get("estimated_time", ""),
                                                         r.get("actual_time", ""), r.get("status", "")), axis=1)
    df = add_leg_key(df)

    # Chỉ giữ chuyến đã bay đủ lâu so với lần cào cuối (tránh nhãn "trễ" mới là dự kiến, chưa chốt)
    if min_age_hours:
        hm = df["scheduled_time"].str.extract(r"(\d{1,2}):(\d{2})").astype(float)
        sdt = pd.to_datetime(df["flight_date"], errors="coerce") + pd.to_timedelta(hm[0] * 60 + hm[1], unit="m")
        cutoff = pd.to_datetime(df["scraped_at"], errors="coerce").max() - pd.Timedelta(hours=min_age_hours)
        keep = sdt.isna() | (sdt <= cutoff)
        stats["dropped_too_recent"] = int((~keep).sum())
        df = df[keep].copy()
    stats["min_age_hours"] = min_age_hours

    df["has_info"] = (~df["status"].isin(["", "(TRỐNG)", "SCHEDULED"])).astype(int)
    df["is_pref_side"] = (df["direction"] == prefer_side).astype(int)

    # Bước 1: khử trùng trong cùng một phía (cùng sân bay + hướng + chuyến + giờ), đối chiếu đa nguồn
    k1 = K1
    df = df.sort_values(k1 + ["has_info", "scraped_at"])
    # Bản ghi tốt nhất của TỪNG nguồn cho mỗi sự kiện: dùng để đo độ nhất quán giữa các nguồn
    per_src = df.drop_duplicates(k1 + ["source"], keep="last")
    src1 = df.groupby(k1)["source"].nunique().rename("source_recorded_count")
    side = df.drop_duplicates(k1, keep="last").merge(src1, on=k1, how="left")
    stats["after_side_dedup"] = len(side)

    # Bước 2: khử trùng cấp chuyến bay (HAN-dep và SGN-arr của cùng 1 chuyến -> 1 mẫu)
    side = side.sort_values(["leg_key", "has_info", "is_pref_side", "scraped_at"])
    legs_cnt = side.groupby("leg_key")["direction"].nunique().rename("n_sides_seen")
    if unit == "event":
        # Mỗi sự kiện tại một sân bay (cất cánh ở A, hạ cánh ở B) là một mẫu: độ trễ lúc cất và lúc hạ là hai phép đo khác nhau
        leg = side.merge(legs_cnt, on="leg_key", how="left")
    else:
        leg = side.drop_duplicates("leg_key", keep="last").merge(legs_cnt, on="leg_key", how="left")
    stats["unit"] = unit
    stats["after_leg_dedup"] = len(leg)
    stats["legs_seen_both_sides"] = int((leg["n_sides_seen"] > 1).sum())

    unm = side[~side["leg_matched"]]
    stats["unmatched_side"] = len(unm)
    stats["unmapped_top"] = unm["other_airport"].str.strip().replace("", "(trống)").value_counts().head(20)

    leg = add_time_features(leg)
    leg = attach_weather(leg)
    return side, leg, stats, per_src


# ============================================================ RỜI RẠC HÓA -> ITEM
def build_items(v):
    cols = {
        "Airline": "Airline=" + v["airline_code"],
        "Airport": "Airport=" + v["airport"],
        "Dir": "Dir=" + v["direction"].str.lower(),
        "Slot": "Slot=" + v["time_slot"].astype(object),
        "Day": "Day=" + v["day_type"].astype(object),
        "Holiday": pd.Series(np.where(v["is_holiday"], "Holiday=yes", None), index=v.index),
        "Load": "Load=" + _cut(v["airport_load"], [0, 8, 15, 1e9], ["low", "mid", "high"]),
        "Status": "Status=" + v["status_clean"],
    }
    if "precipitation" in v.columns:
        cols["Rain"] = "Rain=" + _cut(v["precipitation"], [0, 0.1, 2.5, 1e9], ["none", "light", "heavy"])
        cols["Gust"] = "Gust=" + _cut(v["wind_gusts_10m"], [0, 30, 50, 1e9], ["<30", "30-50", ">50"])
        cols["Vis"] = "Vis=" + _cut(v["visibility"], [0, 5000, 1e9], ["<5km", ">=5km"])
        cols["Storm"] = pd.Series(np.where(v["weather_code"] >= 95, "Storm=yes", None), index=v.index)
        if "dst_precipitation" in v.columns:
            cols["DstRain"] = "DstRain=" + _cut(v["dst_precipitation"], [0, 0.1, 2.5, 1e9], ["none", "light", "heavy"])
    parts = [pd.get_dummies(s.where(s.notna(), None)).astype(bool) for s in cols.values()]
    return pd.concat(parts, axis=1)


# ============================================================ KHAI PHÁ LUẬT
def _apriori(items, min_sup, max_len):
    """Apriori gọn (dự phòng khi chưa có mlxtend). Trả DataFrame [support, itemsets]."""
    X, names, n = items.values, list(items.columns), len(items)
    cur = {frozenset([c]): X[:, i] for i, c in enumerate(names) if X[:, i].mean() >= min_sup}
    out = {k: v.mean() for k, v in cur.items()}
    k = 1
    while cur and k < max_len:
        nxt, keys = {}, list(cur)
        for a, b in combinations(keys, 2):
            u = a | b
            if len(u) != k + 1 or u in nxt:
                continue
            m = cur[a] & cur[b]
            if m.mean() >= min_sup:
                nxt[u] = m
        out.update({u: m.mean() for u, m in nxt.items()})
        cur, k = nxt, k + 1
    return pd.DataFrame({"support": list(out.values()), "itemsets": list(out.keys())})


def mine_rules(items, target, min_sup, min_conf, min_lift):
    others = [c for c in items.columns if c.startswith("Status=") and c != target]
    it = items.drop(columns=others)
    if target not in it.columns or it[target].mean() < min_sup:
        return pd.DataFrame()
    fi = (fpgrowth(it, min_support=min_sup, use_colnames=True, max_len=MAX_LEN)
          if fpgrowth else _apriori(it, min_sup, MAX_LEN))
    sup = dict(zip(fi["itemsets"], fi["support"]))
    n, sup_c = len(it), sup[frozenset([target])]
    conf_of = {fs - {target}: s / sup[fs - {target}] for fs, s in sup.items()
               if target in fs and len(fs) > 1 and (fs - {target}) in sup}
    rows = []
    for ante, conf in conf_of.items():
        s = conf * sup[ante]
        lift = conf / sup_c
        if not (s >= min_sup and conf >= min_conf and lift >= min_lift):
            continue
        # loại luật dư thừa: điều kiện con đã cho confidence tương đương
        if any(conf_of.get(frozenset(sub), 0) >= conf - 0.01
               for r in range(1, len(ante)) for sub in combinations(ante, r)):
            continue
        rows.append({"Antecedent": " & ".join(sorted(ante)), "Consequent": target,
                     "Count": int(round(s * n)), "Support (%)": round(s * 100, 2),
                     "Confidence (%)": round(conf * 100, 2), "Lift": round(lift, 2),
                     "Leverage": round(s - sup[ante] * sup_c, 4),
                     "Conviction": round((1 - sup_c) / (1 - conf), 2) if conf < 1 else float("inf")})
    return pd.DataFrame(rows).sort_values(["Lift", "Support (%)"], ascending=False) if rows else pd.DataFrame()


# ============================================================ ĐỐI CHIẾU NGUỒN / HÃNG
def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def source_agreement(per_src, thr):
    """So nhãn giữa các NGUỒN KHÁC NHAU cho cùng sân bay + hướng + chuyến + giờ."""
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
        rows.append({"Nguồn A": a, "Nguồn B": b, "Số sự kiện chung": len(both),
                     "Trùng nhãn (%)": round((both[a] == both[b]).mean() * 100, 1),
                     "Kappa": round(cohen_kappa(both[a], both[b]), 3),
                     "Lệch trễ trung vị (phút)": round(float(d.median()), 1) if len(d) else np.nan})
    return pd.DataFrame(rows)


def airline_table(valid):
    """Tỷ lệ theo hãng, kèm số chính thức (OFFICIAL_STATS) nếu có. OTP = đúng giờ / tất cả chuyến hợp lệ."""
    rows = []
    for al, g in list(valid.groupby("airline_code")) + [("ALL", valid)]:
        n = len(g)
        vc = g["status_clean"].value_counts()
        off = OFFICIAL_STATS.get(al, {})
        rows.append({"Hãng": al, "Số chuyến": n,
                     "Trễ (%)": round(vc.get("delayed", 0) / n * 100, 1),
                     "Hủy (%)": round(vc.get("cancelled", 0) / n * 100, 2),
                     "OTP (%)": round(vc.get("on_time", 0) / n * 100, 1),
                     "Chính thức: trễ": off.get("delay", ""), "Chính thức: hủy": off.get("cancel", ""),
                     "Chính thức: OTP": off.get("otp", "")})
    return pd.DataFrame(rows).sort_values("Số chuyến", ascending=False)


def print_diagnostics(stats):
    print("-" * 62)
    print("CHẨN ĐOÁN SÂN BAY CHƯA NHẬN RA (other_airport không ra mã IATA)")
    print(f"Số bản ghi cùng phía không ghép được khóa cấp chuyến: {stats['unmatched_side']:,} / {stats['after_side_dedup']:,}")
    if stats["unmatched_side"]:
        print("Giá trị hay gặp nhất (thêm vào CITY_IATA trong config.py, dạng bỏ dấu, viết thường, bỏ khoảng trắng):")
        print(stats["unmapped_top"].to_string())
    if stats.get("min_age_hours"):
        print(f"Đã loại {stats.get('dropped_too_recent', 0):,} dòng thô có giờ bay cách lần cào cuối < {stats['min_age_hours']}h")
    print("-" * 62)


# ============================================================ PIPELINE
def label_and_report(side, leg, stats, thr, per_src=None):
    side = side.copy()
    leg = leg.copy()
    side["status_clean"] = side.apply(standardize_status, axis=1, delay_threshold=thr)
    leg["status_clean"] = leg.apply(standardize_status, axis=1, delay_threshold=thr)
    valid = leg[leg["status_clean"].isin(["on_time", "delayed", "cancelled"])].copy()

    # Độ nhất quán nhãn giữa các bản ghi của cùng một chuyến (phía đi vs phía đến / nguồn khác nhau)
    sv = side[side["status_clean"].isin(["on_time", "delayed", "cancelled"])]
    g = sv.groupby("leg_key")["status_clean"].agg(["nunique", "size"])
    multi = g[g["size"] > 1]
    agree = (multi["nunique"] == 1).mean() * 100 if len(multi) else float("nan")

    print("=" * 62)
    print(f"BÁO CÁO TÍCH HỢP DỮ LIỆU (ngưỡng trễ = {thr} phút)")
    print("=" * 62)
    print(f"Snapshot thô:                               {stats['snapshots']:,}")
    print(f"Sau khử trùng cùng phía:                    {stats['after_side_dedup']:,}")
    unit_txt = "1 sự kiện sân bay=1" if stats.get("unit") == "event" else "1 chuyến=1"
    print(f"Đơn vị mẫu:                                 {'sự kiện (cất/hạ cánh tính riêng)' if stats.get('unit') == 'event' else 'chuyến bay (gộp phía đi và đến)'}")
    print(f"Sau khử trùng ({unit_txt}):".ljust(44) + f"{stats['after_leg_dedup']:,}")
    print(f"  - chuyến thấy ở cả bảng đi và bảng đến:   {stats['legs_seen_both_sides']:,}")
    print(f"  - sự kiện có >=2 nguồn (cùng phía):       {(leg['source_recorded_count'] > 1).sum():,}")
    print(f"Mẫu hợp lệ cho khai phá:                    {len(valid):,} / 10,000")
    print(f"Nhất quán nhãn giữa các bản ghi cùng chuyến: {agree:.1f}% ({len(multi):,} chuyến có >=2 bản ghi)")
    print("-" * 62)
    print(valid["status_clean"].value_counts().to_string())
    print(f"Chuyến bị loại vì chưa có nhãn cuối (unknown): {len(leg) - len(valid):,} / {len(leg):,}")
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
    print(f"Số item: {items.shape[1]} | Giao dịch: {len(items):,} | "
          f"Thuật toán: {'FP-Growth (mlxtend)' if fpgrowth else 'Apriori nội bộ'}")
    res = [mine_rules(items, t, **p) for t, p in TARGETS.items()]
    res = [r for r in res if not r.empty]
    return valid, (pd.concat(res, ignore_index=True) if res else pd.DataFrame())


def process_and_mine(threshold=15, compare=None, min_age_hours=0, unit="leg"):
    prep = prepare_dataset(min_age_hours=min_age_hours, unit=unit)
    if prep is None:
        return None
    side, leg, stats, per_src = prep
    print_diagnostics(stats)
    valid, rules = label_and_report(side, leg, stats, threshold, per_src)
    valid.to_csv(FILE_CLEANED, index=False, encoding="utf-8-sig")
    if not rules.empty:
        rules.to_csv(FILE_RULES, index=False, encoding="utf-8-sig")
        print("\nTOP 10 LUẬT (sắp theo Lift):")
        print(rules.head(10).to_string(index=False))
        print(f"\n=> Đã xuất: {FILE_RULES}")
    else:
        print("\nKhông có luật nào qua ngưỡng. Thử giảm min_sup trong TARGETS.")
    if compare:
        _, r2 = label_and_report(side, leg, stats, compare, per_src)
        if not r2.empty:
            print(f"\nTOP 10 LUẬT ở ngưỡng {compare} phút:")
            print(r2.head(10).to_string(index=False))
        if not r2.empty:
            p = FILE_RULES.with_name(f"mined_association_rules_{compare}min.csv")
            r2.to_csv(p, index=False, encoding="utf-8-sig")
            print(f"=> Luật ở ngưỡng {compare} phút: {p}")
    return valid, rules
