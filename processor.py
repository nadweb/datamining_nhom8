# -*- coding: utf-8 -*-
"""Processor Module (Parts 1-2: Cleaning, Integration & Pipeline Orchestration)
Orchestrates:
1. Rebuilding raw CSVs from debug JSON
2. Integration across airport FIDS and Aviationstack API
3. Cross-source deduplication (leg-level and side-level)
4. Integration with weather and airport load
5. Labeling and data contract validation
6. Association rule mining via FP-Growth
"""
import glob
import hashlib
import re
import sys
from itertools import combinations
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from config import (
    AIRPORT_COORD, CLEANED_FIELDS, DEBUG_DIR, FIELDS, FILE_CLEANED, FILE_RULES,
    OFFICIAL_STATS, RAW_DIR, VN_AIRLINES
)
from pipeline.collector import load_standard_records, rebuild_raw_csv
from pipeline.transformation import (
    EARLY_LIMIT, add_leg_key, add_time_features, apply_labels, attach_weather,
    calculate_delay, compute_airport_load, normalize_flight_no, sched_datetime,
    standardize_status
)
from pipeline.check_data import validate_flights_data

TARGETS = {
    "Status=delayed": dict(min_sup=0.01, min_conf=0.25, min_lift=1.2),
    "Status=cancelled": dict(min_sup=0.002, min_conf=0.03, min_lift=1.5),
}
MAX_LEN = 4
LABELS = ["on_time", "delayed", "cancelled"]
K1 = ["airport", "direction", "flight_date", "flight_no", "scheduled_time"]
MISSING_WARN = 40.0

FORMAT_RULES = {
    "flight_date": r"^\d{4}-\d{2}-\d{2}$",
    "scheduled_time": r"^([01]\d|2[0-3]):[0-5]\d$",
    "flight_no": r"^[A-Z0-9]{2}\s?\d{1,4}[A-Z]?$",
    "direction": r"^(DEPARTURES|ARRIVALS)$",
    "airport": r"^[A-Z]{3}$",
}


def row_hash(df: pd.DataFrame, cols) -> pd.Series:
    """Băm SHA-256 (16 ký tự hex đầu) của các cột khóa đã chuẩn hóa.
    Hai bản ghi có cùng giá trị khóa -> cùng mã băm -> được coi là trùng lặp."""
    joined = df[list(cols)].astype(str).apply(lambda s: s.str.strip().str.upper()).agg("|".join, axis=1)
    return joined.map(lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()[:16])


def format_report(df: pd.DataFrame) -> pd.DataFrame:
    """Tỷ lệ thiếu và sai định dạng theo từng cột của dữ liệu THÔ (trước khi làm sạch)."""
    rows = []
    n = len(df)
    for c in FIELDS:
        if c not in df.columns:
            continue
        s = df[c].astype(str).str.strip()
        empty = s.isin(["", "nan", "None", "NaN"])
        row = {"Cột": c, "Thiếu": int(empty.sum()), "Thiếu (%)": round(empty.mean() * 100, 2)}
        if c in FORMAT_RULES:
            bad = ~empty & ~s.str.upper().str.match(FORMAT_RULES[c])
            row["Sai định dạng"] = int(bad.sum())
            row["Sai định dạng (%)"] = round(bad.sum() / n * 100, 2) if n else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def integration_report(leg: pd.DataFrame) -> pd.DataFrame:
    """Kiểm tra tính nhất quán sau tích hợp chuyến bay + thời tiết + tải sân bay."""
    n = len(leg)
    checks = [
        ("Ghép được sân bay đầu kia (leg_matched)", leg["leg_matched"].mean() * 100),
        ("Chuyến thấy ở cả bảng đi và bảng đến", (leg["n_sides_seen"] > 1).mean() * 100),
        ("Sự kiện có >= 2 nguồn ghi nhận", (leg["source_recorded_count"] > 1).mean() * 100),
        ("Ghép được thời tiết (precipitation có giá trị)", leg["precipitation"].notna().mean() * 100),
        ("Ghép được tải sân bay (airport_load)", leg["airport_load"].notna().mean() * 100),
        ("origin != destination", (leg["origin"] != leg["destination"]).mean() * 100),
        ("leg_key duy nhất", 100.0 if leg["leg_key"].is_unique else 0.0),
    ]
    return pd.DataFrame(
        [{"Kiểm tra": k, "Đạt (%)": round(float(v), 2) if pd.notna(v) else np.nan, "Số dòng": n} for k, v in checks]
    )


def prepare_dataset(prefer_side="DEPARTURES", min_age_hours=0, unit="leg"):
    """Integrate and deduplicate snapshots from all sources into canonical flight legs."""
    rebuild_raw_csv()
    std = load_standard_records()
    frames = [std] if not std.empty else []

    if not frames:
        print("Chưa có dữ liệu trong data/debug_data hay data/raw!")
        return None

    df = pd.concat(frames, ignore_index=True)
    stats = {"snapshots": len(df), "from_debug_json": len(df)}
    stats["format_report"] = format_report(df)

    for c in ["airport", "direction", "flight_date", "flight_no", "scheduled_time", "airline_code"]:
        df[c] = df[c].astype(str).str.strip().str.upper()
    df["status"] = df["status"].astype(str).str.strip()
    df["raw_hash"] = row_hash(df, FIELDS)
    stats["exact_duplicates"] = int(df["raw_hash"].duplicated().sum())
    df = df.drop_duplicates("raw_hash", keep="last").copy()
    stats["after_exact_dedup"] = len(df)

    # Completed flight without actual_time -> fill from estimated_time
    done = df["status"].str.contains(r"arrive|depart|land|hacan|catcan", case=False, regex=True, na=False)
    empty_act = df["actual_time"].astype(str).str.strip().isin(["", "nan", "None"])
    has_est = ~df["estimated_time"].astype(str).str.strip().isin(["", "nan", "None"])
    df.loc[done & empty_act & has_est, "actual_time"] = df.loc[done & empty_act & has_est, "estimated_time"]

    df["flight_no"] = normalize_flight_no(df["flight_no"])
    df["sched_dt"] = sched_datetime(df)
    df["hour_floor"] = df["sched_dt"].dt.floor("h")

    # Airport load is computed across ALL airlines before filtering
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
    df["at_monitored"] = df["airport"].isin(AIRPORT_COORD).astype(int)
    df["event_hash"] = row_hash(df, K1)
    df["leg_hash"] = row_hash(df, ["leg_key"])

    # Step 1: Dedup same side (giữ bản ghi giàu thông tin nhất, mới nhất trong mỗi event_hash)
    df = df.sort_values(K1 + ["has_info", "has_done_act", "scraped_at"])
    per_src = df.drop_duplicates(["event_hash", "source"], keep="last")
    src1 = df.groupby("event_hash")["source"].nunique().rename("source_recorded_count")
    side = df.drop_duplicates("event_hash", keep="last").merge(src1, on="event_hash", how="left")
    stats["after_side_dedup"] = len(side)

    # Step 2: Dedup across departure and arrival sides (1 leg_hash = 1 chuyến)
    side = side.sort_values(["leg_key", "has_info", "has_done_act", "at_monitored", "is_pref_side", "scraped_at"])
    legs_cnt = side.groupby("leg_hash")["direction"].nunique().rename("n_sides_seen")
    if unit == "event":
        leg = side.merge(legs_cnt, on="leg_hash", how="left")
    else:
        leg = side.drop_duplicates("leg_hash", keep="last").merge(legs_cnt, on="leg_hash", how="left")
    stats["unit"] = unit
    stats["after_leg_dedup"] = len(leg)
    stats["legs_seen_both_sides"] = int((leg["n_sides_seen"] > 1).sum())

    unm = side[~side["leg_matched"]]
    stats["unmatched_side"] = len(unm)
    stats["unmapped_top"] = unm["other_airport"].str.strip().replace("", "(trống)").value_counts().head(20)

    leg = add_time_features(leg, load)
    leg = attach_weather(leg)
    return side, leg, stats, per_src


def _load_level(v: pd.DataFrame) -> pd.Series:
    pct = v.groupby("airport")["airport_load"].rank(pct=True)
    return pd.Series(np.select([pct <= 1 / 3, pct <= 2 / 3, pct > 2 / 3], ["low", "mid", "high"], None), index=v.index)


def build_items(v: pd.DataFrame) -> pd.DataFrame:
    """Discretize variables into boolean itemset dataframe for FP-Growth."""
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


def mine_rules(items: pd.DataFrame, target: str, min_sup: float, min_conf: float, min_lift: float) -> pd.DataFrame:
    """Extract non-redundant association rules using FP-Growth."""
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
        if any(
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


def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def source_agreement(per_src: pd.DataFrame, thr: int):
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


def airline_table(valid: pd.DataFrame) -> pd.DataFrame:
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
    miss = valid.replace("", np.nan).isna().mean().mul(100).round(1)
    miss = miss[miss > 0].sort_values(ascending=False)
    print("-" * 62)
    print("TỶ LỆ THIẾU THEO CỘT (bộ dữ liệu cuối)")
    if miss.empty:
        print("Không cột nào thiếu dữ liệu.")
    for c, p in miss.items():
        print(f"  {c:<22}{p:>6.1f}%" + ("   <-- thiếu nhiều, cân nhắc bỏ" if p > MISSING_WARN else ""))


def label_and_report(side, leg, stats, thr, per_src=None):
    side, leg = apply_labels(side, thr), apply_labels(leg, thr)
    valid = leg[leg["status_clean"].isin(LABELS)].copy()

    sv = side[side["status_clean"].isin(LABELS)].assign(is_cx=lambda d: d["status_clean"].eq("cancelled"))
    g = sv.groupby("leg_key")["is_cx"].agg(["nunique", "size"])
    multi = g[g["size"] > 1]
    agree = (multi["nunique"] == 1).mean() * 100 if len(multi) else float("nan")

    print("=" * 62)
    print(f"BÁO CÁO TÍCH HỢP DỮ LIỆU (ngưỡng trễ = {thr} phút)")
    print("=" * 62)
    print(f"Snapshot thô:                               {stats['snapshots']:,}")
    if "exact_duplicates" in stats:
        print(f"Trùng tuyệt đối (cùng mã băm SHA-256):      {stats['exact_duplicates']:,}")
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


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Part 1: Làm sạch & tích hợp dữ liệu chuyến bay")
    ap.add_argument("--threshold", type=int, default=15, help="Ngưỡng trễ (phút)")
    ap.add_argument("--compare", type=int, default=None, help="Ngưỡng thứ 2 để so sánh (vd 30)")
    ap.add_argument("--unit", choices=["leg", "event"], default="leg")
    ap.add_argument("--min-age-hours", type=float, default=0)
    a = ap.parse_args()
    process_and_mine(threshold=a.threshold, compare=a.compare, min_age_hours=a.min_age_hours, unit=a.unit)

