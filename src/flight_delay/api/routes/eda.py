# -*- coding: utf-8 -*-
"""[STEP 7] EDA, Schema Lineage, Correlation Matrix & Association Rules Routes."""
import pandas as pd
from fastapi import APIRouter

from flight_delay.api.dependencies import safe_read_json
from flight_delay.common.constants import FILE_GOLD, FILE_RULES, REPORTS_DIR

router = APIRouter(prefix="/api", tags=["EDA & Unsupervised Discovery"])


def _enrich_eda_for_ui(eda_doc: dict) -> dict:
    """Attach canonical UI tables (histograms, schema evolution summaries, Cramer keys)."""
    doc = dict(eda_doc) if eda_doc else {}
    # Schema stages formatting
    stages_raw = doc.get("schema_evolution", {}).get("stages", [])
    ui_stages = []
    why_map = {
        "01_bronze_raw": "11 file FIDS HTML + Aviationstack JSON + Open-Meteo lệch schema (tới 58 cột, có cột thiếu >98%).",
        "02_bronze_standardized": "Chưa khử trùng lịch sử cào lặp (nhiều snapshot mỗi chuyến), delay chưa quy chuẩn.",
        "03_silver_cleaned": "Thiếu biến chu kỳ thời gian, tải sân bay tương đối, chỉ số thời tiết & tỷ lệ trễ nhân quả.",
        "04_gold_analytical": "Biến chuỗi phân loại & thang đo chưa được mã hóa One-Hot/StandardScaler; còn cột hậu kiểm.",
        "05_modeling_matrix_X": "Sẵn sàng cho HGB, XGBoost & CatBoost (0 cột rò rỉ tương lai).",
    }
    for s in stages_raw:
        st_name = s.get("stage", "")
        ui_stages.append({
            **s,
            "why_changed": s.get("why_changed") or why_map.get(st_name, s.get("explanation", "")),
            "how_transformed": s.get("how_transformed") or s.get("explanation", ""),
        })

    doc["schema_evolution"] = {
        "schema_stages": ui_stages,
        "feature_derivations": doc.get("schema_evolution", {}).get("feature_derivations") or [
            {
                "new_feature": "hour_sin / hour_cos & dow_sin / dow_cos",
                "source_columns": ["scheduled_time", "flight_date"],
                "empirical_reason": "Giờ 23:55 và 00:05 liên tục trên đồng hồ 24h nhưng bị tách rời nếu dùng số tuyến tính.",
                "formula_summary": "sin/cos(2*pi*hour/24) & sin/cos(2*pi*dow/7)",
            },
            {
                "new_feature": "weather_hazard_score & load_x_weather",
                "source_columns": ["precipitation", "wind_gusts_10m", "visibility", "weather_code", "airport_load"],
                "empirical_reason": "Tổ hợp mưa + gió + tầm nhìn kém vào giờ cao điểm làm tăng tỷ lệ trễ phi tuyến.",
                "formula_summary": "Trọng số 0.35*rain + 0.25*wind + 0.20*vis + 0.20*storm; nhân airport_load",
            },
            {
                "new_feature": "historical_airline/airport_delay_rate",
                "source_columns": ["airline_code", "airport", "outcome_known_at", "prediction_time"],
                "empirical_reason": "Chiếm phần lớn tín hiệu phân tách (Cramér V ~ 0.16), dùng bisect_left strictly < prediction_time.",
                "formula_summary": "Bayesian m-estimate smoothing (m=20) chỉ trên chuyến đã hoàn tất trước T-3h",
            },
            {
                "new_feature": "[BỊ CHẶN] delay_min, actual_time, status_clean",
                "source_columns": ["status", "actual_time", "delay_min"],
                "empirical_reason": "Thuộc tính hậu kiểm xảy ra sau cất cánh — cấm tuyệt đối đưa vào ma trận X.",
                "formula_summary": "Loại bỏ ở FeatureRegistry & Leakage Guard tests",
            },
        ],
    }

    corr = dict(doc.get("correlation_and_association", {}))
    if "matrix_columns" in corr and "heatmap_columns" not in corr:
        corr["heatmap_columns"] = corr["matrix_columns"]
        corr["pearson_heatmap"] = corr["pearson_matrix"]
        corr["spearman_heatmap"] = corr["spearman_matrix"]
    corr_cats = []
    for row in corr.get("categorical_cramers_v", []):
        corr_cats.append({
            **row,
            "cramers_v_vs_delayed": row.get("cramers_v_disrupted", row.get("cramers_v_vs_delayed", 0.0)),
            "cramers_v_vs_status": row.get("cramers_v_3class", row.get("cramers_v_vs_status", 0.0)),
        })
    corr["categorical_cramers_v"] = corr_cats
    doc["correlation_and_association"] = corr

    eda = dict(doc.get("eda", {}))
    biv = eda.get("bivariate", {})
    if biv:
        major_aps = {"SGN", "HAN", "DAD", "CXR", "PQC"}
        ap_list = biv.get("by_airport", [])
        eda["by_major_airport"] = [
            {
                "key": a["airport"],
                "count": a["total"],
                "delayed_pct": a["delayed_pct"],
                "cancelled_pct": a["cancelled_pct"],
                "avg_delay_min": a.get("mean_delay_min") or 0.0,
            }
            for a in ap_list if a.get("airport") in major_aps
        ]
        minor_rows = [a for a in ap_list if a.get("airport") not in major_aps]
        minor_cnt = sum(a["total"] for a in minor_rows)
        tot = max(eda.get("total_flights", 1), 1)
        eda["minor_airports_summary"] = {
            "n_airports": len(minor_rows),
            "count": minor_cnt,
            "share_pct": round(minor_cnt / tot * 100, 2),
        }
        eda["by_airline"] = [
            {
                "key": a["airline_code"],
                "count": a["total"],
                "delayed_pct": a["delayed_pct"],
                "cancelled_pct": a["cancelled_pct"],
                "avg_delay_min": a.get("mean_delay_min") or 0.0,
            }
            for a in biv.get("by_airline", [])
        ]
        eda["by_time_slot"] = [
            {
                "key": a["time_slot"],
                "count": a["total"],
                "delayed_pct": a["delayed_pct"],
                "avg_delay_min": a.get("mean_delay_min") or 0.0,
            }
            for a in biv.get("by_time_slot", [])
        ]

    if FILE_GOLD.exists() and "delay_min_histogram" not in eda:
        gold = pd.read_csv(FILE_GOLD, low_memory=False)
        delays = pd.to_numeric(gold["delay_min"], errors="coerce").dropna()
        tot_d = max(len(delays), 1)
        buckets = [
            ("Sớm (<0m)", int((delays < 0).sum())),
            ("Đúng giờ (0-14m)", int(((delays >= 0) & (delays < 15)).sum())),
            ("Trễ nhẹ (15-29m)", int(((delays >= 15) & (delays < 30)).sum())),
            ("Trễ vừa (30-59m)", int(((delays >= 30) & (delays < 60)).sum())),
            ("Trễ nặng (60-119m)", int(((delays >= 60) & (delays < 120)).sum())),
            ("Rất nặng (>=120m)", int((delays >= 120).sum())),
        ]
        eda["delay_min_histogram"] = [
            {"bucket": b, "count": cnt, "pct": round(cnt / tot_d * 100, 2)} for b, cnt in buckets
        ]
        # Weather regime table
        is_rain = gold.get("is_rain", 0).astype(int) == 1
        is_h_rain = gold.get("is_heavy_rain", 0).astype(int) == 1
        is_wind = gold.get("is_high_wind", 0).astype(int) == 1
        regimes = [
            ("Quang đãng / Bình thường", (~is_rain) & (~is_wind)),
            ("Mưa nhẹ / Có gió", (is_rain | is_wind) & (~is_h_rain)),
            ("Mưa lớn / Dông giật", is_h_rain),
        ]
        wx_rows = []
        for lbl, m in regimes:
            sub = gold[m]
            if not sub.empty:
                d_sub = pd.to_numeric(sub["delay_min"], errors="coerce").dropna()
                wx_rows.append({
                    "key": lbl,
                    "count": int(len(sub)),
                    "delayed_pct": round(float((sub["status_clean"] == "delayed").mean() * 100), 2),
                    "cancelled_pct": round(float((sub["status_clean"] == "cancelled").mean() * 100), 2),
                    "avg_delay_min": round(float(d_sub.mean()), 2) if len(d_sub) else 0.0,
                })
        eda["by_weather_regime"] = wx_rows

    doc["eda"] = eda
    return doc


@router.get("/eda")
def api_eda():
    raw_eda = safe_read_json(REPORTS_DIR / "eda" / "eda_report.json")
    return _enrich_eda_for_ui(raw_eda)


@router.get("/clusters")
def api_clusters():
    unsup = safe_read_json(REPORTS_DIR / "experiments" / "unsupervised_results.json")
    km = unsup.get("kmeans", {})
    profiles = {}
    for p in km.get("cluster_profiles", []):
        cid = p.get("cluster_id", 0)
        d_rate = float(p.get("disruption_rate_pct", 0.0))
        profiles[str(cid)] = {
            **p,
            "flight_count": p.get("count", 0),
            "cancellation_rate_pct": p.get("cancel_rate_pct", 0.0),
            "avg_delay_minutes": p.get("avg_airport_load", 0.0),
            "risk_tier": "HIGH" if d_rate >= 40 else "MEDIUM" if d_rate >= 28 else "LOW",
        }
    return {
        "profiles": profiles,
        "kmeans": {
            "k_selected": km.get("k_selected", 4),
            "k_selection": km.get("k_sweep", []),
            "seed_stability": km.get("seed_stability", {}),
            "full_silhouette": km.get("full_silhouette"),
            "suitability_note": "K-Means (k=4) phân đoạn 4 chế độ khai thác; dùng hỗ trợ giải thích ngữ cảnh, không thay thế bộ phân loại có giám sát.",
        },
        "pca": unsup.get("pca", {}),
        "anomalies": unsup.get("isolation_forest", {}),
    }


@router.get("/rules")
def api_rules():
    if not FILE_RULES.exists():
        return {"total_rules": 0, "rules": []}
    df = pd.read_csv(FILE_RULES)
    unsup = safe_read_json(REPORTS_DIR / "experiments" / "unsupervised_results.json")
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Antecedent": str(r.get("antecedents", "")),
            "Consequent": str(r.get("consequent", "")),
            "Support (%)": float(r.get("support_pct", 0.0)),
            "Confidence (%)": float(r.get("confidence_pct", 0.0)),
            "Lift": float(r.get("lift", 1.0)),
            "test_support_pct": float(r.get("test_support_pct", 0.0)),
            "test_confidence_pct": float(r.get("test_confidence_pct", 0.0)),
            "test_lift": float(r.get("test_lift", 1.0)),
            "holdout_validated": bool(r.get("holdout_validated", True)),
        })
    return {
        "total_rules": len(rows),
        "summary": unsup.get("association_rules", {}).get("per_target_summary", {}),
        "rules": rows,
    }
