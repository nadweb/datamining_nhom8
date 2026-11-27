# -*- coding: utf-8 -*-
"""[STEP 4] Unsupervised K-Means Operational Regime Discovery & Stability Analysis."""
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from flight_delay.common.constants import ENCODERS_DIR, EXPERIMENTS_CFG, RANDOM_STATE, ROOT_DIR

CLUSTER_FEATURES = [
    "hour_sin",
    "hour_cos",
    "airport_load",
    "precipitation",
    "wind_gusts_10m",
    "visibility",
    "is_peak_hour",
    "is_weekend_flag",
    "is_rain",
    "is_high_wind",
]


def _circular_hour(sin_vals: pd.Series, cos_vals: pd.Series) -> float:
    ang = np.arctan2(sin_vals.mean(), cos_vals.mean())
    if ang < 0:
        ang += 2 * np.pi
    return round(float(ang * 24.0 / (2 * np.pi)), 1)


def _name_cluster(avg_rain: float, avg_wind: float, avg_load: float, avg_hr: float, delay_rate: float) -> tuple[str, str, str]:
    if avg_rain >= 0.4 or avg_wind >= 28.0:
        return (
            "Bất lợi Thời tiết (Mưa & Gió mạnh)",
            "Tập trung các chuyến gặp mưa lớn hoặc gió giật mạnh tại sân bay.",
            "Cộng thêm đệm quay đầu >= 25 phút và cảnh báo sớm cho hành khách.",
        )
    if avg_load >= 26.0 and (6 <= avg_hr <= 10 or 16 <= avg_hr <= 21):
        return (
            "Giờ Cao điểm Đông tải Trục chính",
            "Mật độ khai thác rất cao trong khung giờ vàng tại SGN/HAN.",
            "Ưu tiên slot cất cánh và đội phục vụ mặt đất nhanh (quick-turnaround).",
        )
    if delay_rate >= 35.0:
        return (
            "Áp lực Dây chuyền & Khai thác Chiều tối",
            "Tích lũy trễ dây chuyền cuối ngày kết hợp tải khai thác trung-cao.",
            "Giám sát tàu bay quay đầu (rotational tail tracking) từ 14h trở đi.",
        )
    return (
        "Điều kiện Thuận lợi / Tải thấp",
        "Thời tiết quang đãng, mật độ sân bay nhẹ — độ đúng giờ cao nhất.",
        "Duy trì lịch khai thác tiêu chuẩn; điểm chuẩn ổn định của mạng bay.",
    )


def run_kmeans_clustering(gold_df: pd.DataFrame) -> dict:
    """Run K sweep (Elbow/Silhouette/Davies-Bouldin/Calinski-Harabasz), Seed-ARI, and fit final K-Means."""
    cfg = EXPERIMENTS_CFG["unsupervised"]["kmeans"]
    k_sel = int(cfg["k_selected"])
    k_sweep = [int(k) for k in cfg["k_sweep"]]
    seeds = [int(s) for s in cfg["stability_seeds"]]
    sample_size = int(cfg["sample_size"])

    avail_feats = [c for c in CLUSTER_FEATURES if c in gold_df.columns]
    X_raw = gold_df[avail_feats].fillna(0.0).copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    rng = np.random.default_rng(RANDOM_STATE)
    n_sub = min(sample_size, len(X_scaled))
    sub_idx = rng.choice(len(X_scaled), size=n_sub, replace=False)
    X_sub = X_scaled[sub_idx]

    sweep_rows = []
    for k in k_sweep:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels_sub = km.fit_predict(X_sub)
        sweep_rows.append({
            "k": k,
            "inertia": round(float(km.inertia_), 1),
            "silhouette": round(float(silhouette_score(X_sub, labels_sub)), 4),
            "davies_bouldin": round(float(davies_bouldin_score(X_sub, labels_sub)), 4),
            "calinski_harabasz": round(float(calinski_harabasz_score(X_sub, labels_sub)), 1),
        })

    # Seed stability across random initializations
    seed_labels = []
    for s in seeds:
        km_s = KMeans(n_clusters=k_sel, n_init=10, random_state=s)
        seed_labels.append(km_s.fit_predict(X_sub))
    aris = []
    for i in range(len(seed_labels)):
        for j in range(i + 1, len(seed_labels)):
            aris.append(float(adjusted_rand_score(seed_labels[i], seed_labels[j])))

    final_km = KMeans(n_clusters=k_sel, n_init=15, random_state=RANDOM_STATE)
    full_labels = final_km.fit_predict(X_scaled)

    ENCODERS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, ENCODERS_DIR / "cluster_scaler.joblib")
    compat_models = ROOT_DIR / "models"
    compat_models.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_km, compat_models / "kmeans_model.joblib")

    work = gold_df.copy().reset_index(drop=True)
    work["_cluster"] = full_labels
    profiles = []
    for cid, grp in work.groupby("_cluster"):
        n = len(grp)
        avg_hr = _circular_hour(grp["hour_sin"], grp["hour_cos"]) if {"hour_sin", "hour_cos"}.issubset(grp) else 12.0
        avg_load = float(grp["airport_load"].mean())
        avg_rain = float(grp["precipitation"].mean())
        avg_wind = float(grp["wind_gusts_10m"].mean())
        avg_vis = float(grp["visibility"].mean())
        delay_rate = float((grp["status_clean"] == "delayed").mean() * 100.0)
        cancel_rate = float((grp["status_clean"] == "cancelled").mean() * 100.0)
        disr_rate = float(grp["status_clean"].isin(["delayed", "cancelled"]).mean() * 100.0)
        title, desc, action = _name_cluster(avg_rain, avg_wind, avg_load, avg_hr, delay_rate)
        profiles.append({
            "cluster_id": int(cid),
            "name": title,
            "description": desc,
            "recommended_action": action,
            "count": int(n),
            "share_pct": round(float(n / len(work) * 100.0), 1),
            "avg_hour": avg_hr,
            "avg_airport_load": round(avg_load, 1),
            "avg_precipitation_mm": round(avg_rain, 2),
            "avg_wind_kmh": round(avg_wind, 1),
            "avg_visibility_m": round(avg_vis, 0),
            "delay_rate_pct": round(delay_rate, 1),
            "cancel_rate_pct": round(cancel_rate, 2),
            "disruption_rate_pct": round(disr_rate, 1),
            "dominant_airport": str(grp["airport"].mode().iloc[0]) if not grp["airport"].mode().empty else "",
            "dominant_airline": str(grp["airline_code"].mode().iloc[0]) if not grp["airline_code"].mode().empty else "",
            "dominant_time_slot": str(grp["time_slot"].mode().iloc[0]) if "time_slot" in grp and not grp["time_slot"].mode().empty else "",
            "dominant_day_type": str(grp["day_type"].mode().iloc[0]) if "day_type" in grp and not grp["day_type"].mode().empty else "",
        })

    profiles.sort(key=lambda r: r["disruption_rate_pct"], reverse=True)
    return {
        "k_selected": k_sel,
        "features_used": avail_feats,
        "k_sweep": sweep_rows,
        "seed_stability": {
            "seeds": seeds,
            "ari_mean": round(float(np.mean(aris)), 4) if aris else 1.0,
            "ari_min": round(float(np.min(aris)), 4) if aris else 1.0,
        },
        "full_silhouette": round(float(silhouette_score(X_sub, final_km.predict(X_sub))), 4),
        "cluster_profiles": profiles,
        "_labels": full_labels,
        "_X_scaled": X_scaled,
    }
