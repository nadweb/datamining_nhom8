# -*- coding: utf-8 -*-
"""Học không giám sát và khai phá mẫu (Part 5B - Thành viên 5).

1. K-Means (k=4): fit trên Train. Tên cụm được SUY RA TỪ DỮ LIỆU (giờ bay trung bình, thời tiết, tải sân bay),
   không gán cứng theo số thứ tự cụm (số thứ tự cụm của K-Means là ngẫu nhiên). Có quét k=2..8 (inertia, silhouette).
   Hồ sơ cụm được đo ở cả Train và Test để kiểm tra độ ổn định.
2. PCA 2D/3D: fit trên Train, kèm các đặc trưng đóng góp nhiều nhất cho PC1, PC2.
3. Isolation Forest: fit trên Train. Lưu ý: tỷ lệ cờ bất thường trên Train (3%) do tham số contamination quy định,
   không phải phát hiện; thông tin có giá trị là các chuyến bị gắn cờ khác bình thường ở điểm nào.
4. Luật kết hợp FP-Growth: lấy top luật theo lift từ data/mined_association_rules.csv.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

from config import BASE_DIR, FILE_RULES
from modeling.encoding import (
    FILE_TEST, FILE_TRAIN, NUMERIC_CONTINUOUS_COLS, PASSTHROUGH_BINARY_COLS, PROCESSED_DIR, FlightDataPreprocessor,
)
from modeling.train import feature_columns

MODELS_DIR = BASE_DIR / "models"
FILE_KMEANS = MODELS_DIR / "kmeans_model.joblib"
FILE_PCA = MODELS_DIR / "pca_model.joblib"
FILE_ISOLATION_FOREST = MODELS_DIR / "isolation_forest.joblib"
FILE_CLUSTER_PROFILES = MODELS_DIR / "cluster_profiles.json"
FILE_UNSUPERVISED_METRICS = MODELS_DIR / "unsupervised_metrics.json"

RANDOM_STATE = 42
# Giữ tên này để mã cũ import không vỡ; nội dung được điền lúc chạy từ dữ liệu (không còn gán cứng)
CLUSTER_NAMES: Dict[int, str] = {}

TIME_OF_DAY = [(6, "Night/Early morning", "Đêm - sáng sớm"), (12, "Morning", "Buổi sáng"),
               (18, "Afternoon", "Buổi chiều"), (24, "Evening", "Buổi tối")]


# ======================================================================= dữ liệu
def load_train_test_matrices() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, test_df = pd.read_csv(FILE_TRAIN), pd.read_csv(FILE_TEST)
    cols = feature_columns(train_df)
    return train_df[cols], test_df[cols], train_df, test_df


def load_full_feature_matrix() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Giữ lại cho mã cũ: train + test gộp."""
    X_tr, X_te, tr, te = load_train_test_matrices()
    return pd.concat([X_tr, X_te], ignore_index=True), pd.concat([tr, te], ignore_index=True)


def _try_scaler():
    try:
        return FlightDataPreprocessor.load(PROCESSED_DIR).scaler
    except Exception as e:                                  # chưa có artifact của Thành viên 4
        print(f"Lưu ý: không nạp được scaler của TV4 ({type(e).__name__}); tên cụm sẽ không có thông tin giờ/thời tiết.")
        return None


# ======================================================================= K-Means
def sweep_k(X: pd.DataFrame, k_range=range(2, 9), sample: int = 3000) -> List[dict]:
    """Elbow (inertia) và silhouette cho nhiều giá trị k; silhouette tính trên mẫu cố định hạt giống."""
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X), min(sample, len(X)), replace=False)
    out = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X)
        out.append({"k": k, "inertia": round(float(km.inertia_), 1),
                    "silhouette": round(float(silhouette_score(X.iloc[idx], km.labels_[idx])), 4)})
    return out


def _circular_hour(sin_raw: float, cos_raw: float) -> float:
    return float((np.arctan2(sin_raw, cos_raw) % (2 * np.pi)) / (2 * np.pi) * 24)


def _time_label(hour: float) -> Tuple[str, str]:
    return next((en, vi) for edge, en, vi in TIME_OF_DAY if hour < edge)


def _weather_label(precip: float, gust: float, vis_km: float, storm_share: float) -> Tuple[str, str]:
    if precip >= 1.0 or storm_share >= 0.3:
        return "Rain/Storm", "Mưa lớn/dông"
    if gust >= 22 or vis_km < 10:
        return "Windy/Reduced visibility", "Gió mạnh/tầm nhìn giảm"
    return "Clear weather", "Thời tiết tốt"


def _dominant(X: pd.DataFrame, mask: np.ndarray, prefix: str) -> str:
    # Chỉ xét cột One-Hot: tiền tố "airport_" cũng trùng với cột số airport_load
    skip = set(NUMERIC_CONTINUOUS_COLS) | set(PASSTHROUGH_BINARY_COLS)
    cols = [c for c in X.columns if c.startswith(prefix) and c not in skip]
    return cols[int(np.argmax(X.loc[mask, cols].mean().values))][len(prefix):] if cols and mask.any() else ""


def train_kmeans_clustering(X_train, train_df, n_clusters: int = 4, X_test=None, test_df=None,
                            details: Optional[dict] = None) -> Tuple[KMeans, dict]:
    """K-Means fit trên Train; hồ sơ cụm tính trên Train+Test, kèm tỷ lệ trễ riêng ở Train và Test."""
    print("\n" + "=" * 65)
    print(f"      K-MEANS (k={n_clusters}) - FIT TRÊN TRAIN")
    print("=" * 65)
    sweep = sweep_k(X_train)
    best_k = max(sweep, key=lambda r: r["silhouette"])["k"]
    print("  k  | inertia    | silhouette")
    for r in sweep:
        print(f"  {r['k']:<2} | {r['inertia']:>10.1f} | {r['silhouette']:.4f}{'  <- tốt nhất' if r['k'] == best_k else ''}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10).fit(X_train)
    sil = next((r["silhouette"] for r in sweep if r["k"] == n_clusters), None)
    print(f"\n  k đang dùng = {n_clusters}: silhouette {sil} | inertia {kmeans.inertia_:.1f}  (k tốt nhất theo silhouette: {best_k})")

    X_all = pd.concat([X_train, X_test], ignore_index=True) if X_test is not None else X_train.reset_index(drop=True)
    df_all = pd.concat([train_df, test_df], ignore_index=True) if test_df is not None else train_df.reset_index(drop=True)
    labels = kmeans.predict(X_all)
    is_train = np.arange(len(X_all)) < len(X_train)

    scaler = _try_scaler()
    raw = None
    if scaler is not None and all(c in X_all.columns for c in NUMERIC_CONTINUOUS_COLS):
        raw = pd.DataFrame(X_all[NUMERIC_CONTINUOUS_COLS].values * scaler.scale_ + scaler.mean_, columns=NUMERIC_CONTINUOUS_COLS)

    profiles = {}
    for cid in range(n_clusters):
        m = labels == cid
        sub = df_all[m]
        delayed, canc = (sub["target"] == 1), (sub["target"] == 2)
        p = {
            "cluster_id": cid, "flight_count": int(m.sum()), "percentage": round(float(m.mean() * 100), 1),
            "delay_rate_pct": round(float(delayed.mean() * 100), 1),
            "cancellation_rate_pct": round(float(canc.mean() * 100), 2),
            "avg_delay_minutes": round(float(sub["delay_min"].dropna().mean()), 1) if sub["delay_min"].notna().any() else 0.0,
            "train_delay_rate_pct": round(float((df_all.loc[m & is_train, "target"] == 1).mean() * 100), 1) if (m & is_train).any() else None,
            "test_delay_rate_pct": round(float((df_all.loc[m & ~is_train, "target"] == 1).mean() * 100), 1) if (m & ~is_train).any() else None,
            "dominant_airline": _dominant(X_all, m, "airline_code_"),
            "dominant_airport": _dominant(X_all, m, "airport_"),
            "domestic_pct": round(float(X_all.loc[m, "route_type_domestic"].mean() * 100), 1) if "route_type_domestic" in X_all else None,
        }
        p["risk_tier"] = "HIGH" if p["delay_rate_pct"] > 30 else ("MEDIUM" if p["delay_rate_pct"] > 20 else "LOW")
        if raw is not None:
            r = raw[m]
            hour = _circular_hour(r["hour_sin"].mean(), r["hour_cos"].mean())
            storm = float(X_all.loc[m, "is_storm"].mean()) if "is_storm" in X_all else 0.0
            tod_en, tod_vi = _time_label(hour)
            wx_en, wx_vi = _weather_label(r["precipitation"].mean(), r["wind_gusts_10m"].mean(),
                                          r["visibility"].mean() / 1000, storm)
            p.update({"mean_hour": round(hour, 1), "avg_airport_load": round(float(r["airport_load"].mean()), 1),
                      "avg_precipitation_mm": round(float(r["precipitation"].mean()), 2),
                      "avg_wind_gust_kmh": round(float(r["wind_gusts_10m"].mean()), 1),
                      "avg_visibility_km": round(float(r["visibility"].mean() / 1000), 1),
                      "storm_share_pct": round(storm * 100, 1)})
            p["name"] = f"{tod_en}, {wx_en} ({tod_vi}, {wx_vi})"
        else:
            p["name"] = f"Cluster {cid}"
        profiles[cid] = p

    # Tên phải khác nhau: nếu trùng thì thêm sân bay chiếm ưu thế, rồi số cụm
    names = [p["name"] for p in profiles.values()]
    for cid, p in profiles.items():
        if names.count(p["name"]) > 1 and p["dominant_airport"]:
            p["name"] += f" - {p['dominant_airport']}"
    names = [p["name"] for p in profiles.values()]
    for cid, p in profiles.items():
        if names.count(p["name"]) > 1:
            p["name"] += f" #{cid}"

    CLUSTER_NAMES.clear()
    CLUSTER_NAMES.update({cid: p["name"] for cid, p in profiles.items()})
    for cid, p in profiles.items():
        print(f"\n* Cụm {cid}: {p['name']}")
        print(f"    - Quy mô {p['flight_count']:,} chuyến ({p['percentage']}%) | trễ {p['delay_rate_pct']}% "
              f"(Train {p['train_delay_rate_pct']}% / Test {p['test_delay_rate_pct']}%) | hủy {p['cancellation_rate_pct']}% | rủi ro {p['risk_tier']}")
    if details is not None:
        details["kmeans"] = {"silhouette": sil, "inertia": round(float(kmeans.inertia_), 1), "k_selection": sweep,
                             "best_k_by_silhouette": best_k}
        details["_labels"] = labels
    return kmeans, {str(k): v for k, v in profiles.items()}


# ======================================================================= PCA
def train_pca_reduction(X: pd.DataFrame, n_components: int = 3) -> Tuple[PCA, dict]:
    print("\n" + "=" * 65)
    print(f"      PCA (N={n_components}) - FIT TRÊN TRAIN")
    print("=" * 65)
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE).fit(X)
    ratios = [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_]
    print(f"  Phương sai giải thích: {ratios} | tổng {round(sum(ratios), 2)}%")
    loadings = {}
    for i in range(min(2, n_components)):
        order = np.argsort(-np.abs(pca.components_[i]))[:5]
        loadings[f"PC{i + 1}"] = [{"feature": X.columns[j], "loading": round(float(pca.components_[i][j]), 3)} for j in order]
    return pca, {
        "n_components": n_components, "explained_variance_ratios": ratios,
        "total_explained_variance_pct": round(sum(ratios), 2),
        "explained_variance_ratio_2d": ratios[:2], "cumulative_variance_2d": round(sum(ratios[:2]) / 100, 4),
        "top_loadings": loadings,
    }


# ======================================================================= Isolation Forest
def train_isolation_forest(X_train, X_test=None, df_all=None, cluster_labels=None) -> Tuple[IsolationForest, dict]:
    print("\n" + "=" * 65)
    print("      ISOLATION FOREST - FIT TRÊN TRAIN")
    print("=" * 65)
    iso = IsolationForest(n_estimators=100, contamination=0.03, random_state=RANDOM_STATE, n_jobs=-1).fit(X_train)
    X_all = pd.concat([X_train, X_test], ignore_index=True) if X_test is not None else X_train.reset_index(drop=True)
    flag = iso.predict(X_all) == -1
    n_tr = len(X_train)
    count, pct = int(flag.sum()), round(float(flag.mean() * 100), 2)
    metrics = {
        "anomaly_count": count, "anomaly_percentage": pct, "anomaly_pct": pct, "contamination": 0.03,
        "train_flagged_pct": round(float(flag[:n_tr].mean() * 100), 2),
        "test_flagged_pct": round(float(flag[n_tr:].mean() * 100), 2) if X_test is not None else None,
        "note": "Tỷ lệ ~3% trên Train do tham số contamination quy định. Tỷ lệ trên Test mới là thông tin: nếu cao hơn nhiều, "
                "dữ liệu mới khác dữ liệu huấn luyện.",
    }
    print(f"  Gắn cờ {count:,} chuyến ({pct}%) | Train {metrics['train_flagged_pct']}% | Test {metrics['test_flagged_pct']}%")

    if df_all is not None and flag.any():
        d = df_all.reset_index(drop=True)
        metrics["delay_rate_flagged_pct"] = round(float((d.loc[flag, "target"] == 1).mean() * 100), 1)
        metrics["delay_rate_normal_pct"] = round(float((d.loc[~flag, "target"] == 1).mean() * 100), 1)
        metrics["cancel_rate_flagged_pct"] = round(float((d.loc[flag, "target"] == 2).mean() * 100), 2)
        metrics["cancel_rate_normal_pct"] = round(float((d.loc[~flag, "target"] == 2).mean() * 100), 2)
        cols = [c for c in NUMERIC_CONTINUOUS_COLS + PASSTHROUGH_BINARY_COLS if c in X_all.columns]
        diff = (X_all.loc[flag, cols].mean() - X_all.loc[~flag, cols].mean()) / X_all[cols].std(ddof=0).replace(0, np.nan)
        top = diff.abs().sort_values(ascending=False).head(8).index
        metrics["top_distinguishing_features"] = [{"feature": c, "std_diff": round(float(diff[c]), 2)} for c in top]
        if cluster_labels is not None:
            metrics["flagged_by_cluster"] = {str(k): int(v) for k, v in
                                             pd.Series(np.asarray(cluster_labels)[flag]).value_counts().sort_index().items()}
        print(f"  Tỷ lệ trễ: chuyến bị gắn cờ {metrics['delay_rate_flagged_pct']}% so với bình thường {metrics['delay_rate_normal_pct']}%")
        print("  Đặc trưng khác biệt nhất: " + ", ".join(f"{r['feature']} ({r['std_diff']:+.2f}σ)" for r in metrics["top_distinguishing_features"][:5]))
    return iso, metrics


# ======================================================================= luật kết hợp
def load_association_rules(top_n: int = 15) -> List[dict]:
    """Top luật theo LIFT giảm dần (không phải 15 dòng đầu của tệp)."""
    if not FILE_RULES.exists():
        return []
    try:
        df = pd.read_csv(FILE_RULES)
        lift_col = next((c for c in df.columns if "lift" in c.lower()), None)
        if lift_col:
            df = df.sort_values(lift_col, ascending=False)
        return json.loads(df.head(top_n).to_json(orient="records", force_ascii=False))
    except Exception as e:
        print(f"Lưu ý: không đọc được luật kết hợp: {e}")
        return []


# ======================================================================= điều phối
def run_unsupervised_pipeline():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, train_df, test_df = load_train_test_matrices()
    details: dict = {}

    kmeans, profiles = train_kmeans_clustering(X_train, train_df, 4, X_test=X_test, test_df=test_df, details=details)
    pca, pca_metrics = train_pca_reduction(X_train, 3)
    df_all = pd.concat([train_df, test_df], ignore_index=True)
    iso, iso_metrics = train_isolation_forest(X_train, X_test, df_all, details.pop("_labels", None))
    top_rules = load_association_rules()

    joblib.dump(kmeans, FILE_KMEANS)
    joblib.dump(pca, FILE_PCA)
    joblib.dump(iso, FILE_ISOLATION_FOREST)
    FILE_CLUSTER_PROFILES.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "kmeans": {"n_clusters": 4, "profiles": profiles, **details["kmeans"]},
        "pca": pca_metrics, "isolation_forest": iso_metrics,
        "top_rules_count": len(top_rules), "association_rules": top_rules,
        "environment": {"sklearn": sklearn.__version__, "random_state": RANDOM_STATE,
                        "fit_on": "Train", "n_train": int(len(X_train)), "n_test": int(len(X_test))},
    }
    FILE_UNSUPERVISED_METRICS.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 65)
    for label, path in [("K-Means", FILE_KMEANS), ("PCA", FILE_PCA), ("Isolation Forest", FILE_ISOLATION_FOREST),
                        ("Hồ sơ cụm", FILE_CLUSTER_PROFILES), ("Chỉ số", FILE_UNSUPERVISED_METRICS)]:
        print(f"Đã lưu {label:<17}: {path}")
    print("=" * 65)
    return summary


if __name__ == "__main__":
    run_unsupervised_pipeline()
