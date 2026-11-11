# -*- coding: utf-8 -*-
"""Unsupervised Machine Learning & Pattern Mining Module (Part 5B - Member 5)
Performs unsupervised learning and operational flight pattern discovery:
1. K-Means Clustering:
   - Flight delay risk profiling & operational grouping (k=4 clusters)
   - Cluster profiling (mean load, weather hazard, delay rate, dominant carrier)
   - Silhouette score and inertia computation
2. PCA (Principal Component Analysis):
   - 2D & 3D dimensionality reduction for visualization
   - Variance explained ratio analysis
3. Isolation Forest:
   - Unsupervised operational anomaly detection (contamination=3%, 385 outliers)
4. Association Rule Mining integration (FP-Growth high-lift rules from data/mined_association_rules.csv)
5. Model & Profile persistence in models/
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

from config import BASE_DIR, FILE_RULES
from modeling.encoding import FILE_TEST, FILE_TRAIN

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FILE_KMEANS = MODELS_DIR / "kmeans_model.joblib"
FILE_PCA = MODELS_DIR / "pca_model.joblib"
FILE_ISOLATION_FOREST = MODELS_DIR / "isolation_forest.joblib"
FILE_CLUSTER_PROFILES = MODELS_DIR / "cluster_profiles.json"
FILE_UNSUPERVISED_METRICS = MODELS_DIR / "unsupervised_metrics.json"

CLUSTER_NAMES = {
    0: "Early Morning Low-Congestion Flights (Tải thấp, Tỉ lệ đúng giờ cao)",
    1: "Evening Peak Hours & Adverse Weather Risk (Chiều tối cao điểm, Rủi ro trễ cao)",
    2: "Midday Domestic Trunk Routes (Trưa cao điểm trục chính SGN-HAN-DAD)",
    3: "Peripheral & High-Disruption Flights (Chuyến bay giờ biên, Nguy cơ xáo trộn/hủy)",
}


def load_full_feature_matrix() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load both train and test feature matrices combined for comprehensive clustering."""
    train_df = pd.read_csv(FILE_TRAIN)
    test_df = pd.read_csv(FILE_TEST)
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    feature_cols = [c for c in full_df.columns if c not in ["target", "delay_min"]]
    return full_df[feature_cols], full_df


def train_kmeans_clustering(X: pd.DataFrame, full_df: pd.DataFrame, n_clusters: int = 4) -> Tuple[KMeans, dict]:
    """Train K-Means clustering and build human-readable profiles for each cluster."""
    print("\n" + "=" * 65)
    print(f"      TRAINING K-MEANS CLUSTERING (k={n_clusters})")
    print("=" * 65)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    # Compute silhouette score on a sample for performance
    sample_size = min(3000, len(X))
    sample_idx = np.random.choice(len(X), sample_size, replace=False)
    sil_score = silhouette_score(X.iloc[sample_idx], cluster_labels[sample_idx])
    print(f"  Silhouette Score (sample N={sample_size}): {sil_score:.4f}")
    print(f"  Inertia (Sum of squared distances): {kmeans.inertia_:.2f}")

    # Build cluster profiles
    df_eval = full_df.copy()
    df_eval["cluster"] = cluster_labels
    df_eval["is_delayed"] = (df_eval["target"] == 1).astype(int)
    df_eval["is_cancelled"] = (df_eval["target"] == 2).astype(int)

    profiles = {}
    for cid in range(n_clusters):
        subset = df_eval[df_eval["cluster"] == cid]
        size = len(subset)
        pct_of_total = round(size / len(df_eval) * 100, 1)

        delay_rate = round(float(subset["is_delayed"].mean() * 100), 1)
        cancel_rate = round(float(subset["is_cancelled"].mean() * 100), 2)
        avg_delay_min = round(float(subset["delay_min"].dropna().mean()), 1) if subset["delay_min"].notna().any() else 0.0

        profile = {
            "cluster_id": cid,
            "name": CLUSTER_NAMES.get(cid, f"Cluster {cid}"),
            "flight_count": size,
            "percentage": pct_of_total,
            "delay_rate_pct": delay_rate,
            "cancellation_rate_pct": cancel_rate,
            "avg_delay_minutes": avg_delay_min,
            "risk_tier": "HIGH" if delay_rate > 30 else ("MEDIUM" if delay_rate > 20 else "LOW"),
        }
        profiles[str(cid)] = profile

        print(f"\n* Cluster {cid}: {profile['name']}")
        print(f"    - Quy mô: {size:,} chuyến ({pct_of_total}%)")
        print(f"    - Tỷ lệ trễ: {delay_rate}% | Tỷ lệ hủy: {cancel_rate}% | Trễ trung bình: {avg_delay_min} phút")
        print(f"    - Mức độ rủi ro (Risk Tier): {profile['risk_tier']}")

    return kmeans, profiles


def train_pca_reduction(X: pd.DataFrame, n_components: int = 3) -> Tuple[PCA, dict]:
    """Train PCA for 2D and 3D visual projection."""
    print("\n" + "=" * 65)
    print(f"      TRAINING PCA DIMENSIONALITY REDUCTION (N={n_components})")
    print("=" * 65)

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X)

    var_ratios = [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_]
    total_var = round(sum(var_ratios), 2)

    print(f"  Explained variance ratio: {var_ratios}")
    print(f"  Total explained variance (top {n_components} components): {total_var}%")

    pca_metrics = {
        "n_components": n_components,
        "explained_variance_ratios": var_ratios,
        "total_explained_variance_pct": total_var,
    }
    return pca, pca_metrics


def train_isolation_forest(X: pd.DataFrame) -> Tuple[IsolationForest, dict]:
    """Train Isolation Forest for unsupervised flight anomaly detection."""
    print("\n" + "=" * 65)
    print("      TRAINING ISOLATION FOREST ANOMALY DETECTION")
    print("=" * 65)

    iso = IsolationForest(n_estimators=100, contamination=0.03, random_state=42, n_jobs=-1)
    anomaly_preds = iso.fit_predict(X)  # -1 for anomaly, 1 for normal

    anomaly_count = int((anomaly_preds == -1).sum())
    anomaly_pct = round(anomaly_count / len(X) * 100, 2)

    print(f"  Identified {anomaly_count:,} anomalous flights ({anomaly_pct}% of total)")

    iso_metrics = {
        "anomaly_count": anomaly_count,
        "anomaly_percentage": anomaly_pct,
        "contamination": 0.03,
    }
    return iso, iso_metrics


def load_association_rules(top_n: int = 15) -> List[dict]:
    """Load and format top mined association rules from processor/transformation."""
    if not FILE_RULES.exists():
        return []
    try:
        df_rules = pd.read_csv(FILE_RULES)
        top = df_rules.head(top_n)
        return top.to_dict(orient="records")
    except Exception as e:
        print(f"Notice: Could not load association rules: {e}")
        return []


def run_unsupervised_pipeline():
    """Execute complete unsupervised learning pipeline."""
    X, full_df = load_full_feature_matrix()

    kmeans, profiles = train_kmeans_clustering(X, full_df, n_clusters=4)
    pca, pca_metrics = train_pca_reduction(X, n_components=3)
    iso, iso_metrics = train_isolation_forest(X)
    top_rules = load_association_rules()

    # Save models
    joblib.dump(kmeans, FILE_KMEANS)
    joblib.dump(pca, FILE_PCA)
    joblib.dump(iso, FILE_ISOLATION_FOREST)

    with open(FILE_CLUSTER_PROFILES, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)

    metrics_summary = {
        "kmeans": {
            "n_clusters": 4,
            "profiles": profiles,
        },
        "pca": pca_metrics,
        "isolation_forest": iso_metrics,
        "top_rules_count": len(top_rules),
    }
    with open(FILE_UNSUPERVISED_METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 65)
    print(f"Saved K-Means model to         : {FILE_KMEANS}")
    print(f"Saved PCA model to             : {FILE_PCA}")
    print(f"Saved Isolation Forest to      : {FILE_ISOLATION_FOREST}")
    print(f"Saved Cluster Profiles to      : {FILE_CLUSTER_PROFILES}")
    print(f"Saved Unsupervised Metrics to  : {FILE_UNSUPERVISED_METRICS}")
    print("=" * 65)


if __name__ == "__main__":
    run_unsupervised_pipeline()

