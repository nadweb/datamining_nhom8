# -*- coding: utf-8 -*-
"""Publication-grade CRISP-DM Figures Generator saved under artifacts/figures/."""
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from flight_delay.common.constants import FIGURES_DIR, FILE_GOLD, REPORTS_DIR
from flight_delay.common.utils import read_json


def generate_all_figures() -> Dict[str, str]:
    """Render clean high-DPI analytical and model figures from real Gold + Evaluation artifacts."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")
    saved: Dict[str, str] = {}

    gold = pd.read_csv(FILE_GOLD, low_memory=False)
    cls_rep = read_json(REPORTS_DIR / "experiments" / "classification_results.json")
    reg_rep = read_json(REPORTS_DIR / "experiments" / "regression_results.json")
    unsup_rep = read_json(REPORTS_DIR / "experiments" / "unsupervised_results.json")
    drift_rep = read_json(REPORTS_DIR / "monitoring" / "drift_report.json")

    # 1. Class distribution 3-class ICAO across splits
    fig, ax = plt.subplots(figsize=(8, 4.6))
    order_splits = ["train", "val", "test"]
    counts = pd.crosstab(gold["split_role"], gold["status_clean"]).reindex(order_splits)
    counts_pct = counts.div(counts.sum(axis=1), axis=0) * 100.0
    counts_pct[["on_time", "delayed", "cancelled"]].plot(
        kind="bar", stacked=True, color=["#059669", "#d97706", "#e11d48"], ax=ax
    )
    ax.set_title("Chronological Split Outcome Distribution (Train 70% / Val 10% / Test 20%)")
    ax.set_ylabel("Percentage of Flights (%)")
    ax.set_xlabel("Chronological Partition")
    ax.legend(title="ICAO Outcome", loc="upper right")
    fig.tight_layout()
    p1 = FIGURES_DIR / "01_class_distribution_by_split.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    saved["class_distribution"] = str(p1)

    # 2. 3-Class Primary vs Binary Ablation Model Comparison
    mc = cls_rep["multiclass_3"]["test_comparison"]
    bn = cls_rep["binary_disrupted"]["test_comparison"]
    models = ["MajorityClassifier", "HeuristicBaseline", "HistGradientBoosting", "XGBoost", "CatBoost"]
    x = np.arange(len(models))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - w / 2, [mc[m]["f1_macro"] for m in models], w, label="3-Class Macro-F1 (Primary)", color="#0d5c58")
    ax.bar(x + w / 2, [bn[m]["f1_macro"] for m in models], w, label="Binary Macro-F1 (Ablation)", color="#0284c7")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Macro-F1 (Holdout Test)")
    ax.set_title("Model Benchmark: Primary 3-Class ICAO vs Binary Disrupted Ablation")
    ax.legend()
    fig.tight_layout()
    p2 = FIGURES_DIR / "02_classification_model_comparison.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    saved["classification_comparison"] = str(p2)

    # 3. 3-Class Confusion Matrix of Production Classifier
    best_name = cls_rep["multiclass_3"]["best_model"]
    cm = np.array(mc[best_name]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Teal" if hasattr(plt.cm, "Teal") else "GnBu",
        xticklabels=["on_time", "delayed", "cancelled"],
        yticklabels=["on_time", "delayed", "cancelled"],
        ax=ax,
    )
    ax.set_title(f"Holdout Confusion Matrix ({best_name} — 3-Class ICAO)")
    ax.set_ylabel("Actual Outcome")
    ax.set_xlabel("Predicted Outcome")
    fig.tight_layout()
    p3 = FIGURES_DIR / "03_confusion_matrix_3class.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    saved["confusion_matrix"] = str(p3)

    # 4. SHAP Global Feature Importance Top 12
    shap_top = cls_rep["multiclass_3"].get("shap_summary", {}).get("global_top_features", [])[:12]
    if shap_top:
        fig, ax = plt.subplots(figsize=(8.2, 5.0))
        feats = [r["feature"] for r in shap_top][::-1]
        vals = [r["mean_abs_shap"] for r in shap_top][::-1]
        ax.barh(feats, vals, color="#0f766e")
        ax.set_title(f"Top 12 Global Mean(|SHAP|) Drivers ({best_name})")
        ax.set_xlabel("Mean(|SHAP value|)")
        fig.tight_layout()
        p4 = FIGURES_DIR / "04_shap_top_features.png"
        fig.savefig(p4, dpi=150)
        plt.close(fig)
        saved["shap_importance"] = str(p4)

    # 5. Regression MAE Comparison
    reg_comp = reg_rep["test_comparison"]
    reg_names = list(reg_comp.keys())
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    colors = ["#94a3b8" if "Median" in r else "#0d5c58" if r == reg_rep["best_model"] else "#0284c7" for r in reg_names]
    ax.bar(reg_names, [reg_comp[r]["MAE_minutes"] for r in reg_names], color=colors)
    ax.set_ylabel("MAE (Minutes)")
    ax.set_title("Delay Duration Regression MAE (Completed Flights Only)")
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    p5 = FIGURES_DIR / "05_regression_mae_comparison.png"
    fig.savefig(p5, dpi=150)
    plt.close(fig)
    saved["regression_mae"] = str(p5)

    # 6. K-Means K-Sweep (Silhouette & Davies-Bouldin)
    sweep = unsup_rep["kmeans"]["k_sweep"]
    fig, ax1 = plt.subplots(figsize=(7.8, 4.6))
    ks = [r["k"] for r in sweep]
    ax1.plot(ks, [r["silhouette"] for r in sweep], marker="o", color="#0d5c58", label="Silhouette (higher=better)")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Silhouette Score", color="#0d5c58")
    ax2 = ax1.twinx()
    ax2.plot(ks, [r["davies_bouldin"] for r in sweep], marker="s", linestyle="--", color="#e11d48", label="Davies-Bouldin (lower=better)")
    ax2.set_ylabel("Davies-Bouldin Index", color="#e11d48")
    ax1.set_title(f"K-Means Cluster Validation (Selected k=4, Seed ARI={unsup_rep['kmeans']['seed_stability']['ari_mean']})")
    fig.tight_layout()
    p6 = FIGURES_DIR / "06_kmeans_validation_curve.png"
    fig.savefig(p6, dpi=150)
    plt.close(fig)
    saved["kmeans_sweep"] = str(p6)

    # 7. PSI Feature Drift Plot
    fdrift = drift_rep["data_drift"]["feature_drift"]
    df_psi = pd.DataFrame([{"feature": k, "psi": v["psi"]} for k, v in fdrift.items()]).sort_values("psi")
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    cols = ["#e11d48" if v >= 0.25 else "#d97706" if v >= 0.10 else "#059669" for v in df_psi["psi"]]
    ax.barh(df_psi["feature"], df_psi["psi"], color=cols)
    ax.axvline(0.10, color="#d97706", linestyle="--", label="Warning PSI = 0.10")
    ax.axvline(0.25, color="#e11d48", linestyle="--", label="Retrain PSI = 0.25")
    ax.set_xlabel("Population Stability Index (PSI)")
    ax.set_title("Production Feature Drift Monitoring (Train vs Holdout Test)")
    ax.legend()
    fig.tight_layout()
    p7 = FIGURES_DIR / "07_feature_drift_psi.png"
    fig.savefig(p7, dpi=150)
    plt.close(fig)
    saved["feature_drift_psi"] = str(p7)

    return saved
