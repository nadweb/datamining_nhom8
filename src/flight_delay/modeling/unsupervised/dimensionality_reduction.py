# -*- coding: utf-8 -*-
"""[STEP 4] PCA Dimensionality Reduction & Factor Loading Interpretation."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from flight_delay.common.constants import EXPERIMENTS_CFG, RANDOM_STATE


def run_pca_analysis(X_scaled: np.ndarray, feature_names: list, cluster_labels: np.ndarray, gold_df: pd.DataFrame) -> dict:
    n_comp = min(int(EXPERIMENTS_CFG["unsupervised"]["pca"]["n_components"]), X_scaled.shape[1])
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)

    loadings = []
    for i in range(n_comp):
        vec = pca.components_[i]
        top_idx = np.argsort(np.abs(vec))[::-1][:5]
        loadings.append({
            "component": f"PC{i + 1}",
            "explained_variance_pct": round(float(pca.explained_variance_ratio_[i] * 100.0), 2),
            "top_features": [
                {"feature": feature_names[j], "loading": round(float(vec[j]), 4)}
                for j in top_idx
            ],
        })

    # Sample up to 600 points for interactive visualization in Web / Reports
    rng = np.random.default_rng(RANDOM_STATE)
    n_pts = min(600, len(coords))
    idx = np.sort(rng.choice(len(coords), size=n_pts, replace=False))
    scatter_sample = [
        {
            "pc1": round(float(coords[k, 0]), 3),
            "pc2": round(float(coords[k, 1]), 3),
            "pc3": round(float(coords[k, 2]), 3) if coords.shape[1] >= 3 else 0.0,
            "cluster": int(cluster_labels[k]),
            "status": str(gold_df.iloc[k]["status_clean"]),
        }
        for k in idx
    ]

    return {
        "n_components": n_comp,
        "explained_variance_ratio": [round(float(x), 4) for x in pca.explained_variance_ratio_],
        "cumulative_variance_pct": round(float(np.sum(pca.explained_variance_ratio_) * 100.0), 2),
        "components": loadings,
        "scatter_sample": scatter_sample,
    }
