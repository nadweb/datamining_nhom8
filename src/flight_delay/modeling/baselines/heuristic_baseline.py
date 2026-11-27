# -*- coding: utf-8 -*-
"""[STEP 4] Domain Operational Heuristic Baseline (Zero-ML Benchmark)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class HeuristicBaselineClassifier(BaseEstimator, ClassifierMixin):
    """Rule-based aviation disruption heuristic using pre-flight rain, wind, LCC, and peak load.

    Produces calibrated-like multi-class [P(ON_TIME), P(DELAYED), P(CANCELLED)] or binary
    [P(ON_TIME), P(DISRUPTED)] probabilities strictly from observable pre-flight inputs.
    """

    def __init__(self, n_classes: int = 3):
        self.n_classes = int(n_classes)

    def fit(self, X: pd.DataFrame, y=None):
        self.classes_ = np.arange(self.n_classes)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        n = len(df)
        score = np.full(n, 0.16, dtype=float)

        def _col(name: str) -> np.ndarray:
            return df[name].to_numpy(dtype=float) if name in df.columns else np.zeros(n)

        score += 0.12 * _col("is_peak_hour")
        score += 0.10 * _col("is_heavy_traffic")
        score += 0.10 * _col("is_rain")
        score += 0.16 * _col("is_heavy_rain")
        score += 0.12 * _col("is_high_wind")
        score += 0.18 * _col("is_storm")
        score += 0.10 * _col("carrier_category_LCC")
        score += 0.08 * _col("airline_code_VJ")
        p_bad = np.clip(score, 0.05, 0.88)

        if self.n_classes == 2:
            return np.column_stack([1.0 - p_bad, p_bad])

        # For 3-class: extreme storm/wind carries minor cancellation risk share
        p_cancel = np.clip(0.005 + 0.04 * _col("is_storm") + 0.02 * _col("is_severe_wind"), 0.002, 0.15)
        p_cancel = np.minimum(p_cancel, p_bad * 0.25)
        p_delay = np.clip(p_bad - p_cancel, 0.01, 0.90)
        p_ontime = np.clip(1.0 - p_delay - p_cancel, 0.05, 0.95)
        mat = np.column_stack([p_ontime, p_delay, p_cancel])
        return mat / mat.sum(axis=1, keepdims=True)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
