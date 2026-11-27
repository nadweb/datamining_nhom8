# -*- coding: utf-8 -*-
"""[STEP 5] Statistical Model Comparison (Paired Bootstrap CI, McNemar Test, Permutation Test)."""
import numpy as np
from scipy.stats import binomtest, chi2
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error


def paired_bootstrap_ci(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    metric: str = "f1_macro",
    n_boot: int = 400,
    seed: int = 42,
) -> dict:
    """Paired bootstrap confidence interval for (model_b - model_a)."""
    yt = np.asarray(y_true)
    pa = np.asarray(preds_a)
    pb = np.asarray(preds_b)
    n = len(yt)
    if n == 0:
        return {"mean_diff": 0.0, "ci_95": [0.0, 0.0]}

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if metric == "mae":
            diffs[b] = mean_absolute_error(yt[idx], pb[idx]) - mean_absolute_error(yt[idx], pa[idx])
        else:
            diffs[b] = f1_score(yt[idx], pb[idx], average="macro", zero_division=0) - f1_score(
                yt[idx], pa[idx], average="macro", zero_division=0
            )
    return {
        "metric": metric,
        "n_bootstrap": n_boot,
        "mean_diff_b_minus_a": round(float(diffs.mean()), 4),
        "ci_95": [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)],
        "significant_005": bool(np.percentile(diffs, 2.5) > 0 or np.percentile(diffs, 97.5) < 0),
    }


def mcnemar_test(y_true: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray) -> dict:
    """McNemar's test comparing correctness of two classifiers on the exact same test instances."""
    yt = np.asarray(y_true)
    ca = np.asarray(preds_a) == yt
    cb = np.asarray(preds_b) == yt
    n01 = int(np.sum(~ca & cb))  # A wrong, B right
    n10 = int(np.sum(ca & ~cb))  # A right, B wrong
    n_disc = n01 + n10
    if n_disc == 0:
        return {"n01_b_better": 0, "n10_a_better": 0, "statistic": 0.0, "p_value": 1.0, "significant_005": False}
    if n_disc < 25:
        pval = float(binomtest(min(n01, n10), n_disc, 0.5).pvalue)
        stat = float((abs(n01 - n10) - 1) ** 2 / n_disc)
    else:
        stat = float((abs(n01 - n10) - 1) ** 2 / n_disc)
        pval = float(1.0 - chi2.cdf(stat, df=1))
    return {
        "n01_b_better": n01,
        "n10_a_better": n10,
        "statistic": round(stat, 4),
        "p_value": round(pval, 6),
        "significant_005": bool(pval < 0.05),
    }


def permutation_significance_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_permutations: int = 200,
    seed: int = 42,
) -> dict:
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    obs = float(balanced_accuracy_score(yt, yp))
    rng = np.random.default_rng(seed)
    null_scores = np.array([balanced_accuracy_score(rng.permutation(yt), yp) for _ in range(n_permutations)], dtype=float)
    p_val = float((null_scores >= obs).mean())
    return {
        "n_permutations": n_permutations,
        "metric": "balanced_accuracy",
        "model_score": round(obs * 100, 2),
        "chance_level_mean": round(float(null_scores.mean() * 100), 2),
        "chance_level_std": round(float(null_scores.std() * 100), 2),
        "p_value": round(p_val, 4),
        "statistically_significant": bool(p_val < 0.05),
    }
