# -*- coding: utf-8 -*-
"""[STEP 7] Population Stability Index (PSI) Utility."""
import numpy as np


def compute_psi(expected, actual, bins: int = 10) -> float:
    """Compute quantile-binned Population Stability Index with epsilon smoothing."""
    exp = np.asarray(expected, dtype=float)
    act = np.asarray(actual, dtype=float)
    exp = exp[np.isfinite(exp)]
    act = act[np.isfinite(act)]
    if len(exp) == 0 or len(act) == 0:
        return 0.0

    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(exp, quantiles))
    if len(edges) < 3:
        lo = min(float(exp.min()), float(act.min())) - 1e-6
        hi = max(float(exp.max()), float(act.max())) + 1e-6
        edges = np.linspace(lo, hi, bins + 1)
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf

    exp_cnt, _ = np.histogram(exp, bins=edges)
    act_cnt, _ = np.histogram(act, bins=edges)
    exp_pct = np.clip(exp_cnt / max(len(exp), 1), 1e-4, 1.0)
    act_pct = np.clip(act_cnt / max(len(act), 1), 1e-4, 1.0)
    psi_val = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return round(psi_val, 4)
