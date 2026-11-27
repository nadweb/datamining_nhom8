# -*- coding: utf-8 -*-
"""[EVALUATION & MONITORING] Statistical Comparison Tests & 4-Tier PSI/KS Monitoring."""
import numpy as np

from flight_delay.common.constants import REPORTS_DIR
from flight_delay.common.utils import read_json
from flight_delay.evaluation.statistical_comparison import (
    mcnemar_test,
    paired_bootstrap_ci,
    permutation_significance_test,
)
from flight_delay.monitoring.psi import compute_psi


def test_statistical_comparison_utilities():
    yt = np.array([0, 1, 0, 1, 0, 1, 2, 0, 1, 0] * 10)
    pa = np.zeros_like(yt)
    pb = yt.copy()
    pb[0] = 1
    mcn = mcnemar_test(yt, pa, pb)
    assert mcn["significant_005"] is True and mcn["n01_b_better"] > mcn["n10_a_better"]

    boot = paired_bootstrap_ci(yt, pa, pb, metric="f1_macro", n_boot=100, seed=42)
    assert boot["mean_diff_b_minus_a"] > 0.3

    perm = permutation_significance_test(yt, pb, n_permutations=60, seed=42)
    assert perm["statistically_significant"] is True


def test_monitoring_psi_and_drift_report():
    rng = np.random.default_rng(7)
    base = rng.normal(0, 1, 500)
    same = rng.normal(0, 1, 500)
    shifted = rng.normal(2.5, 1, 500)
    assert compute_psi(base, same) < 0.10
    assert compute_psi(base, shifted) > 0.25

    drift = read_json(REPORTS_DIR / "monitoring" / "drift_report.json")
    assert drift["overall_status"] in ("OK", "WARNING", "CRITICAL")
    assert "ks_p_value" in next(iter(drift["data_drift"]["feature_drift"].values()))
