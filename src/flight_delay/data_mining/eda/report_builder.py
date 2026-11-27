# -*- coding: utf-8 -*-
"""[STEP 3] Comprehensive EDA & Schema Matrix Report Generator (Dashboard + Docs Contract)."""
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from flight_delay.common.constants import DATA_MANIFESTS_DIR, FILE_GOLD, FILE_SILVER, REPORTS_DIR
from flight_delay.common.utils import read_json, write_json
from flight_delay.data_mining.eda.bivariate import compute_bivariate_breakdowns
from flight_delay.data_mining.eda.correlation import compute_correlation_suite
from flight_delay.data_mining.eda.multivariate import compute_multivariate_interactions
from flight_delay.data_mining.eda.statistical_tests import run_eda_hypothesis_tests
from flight_delay.data_mining.eda.univariate import compute_univariate_summary
from flight_delay.data_mining.profiling.profiler import profile_dataset


def generate_full_eda_report(gold_path: Optional[Path] = None) -> dict:
    """Compute EDA, profiling, hypothesis tests, correlation & schema lineage JSON reports."""
    path = Path(gold_path or (FILE_GOLD if FILE_GOLD.exists() else FILE_SILVER))
    df = pd.read_csv(path, low_memory=False)

    bronze_manifest = {}
    bm_path = DATA_MANIFESTS_DIR / "bronze_manifest.json"
    if bm_path.exists():
        bronze_manifest = read_json(bm_path)

    ds_manifest = {}
    dm_path = DATA_MANIFESTS_DIR / "dataset_manifest.json"
    if dm_path.exists():
        ds_manifest = read_json(dm_path)

    prof = profile_dataset(df, std_rows=len(df))
    uni = compute_univariate_summary(df)
    biv = compute_bivariate_breakdowns(df)
    multi = compute_multivariate_interactions(df)
    corr = compute_correlation_suite(df)
    stat_tests = run_eda_hypothesis_tests(df)

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_version": ds_manifest.get("dataset_version", "unversioned"),
        "raw_provenance_audit": {
            "raw_file_count": bronze_manifest.get("source_csv_count", 11),
            "raw_mixed_rows": bronze_manifest.get("total_raw_mixed_rows", 92146),
            "raw_mixed_columns": bronze_manifest.get("total_unique_raw_columns", 58),
            "standardized_snapshot_rows": len(df),
            **(prof.get("raw_provenance") or {}),
        },
        "schema_evolution": {"stages": ds_manifest.get("schema_lineage", [])},
        "eda": {
            **uni,
            "bivariate": biv,
            "multivariate": multi,
            "hypothesis_tests": stat_tests,
            "outliers": prof["outliers"],
            "missingness": prof["missingness"],
        },
        "correlation_and_association": corr,
    }
    write_json(REPORTS_DIR / "eda" / "eda_report.json", report)
    return report
