# -*- coding: utf-8 -*-
"""Data validation script and CLI command.
Validates that flights_cleaned_global.csv (or sample) conforms strictly to the data contract:
1. Exactly 24 columns in exact order
2. Unique leg_key (1 row = 1 flight leg)
3. Labels strictly within {'on_time', 'delayed', 'cancelled'}
4. No missing status_clean
5. delay_min is NaN for all cancelled flights
6. delay_min >= -90 for all non-cancelled flights with recorded delays
7. Status consistency between status string and status_clean
8. Demonstrates multi-source data (>= 2 distinct sources)
"""
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union

# Ensure root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from config import CLEANED_FIELDS, FILE_CLEANED

LABELS = {"on_time", "delayed", "cancelled"}
EARLY_LIMIT = -90.0


def validate_flights_data(
    data_source: Union[str, Path, pd.DataFrame] = FILE_CLEANED
) -> Tuple[bool, List[Dict[str, Union[str, bool]]], Dict[str, Union[int, float]]]:
    """Validate flight dataset against data quality constraints.

    Args:
        data_source: Path to CSV or pandas DataFrame.

    Returns:
        Tuple of (all_passed: bool, checks: List[dict], metrics: dict)
    """
    if isinstance(data_source, (str, Path)):
        p = Path(data_source)
        if not p.exists():
            # Fallback to sample if global doesn't exist
            sample_p = p.parent / "flights_sample.csv"
            if sample_p.exists():
                print(f"[check-data] File '{p.name}' not found. Falling back to sample: '{sample_p.name}'")
                p = sample_p
            else:
                return False, [{"check": f"File exists: {p}", "passed": False, "detail": "File not found"}], {}
        df = pd.read_csv(p)
    else:
        df = data_source.copy()

    total_rows = len(df)
    if total_rows == 0:
        return False, [{"check": "Non-empty dataset", "passed": False, "detail": "Dataset has 0 rows"}], {}

    checks = []

    # 1. 24 columns in right order
    cols_match = list(df.columns) == CLEANED_FIELDS
    checks.append({
        "check": "24 columns in exact schema order",
        "passed": cols_match,
        "detail": f"{len(df.columns)}/24 columns matching" if cols_match else f"Columns mismatch. Missing: {set(CLEANED_FIELDS) - set(df.columns)}"
    })

    # 2. Unique leg_key
    is_unique_leg = bool(df["leg_key"].is_unique) if "leg_key" in df.columns else False
    dup_count = int(df["leg_key"].duplicated().sum()) if "leg_key" in df.columns else total_rows
    checks.append({
        "check": "Unique leg_key (1 row = 1 flight leg)",
        "passed": is_unique_leg,
        "detail": "All unique" if is_unique_leg else f"{dup_count} duplicate keys found"
    })

    # 3. Valid labels in status_clean
    labels_present = set(df["status_clean"].dropna().unique()) if "status_clean" in df.columns else set()
    valid_labels = labels_present <= LABELS and len(labels_present) > 0
    checks.append({
        "check": "Labels strictly in {on_time, delayed, cancelled}",
        "passed": valid_labels,
        "detail": f"Observed labels: {labels_present}"
    })

    # 4. No null status_clean
    no_null_status = bool(df["status_clean"].notna().all()) if "status_clean" in df.columns else False
    null_status_cnt = int(df["status_clean"].isna().sum()) if "status_clean" in df.columns else total_rows
    checks.append({
        "check": "No missing status_clean target values",
        "passed": no_null_status,
        "detail": "0 missing" if no_null_status else f"{null_status_cnt} nulls"
    })

    # 5. Cancelled flights must have null delay_min
    if "status_clean" in df.columns and "delay_min" in df.columns:
        cancelled_delays = df.loc[df["status_clean"] == "cancelled", "delay_min"]
        no_cancelled_delay = bool(cancelled_delays.isna().all())
        bad_cnt = int(cancelled_delays.notna().sum())
    else:
        no_cancelled_delay = False
        bad_cnt = -1
    checks.append({
        "check": "Cancelled flights have no delay_min (must be NaN)",
        "passed": no_cancelled_delay,
        "detail": "All cancelled flights have null delay_min" if no_cancelled_delay else f"{bad_cnt} cancelled flights have non-null delay"
    })

    # 6. delay_min >= EARLY_LIMIT (-90 min)
    if "delay_min" in df.columns:
        valid_delay = df["delay_min"].dropna()
        no_extreme_early = bool((valid_delay >= EARLY_LIMIT).all())
        outliers = int((valid_delay < EARLY_LIMIT).sum())
    else:
        no_extreme_early = False
        outliers = -1
    checks.append({
        "check": f"delay_min >= {EARLY_LIMIT} minutes (plausible delay bound)",
        "passed": no_extreme_early,
        "detail": "Within valid bounds" if no_extreme_early else f"{outliers} flights departed >90 min early"
    })

    # 7. Status label consistency
    if "status" in df.columns and "status_clean" in df.columns:
        cancelled_sync = bool(((df["status_clean"] == "cancelled") == (df["status"] == "Cancelled")).all())
    else:
        cancelled_sync = False
    checks.append({
        "check": "Consistency between status string and status_clean",
        "passed": cancelled_sync,
        "detail": "Consistent" if cancelled_sync else "Mismatch between text status and label"
    })

    # 8. Multi-source proof
    if "source" in df.columns:
        num_sources = int(df["source"].nunique())
        multi_source = num_sources >= 2
    else:
        num_sources = 0
        multi_source = False
    checks.append({
        "check": "Multi-source integration (>= 2 distinct data sources)",
        "passed": multi_source,
        "detail": f"{num_sources} sources recorded ({list(df['source'].unique()) if 'source' in df.columns else []})"
    })

    all_passed = all(c["passed"] for c in checks)

    metrics = {
        "total_rows": total_rows,
        "total_cols": len(df.columns),
        "on_time_count": int((df["status_clean"] == "on_time").sum()) if "status_clean" in df.columns else 0,
        "delayed_count": int((df["status_clean"] == "delayed").sum()) if "status_clean" in df.columns else 0,
        "cancelled_count": int((df["status_clean"] == "cancelled").sum()) if "status_clean" in df.columns else 0,
        "sources": list(df["source"].unique()) if "source" in df.columns else [],
    }

    return all_passed, checks, metrics


def run_cli_check(target_path=None) -> int:
    """CLI runner for data validation."""
    path = target_path or FILE_CLEANED
    print("=" * 70)
    print("          VIETNAM FLIGHT DATA CONTRACT & QUALITY CHECK")
    print("=" * 70)
    print(f"Target file: {path}")

    all_passed, checks, metrics = validate_flights_data(path)

    print("-" * 70)
    for i, c in enumerate(checks, 1):
        status_tag = "[PASS]" if c["passed"] else "[FAIL]"
        print(f"{i:2d}. {status_tag} {c['check']}")
        print(f"    --> {c['detail']}")

    print("-" * 70)
    if metrics:
        print("Dataset Summary:")
        print(f"  * Total rows: {metrics['total_rows']:,}")
        print(f"  * Target counts:")
        print(f"      - on_time   : {metrics['on_time_count']:,} ({metrics['on_time_count']/metrics['total_rows']*100:.1f}%)")
        print(f"      - delayed   : {metrics['delayed_count']:,} ({metrics['delayed_count']/metrics['total_rows']*100:.1f}%)")
        print(f"      - cancelled : {metrics['cancelled_count']:,} ({metrics['cancelled_count']/metrics['total_rows']*100:.1f}%)")
        print(f"  * Sources ({len(metrics['sources'])}): {', '.join(metrics['sources'])}")

    print("=" * 70)
    if all_passed:
        print(">>> ALL DATA QUALITY CHECKS PASSED SUCCESSFULLY. DATA CONTRACT IS VALID.")
        print("=" * 70)
        return 0
    else:
        print(">>> CRITICAL: SOME CHECKS FAILED. PLEASE FIX DATA ANOMALIES BEFORE TRAINING.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(run_cli_check(target))

