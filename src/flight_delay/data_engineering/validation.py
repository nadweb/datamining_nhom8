# -*- coding: utf-8 -*-
"""[STEP 2] 8-Gate Data Contract Validator for Cleaned 24-Column Silver Datasets."""
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from flight_delay.common.constants import (
    CLEANED_FIELDS,
    EARLY_LIMIT,
    FILE_CLEANED,
    FILE_SILVER,
    VALID_OUTCOME_LABELS,
)

LABELS = set(VALID_OUTCOME_LABELS)


def validate_flights_data(
    data_source: Union[str, Path, pd.DataFrame] = FILE_CLEANED,
) -> Tuple[bool, List[Dict[str, Union[str, bool]]], Dict[str, Union[int, float, list]]]:
    """Run 8 mandatory data contract gates on the 24-col cleaned flight dataset."""
    if isinstance(data_source, (str, Path)):
        path = Path(data_source)
        if not path.exists() and path == FILE_CLEANED and FILE_SILVER.exists():
            path = FILE_SILVER
        if not path.exists():
            return False, [{"check": "Dataset file exists", "passed": False, "detail": f"File not found: {path}"}], {}
        df = pd.read_csv(path, low_memory=False)
    else:
        df = data_source.copy()

    n = len(df)
    if n == 0:
        return False, [{"check": "Non-empty dataset", "passed": False, "detail": "Dataset has 0 rows"}], {}

    cols = set(df.columns)
    missing = sorted(set(CLEANED_FIELDS) - cols)
    checks: List[Dict[str, Union[str, bool]]] = [
        {
            "check": "24 columns in exact schema order",
            "passed": list(df.columns) == CLEANED_FIELDS,
            "detail": "Exact 24-column schema matched" if list(df.columns) == CLEANED_FIELDS else f"missing={missing}; unexpected={sorted(cols - set(CLEANED_FIELDS))}",
        }
    ]

    has_key = "leg_key" in cols
    dup_count = int(df["leg_key"].astype("string").duplicated().sum()) if has_key else n
    key_ok = has_key and df["leg_key"].notna().all() and df["leg_key"].astype("string").str.strip().ne("").all() and dup_count == 0
    checks.append({"check": "Unique leg_key (1 row = 1 flight leg; non-empty)", "passed": bool(key_ok), "detail": f"duplicates={dup_count}"})

    has_label = "status_clean" in cols
    labels = set(df["status_clean"].dropna().astype(str).str.strip().str.lower()) if has_label else set()
    checks.append({"check": "Labels strictly in {on_time, delayed, cancelled}", "passed": bool(has_label and labels and labels <= LABELS), "detail": f"Observed labels: {sorted(labels)}"})

    no_null = has_label and bool(df["status_clean"].notna().all())
    checks.append({"check": "No missing status_clean target values", "passed": bool(no_null), "detail": "0 missing" if no_null else "has missing"})

    has_delay = "delay_min" in cols
    cancel_delay = int(df.loc[df["status_clean"].astype("string").str.lower() == "cancelled", "delay_min"].notna().sum()) if has_delay and has_label else -1
    checks.append({"check": "Cancelled flights have no delay_min (must be NaN)", "passed": bool(cancel_delay == 0 and has_delay and has_label), "detail": f"cancelled_with_delay={cancel_delay}"})

    if has_delay:
        delays = pd.to_numeric(df["delay_min"], errors="coerce")
        invalid_delay = int((delays.dropna() < EARLY_LIMIT).sum())
        delay_ok = invalid_delay == 0
    else:
        invalid_delay, delay_ok = -1, False
    checks.append({"check": f"delay_min >= {EARLY_LIMIT} minutes (plausible delay bound)", "passed": bool(delay_ok), "detail": f"below_bound={invalid_delay}"})

    if {"status", "status_clean"} <= cols:
        text_cancelled = df["status"].fillna("").astype(str).str.contains(r"cancel|huy|hủy", case=False, regex=True)
        label_cancelled = df["status_clean"].fillna("").astype(str).str.lower().eq("cancelled")
        mismatch = int((text_cancelled != label_cancelled).sum())
        status_ok = mismatch == 0
    else:
        mismatch, status_ok = -1, False
    checks.append({"check": "Consistency between status string and status_clean", "passed": bool(status_ok), "detail": f"mismatch={mismatch}"})

    if "source" in cols:
        sources = sorted(df["source"].dropna().astype(str).str.strip().replace("", np.nan).dropna().unique().tolist())
        source_ok = len(sources) >= 2
    else:
        sources, source_ok = [], False
    checks.append({"check": "Multi-source integration (>= 2 distinct data sources)", "passed": bool(source_ok), "detail": f"{len(sources)} sources: {sources}"})

    metrics = {
        "total_rows": n,
        "total_cols": len(df.columns),
        "on_time_count": int((df["status_clean"] == "on_time").sum()) if has_label else 0,
        "delayed_count": int((df["status_clean"] == "delayed").sum()) if has_label else 0,
        "cancelled_count": int((df["status_clean"] == "cancelled").sum()) if has_label else 0,
        "sources": sources,
    }
    return all(bool(c["passed"]) for c in checks), checks, metrics


def run_validation_report(target_path=None) -> int:
    path = Path(target_path) if target_path else (FILE_CLEANED if FILE_CLEANED.exists() else FILE_SILVER)
    passed, checks, metrics = validate_flights_data(path)
    for c in checks:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['check']} :: {c['detail']}")
    print("Metrics:", metrics)
    return 0 if passed else 1
