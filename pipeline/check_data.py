# -*- coding: utf-8 -*-
"""Validate the canonical 24-column flight dataset against docs/schema.md."""
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from config import CLEANED_FIELDS, FILE_CLEANED

LABELS = {"on_time", "delayed", "cancelled"}
EARLY_LIMIT = -90.0


def validate_flights_data(
    data_source: Union[str, Path, pd.DataFrame] = FILE_CLEANED,
) -> Tuple[bool, List[Dict[str, Union[str, bool]]], Dict[str, Union[int, float, list]]]:
    """Return (all_passed, eight_check_results, summary_metrics).

    Missing columns are reported as failed checks; validation never crashes with
    KeyError when a malformed input schema is supplied.
    """
    if isinstance(data_source, (str, Path)):
        path = Path(data_source)
        if not path.exists():
            sample_path = path.parent / "flights_sample.csv"
            if sample_path.exists():
                print(f"[check-data] '{path.name}' not found; using '{sample_path.name}'.")
                path = sample_path
            else:
                return False, [{"check": "Dataset file exists", "passed": False, "detail": f"File not found: {path}"}], {}
        df = pd.read_csv(path)
    else:
        df = data_source.copy()

    n = len(df)
    if n == 0:
        return False, [{"check": "Non-empty dataset", "passed": False, "detail": "Dataset has 0 rows"}], {}

    cols = set(df.columns)
    missing = sorted(set(CLEANED_FIELDS) - cols)
    checks: List[Dict[str, Union[str, bool]]] = []
    checks.append({
        "check": "24 columns in exact schema order",
        "passed": list(df.columns) == CLEANED_FIELDS,
        "detail": "Exact 24-column schema matched" if list(df.columns) == CLEANED_FIELDS else f"Found {len(df.columns)} columns; missing={missing}; unexpected={sorted(cols-set(CLEANED_FIELDS))}",
    })

    has_key = "leg_key" in cols
    duplicate_count = int(df["leg_key"].astype("string").duplicated().sum()) if has_key else n
    key_ok = has_key and df["leg_key"].notna().all() and df["leg_key"].astype("string").str.strip().ne("").all() and duplicate_count == 0
    checks.append({"check": "Unique leg_key (1 row = 1 flight leg; non-empty)", "passed": bool(key_ok), "detail": "All keys are present and unique" if key_ok else f"Missing/empty key or {duplicate_count} duplicates"})

    has_label = "status_clean" in cols
    labels = set(df["status_clean"].dropna().astype(str).str.strip().str.lower()) if has_label else set()
    label_ok = has_label and bool(labels) and labels <= LABELS
    checks.append({"check": "Labels strictly in {on_time, delayed, cancelled}", "passed": bool(label_ok), "detail": f"Observed labels: {sorted(labels)}"})

    no_null = has_label and bool(df["status_clean"].notna().all())
    checks.append({"check": "No missing status_clean target values", "passed": bool(no_null), "detail": "0 missing" if no_null else f"{int(df['status_clean'].isna().sum()) if has_label else n} nulls"})

    has_delay = "delay_min" in cols
    has_label = "status_clean" in cols
    cancelled_delay_count = int(df.loc[df["status_clean"].astype("string").str.lower() == "cancelled", "delay_min"].notna().sum()) if has_delay and has_label else -1
    cancel_delay_ok = cancelled_delay_count == 0 and has_delay and has_label
    checks.append({"check": "Cancelled flights have no delay_min (must be NaN)", "passed": bool(cancel_delay_ok), "detail": "All cancelled flights have null delay_min" if cancel_delay_ok else f"{cancelled_delay_count} cancelled flights have non-null delay (or required columns missing)"})

    if has_delay:
        delays = pd.to_numeric(df["delay_min"], errors="coerce")
        invalid_delay = int((delays.dropna() < EARLY_LIMIT).sum())
        delay_ok = invalid_delay == 0
    else:
        invalid_delay, delay_ok = -1, False
    checks.append({"check": f"delay_min >= {EARLY_LIMIT} minutes (plausible delay bound)", "passed": bool(delay_ok), "detail": "Within valid bounds" if delay_ok else f"{invalid_delay} values below lower bound (or column missing)"})

    if {"status", "status_clean"} <= cols:
        text_cancelled = df["status"].fillna("").astype(str).str.contains(r"cancel|huy|hủy", case=False, regex=True)
        label_cancelled = df["status_clean"].fillna("").astype(str).str.lower().eq("cancelled")
        mismatch = int((text_cancelled != label_cancelled).sum())
        status_ok = mismatch == 0
    else:
        mismatch, status_ok = -1, False
    checks.append({"check": "Consistency between status string and status_clean", "passed": bool(status_ok), "detail": "Consistent" if status_ok else f"{mismatch} mismatched rows (or required columns missing)"})

    if "source" in cols:
        sources = sorted(df["source"].dropna().astype(str).str.strip().replace("", np.nan).dropna().unique().tolist())
        source_ok = len(sources) >= 2
    else:
        sources, source_ok = [], False
    checks.append({"check": "Multi-source integration (>= 2 distinct data sources)", "passed": bool(source_ok), "detail": f"{len(sources)} sources recorded: {sources}"})

    metrics = {
        "total_rows": n,
        "total_cols": len(df.columns),
        "on_time_count": int((df["status_clean"] == "on_time").sum()) if "status_clean" in cols else 0,
        "delayed_count": int((df["status_clean"] == "delayed").sum()) if "status_clean" in cols else 0,
        "cancelled_count": int((df["status_clean"] == "cancelled").sum()) if "status_clean" in cols else 0,
        "sources": sources,
    }
    return all(bool(c["passed"]) for c in checks), checks, metrics


def run_cli_check(target_path=None) -> int:
    """CLI runner for data validation."""
    path = target_path or FILE_CLEANED
    print("=" * 70)
    print("          VIETNAM FLIGHT DATA CONTRACT & QUALITY CHECK")
    print("=" * 70)
    print(f"Target file: {path}")
    all_passed, checks, metrics = validate_flights_data(path)
    print("-" * 70)
    for i, check in enumerate(checks, 1):
        print(f"{i:2d}. {'[PASS]' if check['passed'] else '[FAIL]'} {check['check']}\n    --> {check['detail']}")
    if metrics:
        total = metrics["total_rows"]
        print("-" * 70)
        print(f"Dataset Summary: {total:,} rows, {metrics['total_cols']} columns")
        for label in ("on_time", "delayed", "cancelled"):
            count = metrics[f"{label}_count"]
            print(f"  * {label:10}: {count:,} ({count/total*100:.1f}%)")
        print(f"  * Sources ({len(metrics['sources'])}): {', '.join(metrics['sources'])}")
    print("=" * 70)
    if all_passed:
        print(">>> ALL DATA QUALITY CHECKS PASSED SUCCESSFULLY. DATA CONTRACT IS VALID.")
        return 0
    print(">>> CRITICAL: SOME CHECKS FAILED. PLEASE FIX DATA ANOMALIES BEFORE TRAINING.")
    return 1


if __name__ == "__main__":
    sys.exit(run_cli_check(sys.argv[1] if len(sys.argv) > 1 else None))
