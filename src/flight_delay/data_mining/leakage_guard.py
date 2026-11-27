# -*- coding: utf-8 -*-
"""[STEP 3] Runtime Data Leakage Guard for Gold Features, Chronological Splits & Targets."""
from typing import Dict, Iterable, Sequence

import pandas as pd

from flight_delay.common.constants import FORBIDDEN_POST_FLIGHT_COLUMNS
from flight_delay.common.exceptions import LeakageError

ALLOWED_TARGET_COLUMNS = frozenset({"target", "target_binary", "delay_min"})


def assert_no_forbidden_features(feature_names: Iterable[str], context: str = "model_matrix") -> None:
    """Reject any post-flight or target columns inside model feature inputs X."""
    feats = set(feature_names)
    leaked = sorted(feats & (set(FORBIDDEN_POST_FLIGHT_COLUMNS) | ALLOWED_TARGET_COLUMNS))
    if leaked:
        raise LeakageError(f"[{context}] Forbidden post-flight or target columns detected in X: {leaked}")


def assert_causal_horizon(gold_df: pd.DataFrame, min_horizon_hours: int = 1) -> None:
    """Verify prediction_time precedes scheduled_dt by at least min_horizon_hours."""
    if gold_df.empty:
        return
    if "prediction_time" not in gold_df.columns or "scheduled_dt" not in gold_df.columns:
        raise LeakageError("[gold_df] Missing prediction_time or scheduled_dt required for causal horizon check.")
    pred = pd.to_datetime(gold_df["prediction_time"], errors="coerce")
    sched = pd.to_datetime(gold_df["scheduled_dt"], errors="coerce")
    valid = pred.notna() & sched.notna()
    lead_hours = (sched[valid] - pred[valid]).dt.total_seconds() / 3600.0
    if bool((lead_hours < (min_horizon_hours - 1e-6)).any()):
        raise LeakageError(
            f"[gold_df] Causal horizon violated: minimum lead time={float(lead_hours.min()):.2f}h < {min_horizon_hours}h"
        )


def assert_chronological_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    anchor_col: str = "flight_date",
) -> None:
    """Verify monotonic non-overlapping temporal progression Train <= Val <= Test."""
    pairs = [("train", train_df, "val", val_df), ("val", val_df, "test", test_df)]
    for left_name, left_df, right_name, right_df in pairs:
        if left_df.empty or right_df.empty or anchor_col not in left_df.columns or anchor_col not in right_df.columns:
            continue
        left_max = str(left_df[anchor_col].astype(str).max())
        right_min = str(right_df[anchor_col].astype(str).min())
        if left_max > right_min:
            raise LeakageError(
                f"[split_order] Temporal overlap between {left_name} (max={left_max}) and {right_name} (min={right_min})."
            )


def assert_target_separation(exported_columns: Sequence[str], feature_names: Iterable[str], split_name: str) -> None:
    """Verify exported split CSV separates feature matrix X strictly from supervised targets."""
    feat_set = set(feature_names)
    col_set = set(exported_columns)
    assert_no_forbidden_features(feat_set, context=split_name)
    extra = col_set - feat_set
    if extra != set(ALLOWED_TARGET_COLUMNS):
        raise LeakageError(
            f"[{split_name}] Expected exact supervised targets {sorted(ALLOWED_TARGET_COLUMNS)}, got {sorted(extra)}."
        )


def assert_no_future_leakage(
    gold_df: pd.DataFrame,
    feature_names: Sequence[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    exported_splits: Dict[str, pd.DataFrame],
    min_horizon_hours: int = 1,
) -> Dict:
    """Execute full runtime leakage suite during Gold dataset creation."""
    assert_no_forbidden_features(feature_names, context="preprocessor.feature_names")
    assert_causal_horizon(gold_df, min_horizon_hours=min_horizon_hours)
    assert_chronological_splits(train_df, val_df, test_df)
    for split_name, split_df in exported_splits.items():
        assert_target_separation(list(split_df.columns), feature_names, split_name=split_name)
    return {
        "passed": True,
        "checked_rows": int(len(gold_df)),
        "feature_count": int(len(feature_names)),
        "supervised_targets": sorted(ALLOWED_TARGET_COLUMNS),
        "min_horizon_hours": int(min_horizon_hours),
    }
