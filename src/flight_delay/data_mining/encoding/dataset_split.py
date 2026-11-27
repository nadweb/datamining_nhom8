# -*- coding: utf-8 -*-
"""[STEP 3] Chronological Train/Validation/Holdout Splitting (Zero Lookahead Bias)."""
from typing import List, Tuple

import pandas as pd


def split_chronological_three_way(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    """Chronologically sort by (flight_date, scheduled_time, leg_key) into Train, Val, Holdout Test.

    Ensures zero timestamp mixing across partition boundaries when overlapping exact timestamps occur.
    """
    if abs(train_ratio + val_ratio - 0.80) > 1e-6 and train_ratio + val_ratio >= 0.95:
        raise ValueError("Invalid split ratios")
    sort_cols = [c for c in ("flight_date", "scheduled_time", "leg_key") if c in df.columns]
    df_sorted = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    # Group atomic timestamps (flight_date + scheduled_time) so simultaneous flights stay in the same split
    df_sorted["_ts_key"] = df_sorted["flight_date"].astype(str) + " " + df_sorted["scheduled_time"].astype(str)
    n = len(df_sorted)
    tr_target = int(n * train_ratio)
    va_target = int(n * (train_ratio + val_ratio))

    # Move cut boundaries to end of current timestamp block
    while tr_target < n - 2 and df_sorted.iloc[tr_target]["_ts_key"] == df_sorted.iloc[tr_target - 1]["_ts_key"]:
        tr_target += 1
    while va_target < n - 1 and df_sorted.iloc[va_target]["_ts_key"] == df_sorted.iloc[va_target - 1]["_ts_key"]:
        va_target += 1

    train_df = df_sorted.iloc[:tr_target].drop(columns=["_ts_key"]).copy()
    val_df = df_sorted.iloc[tr_target:va_target].drop(columns=["_ts_key"]).copy()
    test_df = df_sorted.iloc[va_target:].drop(columns=["_ts_key"]).copy()

    val_cutoff = str(train_df["flight_date"].max())
    holdout_cutoff = str(val_df["flight_date"].max()) if not val_df.empty else val_cutoff
    return train_df, val_df, test_df, val_cutoff, holdout_cutoff


def expanding_temporal_folds(df_train_val: pd.DataFrame, n_folds: int = 3) -> List[Tuple[pd.Index, pd.Index]]:
    """Build strictly chronological expanding validation folds on Train+Val."""
    n = len(df_train_val)
    step = max(n // (n_folds + 1), 1)
    folds = []
    for k in range(1, n_folds + 1):
        tr_end = step * k
        va_end = min(step * (k + 1), n) if k < n_folds else n
        if tr_end > 0 and va_end > tr_end:
            folds.append((df_train_val.index[:tr_end], df_train_val.index[tr_end:va_end]))
    return folds
