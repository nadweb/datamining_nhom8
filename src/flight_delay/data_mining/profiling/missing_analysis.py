# -*- coding: utf-8 -*-
"""[STEP 3] Missing Value Analysis for Bronze & Silver Layers."""
import numpy as np
import pandas as pd


def analyze_missingness(df: pd.DataFrame) -> dict:
    """Compute column-wise null/empty statistics."""
    cleaned = df.replace("", np.nan)
    counts = cleaned.isna().sum()
    pcts = (cleaned.isna().mean() * 100).round(2)
    summary = [
        {"column": col, "missing_count": int(counts[col]), "missing_pct": float(pcts[col])}
        for col in df.columns
    ]
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "columns_with_missing": [r for r in summary if r["missing_count"] > 0],
        "all_columns": summary,
    }
