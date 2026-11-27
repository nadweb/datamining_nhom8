# -*- coding: utf-8 -*-
"""[STEP 3] Unified Dataset Profiler for Silver and Gold Layers."""
import pandas as pd

from flight_delay.data_mining.profiling.missing_analysis import analyze_missingness
from flight_delay.data_mining.profiling.outlier_analysis import analyze_outliers
from flight_delay.data_mining.profiling.schema_analysis import analyze_raw_mixed_schema


def profile_dataset(df: pd.DataFrame, std_rows: int = 0) -> dict:
    return {
        "raw_provenance": analyze_raw_mixed_schema(std_rows) if std_rows else {},
        "missingness": analyze_missingness(df),
        "outliers": analyze_outliers(df),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
    }
