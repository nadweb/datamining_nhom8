# -*- coding: utf-8 -*-
"""[STEP 3] Airport Load & Congestion Features."""
from typing import Dict, Optional

import numpy as np
import pandas as pd


def compute_airport_features(
    df: pd.DataFrame,
    airport_baseline_loads: Optional[Dict[str, float]] = None,
    default_median_load: float = 20.0,
) -> pd.DataFrame:
    df = df.copy()
    raw_load = pd.to_numeric(df["airport_load"], errors="coerce") if "airport_load" in df.columns else pd.Series(np.nan, index=df.index)
    fill_val = float(raw_load.median()) if raw_load.notna().any() else float(default_median_load)
    load = raw_load.fillna(fill_val)
    df["airport_load"] = load
    df["is_heavy_traffic"] = (load > 28).astype(int)

    if airport_baseline_loads:
        baselines = df["airport"].astype(str).str.upper().map(airport_baseline_loads).fillna(fill_val).clip(lower=1.0)
    else:
        baselines = pd.Series(max(fill_val, 1.0), index=df.index)
    df["airport_congestion"] = np.round(load / baselines, 4)
    return df
