# -*- coding: utf-8 -*-
"""[STEP 3] Operational Compound Interaction Terms (Load x Weather, Peak x Rain)."""
import numpy as np
import pandas as pd


def compute_operational_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["load_x_weather"] = np.round(df["airport_load"].fillna(20.0) * df["weather_hazard_score"].fillna(0.0), 4)
    df["peak_x_rain"] = (df["is_peak_hour"].fillna(0).astype(int) * df["is_rain"].fillna(0).astype(int))
    return df
