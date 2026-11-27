# -*- coding: utf-8 -*-
"""[STEP 3] Airline Carrier Category & Route Importance Features."""
import pandas as pd

from flight_delay.common.constants import CARRIER_TYPE, GOLDEN_TRIANGLE


def compute_airline_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["carrier_category"] = df["airline_code"].astype(str).str.upper().map(CARRIER_TYPE).fillna("OTHER")
    orig = df["origin"].astype(str).str.upper() if "origin" in df.columns else pd.Series("", index=df.index)
    dest = df["destination"].astype(str).str.upper() if "destination" in df.columns else pd.Series("", index=df.index)
    df["is_golden_triangle"] = [1 if (o, d) in GOLDEN_TRIANGLE else 0 for o, d in zip(orig, dest)]
    df["is_domestic_flag"] = (df["route_type"].astype(str).str.lower() == "domestic").astype(int)
    df["is_departure"] = (df["direction"].astype(str).str.upper() == "DEPARTURES").astype(int)
    return df
