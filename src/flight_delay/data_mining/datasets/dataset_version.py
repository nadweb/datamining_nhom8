# -*- coding: utf-8 -*-
"""[STEP 3] Deterministic Dataset Semantic Versioning & Fingerprinting."""
import pandas as pd

from flight_delay.common.hashing import sha256_dict


def compute_dataset_version(df: pd.DataFrame, horizon_hours: int) -> str:
    """Return deterministic semantic version string tied to max flight_date and row digest."""
    max_date = str(df["flight_date"].max()).replace("-", ".") if "flight_date" in df.columns and not df.empty else "0000.00.00"
    sig = sha256_dict({
        "rows": int(len(df)),
        "cols": list(df.columns),
        "horizon_hours": int(horizon_hours),
        "first_leg": str(df["leg_key"].iloc[0]) if not df.empty and "leg_key" in df.columns else "",
        "last_leg": str(df["leg_key"].iloc[-1]) if not df.empty and "leg_key" in df.columns else "",
    })[:8]
    return f"{max_date}-h{horizon_hours}-{sig}"
