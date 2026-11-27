# -*- coding: utf-8 -*-
"""[STEP 7] Concept & Label Distribution Drift Monitor."""
from typing import Optional

import pandas as pd

from flight_delay.common.constants import FILE_GOLD, MONITORING_CFG


def run_concept_drift_monitor(gold_df: Optional[pd.DataFrame] = None) -> dict:
    if gold_df is None:
        gold_df = pd.read_csv(FILE_GOLD, low_memory=False)
    if "split_role" in gold_df.columns:
        ref = gold_df[gold_df["split_role"] != "test"]
        cur = gold_df[gold_df["split_role"] == "test"]
    else:
        cut = int(len(gold_df) * 0.8)
        ref, cur = gold_df.iloc[:cut], gold_df.iloc[cut:]

    def _rates(part: pd.DataFrame) -> dict:
        if len(part) == 0:
            return {"on_time_pct": 0.0, "delayed_pct": 0.0, "cancelled_pct": 0.0, "disrupted_pct": 0.0}
        on_t = float((part["status_clean"] == "on_time").mean() * 100.0)
        dly = float((part["status_clean"] == "delayed").mean() * 100.0)
        cnc = float((part["status_clean"] == "cancelled").mean() * 100.0)
        return {
            "on_time_pct": round(on_t, 2),
            "delayed_pct": round(dly, 2),
            "cancelled_pct": round(cnc, 2),
            "disrupted_pct": round(dly + cnc, 2),
        }

    ref_r = _rates(ref)
    cur_r = _rates(cur)
    diff_pp = round(abs(cur_r["disrupted_pct"] - ref_r["disrupted_pct"]), 2)
    thresh = float(MONITORING_CFG.get("label_drift_alert_pct_points", 5.0))
    return {
        "status": "ALERT" if diff_pp > thresh else "STABLE",
        "reference_rates_pct": ref_r,
        "current_rates_pct": cur_r,
        "disruption_shift_pct_points": diff_pp,
        "alert_threshold_pct_points": thresh,
        "concept_drift_detected": bool(diff_pp > thresh),
    }
