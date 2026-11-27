# -*- coding: utf-8 -*-
"""[STEP 7] 4-Tier Monitoring Aggregator & Retrain Alert Generator."""
from datetime import datetime
from typing import Optional

import pandas as pd

from flight_delay.common.constants import FILE_GOLD, MONITORING_CFG, REPORTS_DIR
from flight_delay.common.utils import write_json
from flight_delay.monitoring.concept_drift import run_concept_drift_monitor
from flight_delay.monitoring.data_drift import run_data_drift_monitor
from flight_delay.monitoring.data_quality_monitor import run_data_quality_check
from flight_delay.monitoring.performance_monitor import run_performance_monitor


def run_full_monitoring_suite(gold_df: Optional[pd.DataFrame] = None) -> dict:
    if gold_df is None:
        gold_df = pd.read_csv(FILE_GOLD, low_memory=False)
    if "split_role" in gold_df.columns:
        ref = gold_df[gold_df["split_role"] != "test"]
        cur = gold_df[gold_df["split_role"] == "test"]
    else:
        cut = int(len(gold_df) * 0.8)
        ref, cur = gold_df.iloc[:cut], gold_df.iloc[cut:]

    dq = run_data_quality_check(cur)
    dd = run_data_drift_monitor(ref, cur)
    cd = run_concept_drift_monitor(gold_df)
    perf = run_performance_monitor()

    alerts = []
    if dq["status"] != "OK":
        alerts.append({"tier": "data_quality", "severity": "WARNING", "message": "Detected anomalies in incoming data completeness/schema."})
    if dd["overall_status"] == "CRITICAL":
        alerts.append({"tier": "data_drift", "severity": "CRITICAL", "message": f"Max feature PSI={dd['max_psi']} exceeds critical threshold."})
    elif dd["overall_status"] == "WARNING":
        alerts.append({"tier": "data_drift", "severity": "WARNING", "message": f"Moderate feature shift observed (max PSI={dd['max_psi']})."})
    if cd["concept_drift_detected"]:
        alerts.append({"tier": "concept_drift", "severity": "WARNING", "message": f"Disruption rate shifted by {cd['disruption_shift_pct_points']} pp."})
    if perf.get("status") == "DEGRADED":
        alerts.append({"tier": "model_performance", "severity": "CRITICAL", "message": "Model metrics fell below healthy threshold."})

    retrain_recommended = bool(
        dd["max_psi"] >= float(MONITORING_CFG["psi_thresholds"]["retrain_trigger"])
        or cd["concept_drift_detected"]
        or perf.get("status") == "DEGRADED"
    )
    overall = "CRITICAL" if any(a["severity"] == "CRITICAL" for a in alerts) else "WARNING" if alerts else "OK"

    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "overall_status": overall,
        "retrain_recommended": retrain_recommended,
        "alerts": alerts,
        "data_quality": dq,
        "data_drift": dd,
        "concept_drift": cd,
        "model_performance": perf,
        # Flattened UI compatibility keys
        "psi_thresholds": dd["psi_thresholds"],
        "feature_drift": dd["feature_drift"],
        "categorical_drift": dd["categorical_drift"],
        "label_drift": cd,
        "retrain_policy": MONITORING_CFG.get("retrain_policy", ""),
    }
    write_json(REPORTS_DIR / "monitoring" / "drift_report.json", payload)
    return payload
