# -*- coding: utf-8 -*-
"""[STEP 4] Isolation Forest Operational & Weather Anomaly Detection."""
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from flight_delay.common.constants import EXPERIMENTS_CFG, RANDOM_STATE, ROOT_DIR

ANOMALY_FEATURES = [
    "airport_load",
    "airport_congestion",
    "precipitation",
    "wind_gusts_10m",
    "visibility",
    "weather_hazard_score",
    "hour_sin",
    "hour_cos",
]


def run_anomaly_detection(gold_df: pd.DataFrame) -> dict:
    cfg = EXPERIMENTS_CFG["unsupervised"]["isolation_forest"]
    contam = float(cfg["contamination"])
    n_est = int(cfg["n_estimators"])

    avail = [c for c in ANOMALY_FEATURES if c in gold_df.columns]
    X_raw = gold_df[avail].fillna(0.0).to_numpy(dtype=float)
    X_std = StandardScaler().fit_transform(X_raw)

    iso = IsolationForest(n_estimators=n_est, contamination=contam, random_state=RANDOM_STATE, n_jobs=-1)
    pred = iso.fit_predict(X_std)  # -1 = anomaly, 1 = normal
    scores = -iso.decision_function(X_std)  # Higher = more anomalous

    compat_models = ROOT_DIR / "models"
    compat_models.mkdir(parents=True, exist_ok=True)
    joblib.dump(iso, compat_models / "isolation_forest.joblib")

    work = gold_df.copy().reset_index(drop=True)
    work["_is_anomaly"] = (pred == -1)
    work["_anomaly_score"] = scores

    anom = work[work["_is_anomaly"]]
    norm = work[~work["_is_anomaly"]]

    def _slice_rates(col: str) -> list:
        if col not in work.columns:
            return []
        rows = []
        for k, g in work.groupby(col):
            a_rate = float(g["_is_anomaly"].mean() * 100.0)
            rows.append({
                col: str(k),
                "total_flights": int(len(g)),
                "anomaly_count": int(g["_is_anomaly"].sum()),
                "anomaly_rate_pct": round(a_rate, 2),
                "disruption_rate_in_anomalies_pct": round(
                    float(g.loc[g["_is_anomaly"], "status_clean"].isin(["delayed", "cancelled"]).mean() * 100.0), 1
                ) if g["_is_anomaly"].any() else None,
            })
        return sorted(rows, key=lambda r: r["anomaly_rate_pct"], reverse=True)

    cols = [c for c in ("leg_key", "airline_code", "airport", "scheduled_time", "airport_load", "precipitation", "wind_gusts_10m", "weather_hazard_score", "status_clean") if c in work.columns]
    top_anoms = anom.sort_values("_anomaly_score", ascending=False).head(12)[cols + ["_anomaly_score"]].copy()
    top_anoms["_anomaly_score"] = top_anoms["_anomaly_score"].round(4)

    return {
        "features_used": avail,
        "contamination": contam,
        "total_flights": int(len(work)),
        "anomaly_count": int(len(anom)),
        "anomaly_rate_pct": round(float(len(anom) / len(work) * 100.0), 2),
        "post_hoc_audit_note": "Anomaly detection itself uses strictly pre-flight features; outcome delay/cancel breakdown below is post-hoc evaluation.",
        "anomaly_disruption_rate_pct": round(float(anom["status_clean"].isin(["delayed", "cancelled"]).mean() * 100.0), 2),
        "normal_disruption_rate_pct": round(float(norm["status_clean"].isin(["delayed", "cancelled"]).mean() * 100.0), 2),
        "anomaly_cancel_rate_pct": round(float((anom["status_clean"] == "cancelled").mean() * 100.0), 2),
        "normal_cancel_rate_pct": round(float((norm["status_clean"] == "cancelled").mean() * 100.0), 2),
        "by_airport": _slice_rates("airport"),
        "by_airline": _slice_rates("airline_code"),
        "top_anomalies": top_anoms.fillna("").to_dict("records"),
    }
