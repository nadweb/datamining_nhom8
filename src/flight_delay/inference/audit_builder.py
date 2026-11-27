# -*- coding: utf-8 -*-
"""[STEP 7] Authentic Holdout Ground-Truth Audit Sample Builder using `predict_risk()`."""
from pathlib import Path
from typing import Optional

import pandas as pd

from flight_delay.common.constants import FILE_GOLD, PROCESSED_DIR, RANDOM_STATE
from flight_delay.common.utils import write_json
from flight_delay.inference.model_loader import load_production_artifacts
from flight_delay.inference.predict import predict_risk


def build_holdout_audit_benchmark(max_records: int = 80, output_path: Optional[Path] = None) -> list:
    """Evaluate a stratified chronological sample of Holdout Test flights via production `predict_risk()`."""
    gold = pd.read_csv(FILE_GOLD, low_memory=False)
    test_df = gold[gold.get("split_role", "test") == "test"].copy()
    if test_df.empty:
        test_df = gold.tail(2000).copy()

    bundle = load_production_artifacts(force_reload=True)

    # Stratified sample across on_time / delayed / cancelled from Holdout Test
    samples = []
    for status, quota in [("on_time", 42), ("delayed", 30), ("cancelled", 8)]:
        sub = test_df[test_df["status_clean"] == status]
        if len(sub) > 0:
            n_take = min(len(sub), quota)
            samples.append(sub.sample(n=n_take, random_state=RANDOM_STATE))
    sampled = pd.concat(samples, ignore_index=True)
    if len(sampled) < max_records:
        rem = test_df.drop(index=sampled.index, errors="ignore")
        if not rem.empty:
            sampled = pd.concat([sampled, rem.head(max_records - len(sampled))], ignore_index=True)

    audit_records = []
    for idx, row in sampled.head(max_records).iterrows():
        payload = {
            "airline_code": str(row["airline_code"]),
            "airport": str(row["airport"]),
            "direction": str(row.get("direction", "DEPARTURES")),
            "route_type": str(row.get("route_type", "domestic")),
            "scheduled_time": str(row["scheduled_time"])[:5],
            "flight_date": str(row["flight_date"]),
            "origin": str(row.get("origin", row["airport"])),
            "destination": str(row.get("destination", "HAN")),
            "precipitation": float(row.get("precipitation", 0.0) or 0.0),
            "wind_gusts_10m": float(row.get("wind_gusts_10m", 10.0) or 10.0),
            "visibility": float(row.get("visibility", 10000.0) or 10000.0),
            "weather_code": int(float(row.get("weather_code", 0) or 0)),
            "airport_load": int(float(row.get("airport_load", 20) or 20)),
        }
        pred = predict_risk(payload, prediction_time=str(row.get("prediction_time", "")) or None, bundle=bundle)
        actual_label = str(row["status_clean"])
        raw_delay = pd.to_numeric(row.get("delay_min"), errors="coerce")
        actual_delay = None if (actual_label == "cancelled" or pd.isna(raw_delay)) else round(float(raw_delay), 1)
        is_correct = pred["predicted_class"] == actual_label
        err_min = round(abs(pred["predicted_delay_minutes"] - actual_delay), 1) if actual_delay is not None else None

        note = (
            f"Dự báo khớp trạng thái {actual_label.upper()} trước T-{pred['horizon_hours']}h."
            if is_correct
            else f"Lệch nhãn ({pred['predicted_class']} vs {actual_label}); P(disrupted)={pred['probabilities_pct']['disrupted']}%."
        )
        audit_records.append({
            "id": f"HOLD_{idx + 1:03d}_{row.get('flight_no', 'VN')}",
            "flight_no": str(row.get("flight_no", "")),
            "flight_date": str(row.get("flight_date", "")),
            "airline_code": str(row["airline_code"]),
            "airport": str(row["airport"]),
            "origin": str(row.get("origin", "")),
            "destination": str(row.get("destination", "")),
            "direction": str(row.get("direction", "DEPARTURES")),
            "route_type": str(row.get("route_type", "domestic")),
            "scheduled_time": str(row["scheduled_time"])[:5],
            "actual_time": str(row.get("actual_time") or "--"),
            "weather": {
                "precipitation": payload["precipitation"],
                "wind": payload["wind_gusts_10m"],
                "visibility": payload["visibility"],
                "load": payload["airport_load"],
            },
            "prediction": {
                "predicted_label": pred["predicted_class"],
                "predicted_delay_min": pred["predicted_delay_minutes"],
                "probabilities": pred["probabilities"],
                "risk_tier": pred["risk_tier"],
            },
            "ground_truth": {
                "actual_label": actual_label,
                "actual_time": str(row.get("actual_time") or "--"),
                "actual_delay_min": actual_delay,
            },
            "audit": {
                "is_correct": bool(is_correct),
                "delay_error_min": err_min,
                "note": note,
            },
        })

    target = Path(output_path or (PROCESSED_DIR / "audit_benchmark_flights.json"))
    write_json(target, audit_records)
    return audit_records
