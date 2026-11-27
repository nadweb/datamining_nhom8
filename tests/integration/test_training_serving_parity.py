# -*- coding: utf-8 -*-
"""[INTEGRATION] Training-Serving Feature Parity & Fail-Closed Artifact Security."""
import numpy as np
import pandas as pd
import pytest

from flight_delay.common.constants import FILE_GOLD, PROCESSED_DIR
from flight_delay.common.exceptions import ArtifactError
from flight_delay.common.utils import read_json
from flight_delay.data_mining.encoding.preprocessing import FlightPreprocessor
from flight_delay.data_mining.feature_engineering.builder import FeatureBuilder
from flight_delay.data_mining.feature_engineering.historical import CausalHistoryStore
from flight_delay.inference.model_loader import clear_model_cache, load_production_artifacts
from flight_delay.inference.predict import predict_risk


def test_training_serving_parity_exact_match():
    gold = pd.read_csv(FILE_GOLD, low_memory=False)
    te_gold = gold[gold["split_role"] == "test"].reset_index(drop=True)
    te_mat = pd.read_csv(PROCESSED_DIR / "test.csv")

    bundle = load_production_artifacts()
    store = CausalHistoryStore.from_dict(read_json(PROCESSED_DIR / "causal_history_store.json"))
    prep = FlightPreprocessor.load(PROCESSED_DIR)
    fb = FeatureBuilder(horizon_hours=3, history_store=store)

    for idx in (0, 50, len(te_gold) - 1):
        row = te_gold.iloc[idx]
        single_gold = fb.build_single(
            airline_code=str(row["airline_code"]),
            airport=str(row["airport"]),
            direction=str(row["direction"]),
            route_type=str(row["route_type"]),
            scheduled_time=str(row["scheduled_time"])[:5],
            flight_date=str(row["flight_date"]),
            precipitation=float(row["precipitation"]),
            wind_gusts_10m=float(row["wind_gusts_10m"]),
            visibility=float(row["visibility"]),
            weather_code=int(row["weather_code"]),
            airport_load=int(round(float(row["airport_load"]))),
            origin=str(row["origin"]),
            destination=str(row["destination"]),
            prediction_time=str(row["prediction_time"]),
        )
        single_vec = prep.transform(single_gold)[0]
        batch_vec = te_mat.iloc[idx][prep.feature_names].to_numpy(dtype=float)
        assert np.allclose(single_vec, batch_vec, atol=1e-6), f"Parity mismatch at test index {idx}"

        pred = predict_risk(
            {
                "airline_code": str(row["airline_code"]),
                "airport": str(row["airport"]),
                "direction": str(row["direction"]),
                "route_type": str(row["route_type"]),
                "scheduled_time": str(row["scheduled_time"])[:5],
                "flight_date": str(row["flight_date"]),
                "origin": str(row["origin"]),
                "destination": str(row["destination"]),
                "precipitation": float(row["precipitation"]),
                "wind_gusts_10m": float(row["wind_gusts_10m"]),
                "visibility": float(row["visibility"]),
                "weather_code": int(row["weather_code"]),
                "airport_load": int(round(float(row["airport_load"]))),
            },
            prediction_time=str(row["prediction_time"]),
            bundle=bundle,
        )
        assert set(pred["probabilities"].keys()) == {"on_time", "delayed", "cancelled", "disrupted"}
        s_3 = pred["probabilities"]["on_time"] + pred["probabilities"]["delayed"] + pred["probabilities"]["cancelled"]
        assert abs(s_3 - 1.0) < 0.01


def test_corrupted_checksum_fails_closed(tmp_path, monkeypatch):
    bundle = load_production_artifacts(force_reload=True)
    csum_file = bundle.bundle_dir / "checksums.sha256"
    orig_text = csum_file.read_text(encoding="utf-8")
    try:
        csum_file.write_text("0" * 64 + "  metadata.json\n", encoding="utf-8")
        clear_model_cache()
        with pytest.raises(ArtifactError):
            load_production_artifacts(force_reload=True)
    finally:
        csum_file.write_text(orig_text, encoding="utf-8")
        clear_model_cache()
