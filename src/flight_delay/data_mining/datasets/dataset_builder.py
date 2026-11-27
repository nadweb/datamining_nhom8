# -*- coding: utf-8 -*-
"""[STEP 3] Gold Analytical & ML Matrix Builder (Causal History + Temporal Split + Leakage Guard)."""
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from flight_delay.common.constants import (
    EXPERIMENTS_CFG,
    FILE_CLEANED,
    FILE_FEATURES,
    FILE_GOLD,
    FILE_SILVER,
    LABEL_MAP_3CLASS,
    PRIMARY_HORIZON_HOURS,
    PROCESSED_DIR,
)
from flight_delay.common.utils import write_json
from flight_delay.data_engineering.contracts import validate_gold_contract
from flight_delay.data_mining.datasets.dataset_manifest import build_dataset_manifest
from flight_delay.data_mining.datasets.dataset_version import compute_dataset_version
from flight_delay.data_mining.encoding.dataset_split import split_chronological_three_way
from flight_delay.data_mining.encoding.preprocessing import FlightPreprocessor
from flight_delay.data_mining.feature_engineering.builder import FeatureBuilder
from flight_delay.data_mining.feature_engineering.historical import CausalHistoryStore
from flight_delay.data_mining.feature_engineering.registry import FEATURE_REGISTRY

log = logging.getLogger(__name__)


def build_analytical_and_modeling_datasets(
    silver_df: Optional[pd.DataFrame] = None,
    horizon_hours: int = PRIMARY_HORIZON_HOURS,
    output_dir: Optional[Path] = None,
) -> Dict:
    """Build Gold analytical dataset (T-H prediction horizon) and leak-free Train/Val/Test matrices."""
    proc_dir = Path(output_dir or PROCESSED_DIR)
    proc_dir.mkdir(parents=True, exist_ok=True)

    if silver_df is None:
        src = FILE_SILVER if FILE_SILVER.exists() else FILE_CLEANED
        if not src.exists():
            raise FileNotFoundError(f"Missing Silver dataset at {src}. Run `python main.py prepare` first.")
        silver_df = pd.read_csv(src, low_memory=False)

    clean = silver_df[silver_df["status_clean"].isin(LABEL_MAP_3CLASS)].copy()
    train_ratio = float(EXPERIMENTS_CFG["split"].get("train_ratio", 0.70))
    val_ratio = float(EXPERIMENTS_CFG["split"].get("validation_ratio", EXPERIMENTS_CFG["split"].get("val_ratio", 0.10)))
    tr_raw, va_raw, te_raw, val_cutoff, holdout_cutoff = split_chronological_three_way(
        clean, train_ratio=train_ratio, val_ratio=val_ratio
    )

    # Fit causal historical store ONLY on Train observations (zero val/test future leakage!)
    train_history_store = CausalHistoryStore.from_observations(tr_raw)
    builder = FeatureBuilder(horizon_hours=horizon_hours, history_store=train_history_store)

    tr_gold = builder.build_dataframe(tr_raw, history_store=train_history_store)
    va_gold = builder.build_dataframe(va_raw, history_store=train_history_store)
    te_gold = builder.build_dataframe(te_raw, history_store=train_history_store)

    gold_all = pd.concat([tr_gold, va_gold, te_gold], ignore_index=True)
    ok_gold, g_errs = validate_gold_contract(gold_all, FEATURE_REGISTRY.model_input_columns)
    if not ok_gold:
        raise RuntimeError(f"Gold contract violated: {g_errs}")

    # Assign explicit split membership & dual targets (Problem A 3-class + Binary ablation + Problem B regression)
    gold_all["split_role"] = (["train"] * len(tr_gold)) + (["val"] * len(va_gold)) + (["test"] * len(te_gold))
    gold_all["target_3class"] = gold_all["status_clean"].map(LABEL_MAP_3CLASS).astype(int)
    gold_all["target_binary"] = gold_all["status_clean"].map(lambda s: 0 if s == "on_time" else 1).astype(int)

    ds_version = compute_dataset_version(clean, horizon_hours)
    gold_all["dataset_version"] = ds_version
    if proc_dir == PROCESSED_DIR:
        gold_all.to_csv(FILE_GOLD, index=False, encoding="utf-8-sig")
        gold_all.to_csv(FILE_FEATURES, index=False, encoding="utf-8-sig")

    # Save frozen causal history store for inference parity
    write_json(proc_dir / "causal_history_store.json", train_history_store.to_dict())

    # Fit FlightPreprocessor STRICTLY on Train
    preprocessor = FlightPreprocessor().fit(tr_gold)
    X_tr = preprocessor.to_dataframe(preprocessor.transform(tr_gold))
    X_va = preprocessor.to_dataframe(preprocessor.transform(va_gold))
    X_te = preprocessor.to_dataframe(preprocessor.transform(te_gold))

    def _export(X_mat: pd.DataFrame, raw_part: pd.DataFrame, filename: str) -> pd.DataFrame:
        out = X_mat.copy()
        out["target"] = raw_part["status_clean"].map(LABEL_MAP_3CLASS).astype(int).values
        out["target_binary"] = raw_part["status_clean"].map(lambda s: 0 if s == "on_time" else 1).astype(int).values
        out["delay_min"] = raw_part["delay_min"].values
        out.to_csv(proc_dir / filename, index=False)
        return out

    tr_exp = _export(X_tr, tr_gold, "train.csv")
    va_exp = _export(X_va, va_gold, "val.csv")
    te_exp = _export(X_te, te_gold, "test.csv")

    split_summary = {
        "train_rows": int(len(tr_exp)),
        "val_rows": int(len(va_exp)),
        "test_rows": int(len(te_exp)),
        "val_cutoff_date": val_cutoff,
        "holdout_cutoff_date": holdout_cutoff,
        "train_3class_counts": tr_gold["status_clean"].value_counts().to_dict(),
        "val_3class_counts": va_gold["status_clean"].value_counts().to_dict(),
        "test_3class_counts": te_gold["status_clean"].value_counts().to_dict(),
    }

    meta = preprocessor.save(
        proc_dir,
        extra_meta={
            "dataset_version": ds_version,
            "horizon_hours": horizon_hours,
            "cutoff_date": holdout_cutoff,
            "val_cutoff_date": val_cutoff,
            "train_rows": int(len(tr_exp)),
            "val_rows": int(len(va_exp)),
            "test_rows": int(len(te_exp)),
            "total_features": len(preprocessor.feature_names),
            "one_hot_columns": preprocessor.encoded_cat_names,
            "label_map": LABEL_MAP_3CLASS,
        },
    )

    manifest = build_dataset_manifest(
        dataset_version=ds_version,
        horizon_hours=horizon_hours,
        silver_rows=int(len(clean)),
        gold_cols_count=int(gold_all.shape[1]),
        preprocessor_meta=meta,
        split_summary=split_summary,
    )
    log.info(
        "Dataset v%s (horizon=%dh) built: Train=%d | Val=%d | Test=%d | Matrix X=%d cols",
        ds_version, horizon_hours, len(tr_exp), len(va_exp), len(te_exp), len(preprocessor.feature_names)
    )
    return {
        "dataset_version": ds_version,
        "manifest": manifest,
        "metadata": meta,
        "splits": split_summary,
    }
