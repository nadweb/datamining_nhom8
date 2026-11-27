# -*- coding: utf-8 -*-
"""Standalone Step 3 Feature & Dataset Builder + EDA Report Script."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flight_delay.common.logging import configure_logging
from flight_delay.data_mining.datasets.dataset_builder import build_analytical_and_modeling_datasets
from flight_delay.data_mining.eda.report_builder import generate_full_eda_report

if __name__ == "__main__":
    configure_logging()
    ds = build_analytical_and_modeling_datasets()
    eda = generate_full_eda_report()
    print(f"Dataset v{ds['dataset_version']} & EDA ready (rows={eda['eda']['total_flights']})")
