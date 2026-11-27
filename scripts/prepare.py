# -*- coding: utf-8 -*-
"""Standalone Step 2 Data Preparation Script."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flight_delay.common.logging import configure_logging
from flight_delay.data_engineering.processor import run_preparation_pipeline

if __name__ == "__main__":
    configure_logging()
    valid, excluded, quality = run_preparation_pipeline()
    print(f"Silver valid={len(valid)}, excluded={len(excluded)}")
