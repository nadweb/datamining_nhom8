# -*- coding: utf-8 -*-
"""Standalone Step 1 Ingestion Script."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flight_delay.common.logging import configure_logging
from flight_delay.ingestion.collector import collect_bronze_and_standardize

if __name__ == "__main__":
    configure_logging()
    df = collect_bronze_and_standardize(run_live_scrape=False)
    print(f"Bronze standardized snapshots: {len(df)}")
