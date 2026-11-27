# -*- coding: utf-8 -*-
"""[REGRESSION SUITE] CLI Subcommands, Bronze Preservation & Artifact Integrity."""
import subprocess
import sys
from pathlib import Path

from flight_delay.common.constants import DEBUG_DIR, RAW_AVIATIONSTACK_DIR, RAW_FIDS_DIR, RAW_WEATHER_DIR


def test_raw_bronze_lake_and_debug_archive_untouched():
    assert len(list(RAW_FIDS_DIR.glob("*.csv"))) == 10
    assert len(list(RAW_AVIATIONSTACK_DIR.glob("*.csv"))) == 1
    assert (RAW_WEATHER_DIR / "weather_hourly.csv").exists()
    assert DEBUG_DIR.exists()
    assert sum(1 for _ in DEBUG_DIR.rglob("*") if _.is_file()) >= 2900


def test_cli_help_and_evaluate_commands():
    proc = subprocess.run(
        [sys.executable, str(Path("main.py")), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "FlightRisk VN" in proc.stdout
