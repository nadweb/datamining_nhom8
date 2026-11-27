# -*- coding: utf-8 -*-
"""Standalone Step 5-7 Evaluation, Audit & Monitoring Script."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flight_delay.common.logging import configure_logging
from flight_delay.inference.audit_builder import build_holdout_audit_benchmark
from flight_delay.monitoring.alerts import run_full_monitoring_suite

if __name__ == "__main__":
    configure_logging()
    mon = run_full_monitoring_suite()
    audit = build_holdout_audit_benchmark(max_records=80)
    print(f"Monitoring={mon['overall_status']} | Audit={len(audit)} rows")
