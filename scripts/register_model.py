# -*- coding: utf-8 -*-
"""Standalone Step 6 Model Governance Registration Script."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flight_delay.common.logging import configure_logging
from flight_delay.governance.approval import validate_and_approve_model

if __name__ == "__main__":
    configure_logging()
    for name in ("flight_delay_classifier_3class", "flight_delay_regressor_mae"):
        entry = validate_and_approve_model(name, "v2.0.0", promote_to_production=True)
        print(f"Promoted {name}:{entry['version']} -> {entry['status']}")
