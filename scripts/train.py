# -*- coding: utf-8 -*-
"""Standalone Step 4-5 Modeling Suite Script."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flight_delay.common.logging import configure_logging
from flight_delay.modeling.runner import run_all_modeling_and_evaluation

if __name__ == "__main__":
    configure_logging()
    res = run_all_modeling_and_evaluation()
    print(f"Best classifier={res['classification']['multiclass_3']['best_model']}")
