# -*- coding: utf-8 -*-
"""[STEP 6] Structured Local Experiment Tracker under experiments/."""
from datetime import datetime

from flight_delay.common.constants import EXPERIMENTS_DIR
from flight_delay.common.hashing import sha256_dict
from flight_delay.common.utils import write_json


def log_experiment_run(category: str, run_name: str, params: dict, metrics: dict, artifacts: dict = None) -> dict:
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sha256_dict({'name': run_name, 'params': params})[:6]}"
    payload = {
        "run_id": run_id,
        "category": category,
        "run_name": run_name,
        "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "params": params,
        "metrics": metrics,
        "artifacts": artifacts or {},
    }
    target = EXPERIMENTS_DIR / category / f"{run_name}_{run_id}.json"
    write_json(target, payload)
    latest = EXPERIMENTS_DIR / category / f"latest_{run_name}.json"
    write_json(latest, payload)
    return payload
