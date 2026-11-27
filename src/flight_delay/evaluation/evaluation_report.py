# -*- coding: utf-8 -*-
"""[STEP 5] Consolidated Evaluation Report & Applicability Limits Builder."""
from datetime import datetime

from flight_delay.common.constants import REPORTS_DIR
from flight_delay.common.utils import write_json


def save_evaluation_report(payload: dict) -> dict:
    report = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **payload}
    write_json(REPORTS_DIR / "experiments" / "evaluation_report.json", report)
    return report
