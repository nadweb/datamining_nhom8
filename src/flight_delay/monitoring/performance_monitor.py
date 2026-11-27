# -*- coding: utf-8 -*-
"""[STEP 7] Model Performance & Latency SLO Monitor."""
from flight_delay.common.constants import REPORTS_DIR
from flight_delay.common.utils import read_json


def run_performance_monitor() -> dict:
    eval_path = REPORTS_DIR / "experiments" / "evaluation_report.json"
    if not eval_path.exists():
        return {"status": "PENDING", "reason": "No evaluation_report.json generated yet."}

    rep = read_json(eval_path)
    cls_m = rep.get("classification_3class", {}).get("test_metrics", {})
    reg_m = rep.get("regression_mae", {}).get("test_metrics", {})

    f1_macro = float(cls_m.get("f1_macro", 0.0))
    bal_acc = float(cls_m.get("balanced_accuracy", 0.0))
    mae_min = float(reg_m.get("MAE_minutes", 999.0))
    infer_ms = float(cls_m.get("infer_ms", 0.0))

    status = "HEALTHY"
    if bal_acc < 0.40 or f1_macro < 0.35 or mae_min > 40.0:
        status = "DEGRADED"

    return {
        "status": status,
        "best_classifier": rep.get("classification_3class", {}).get("best_model"),
        "test_f1_macro": f1_macro,
        "test_balanced_accuracy": bal_acc,
        "delayed_recall": cls_m.get("delayed_recall"),
        "cancelled_recall": cls_m.get("cancelled_recall"),
        "best_regressor": rep.get("regression_mae", {}).get("best_model"),
        "test_mae_minutes": mae_min,
        "batch_test_infer_ms": infer_ms,
        "slo_batch_infer_lt_2000ms": bool(infer_ms < 2000.0),
    }
