# -*- coding: utf-8 -*-
"""[STEP 6] Markdown Model Card Generator."""
from datetime import datetime

from flight_delay.common.constants import ROOT_DIR


def render_model_card(metadata: dict, evaluation: dict, feature_count: int) -> str:
    clf_sel = evaluation.get("classification", {})
    reg_sel = evaluation.get("regression", {})
    subgroups = evaluation.get("subgroups", {})
    card = f"""# Model Card: {metadata.get('model_name', 'flight_risk_classifier')} ({metadata.get('version', '2.0.0')})

- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Algorithm:** `{metadata.get('algorithm')}`
- **Task:** `{metadata.get('task')}`
- **Dataset Version:** `{metadata.get('dataset_version')}`
- **Feature Schema Version:** `{metadata.get('feature_schema_version')}` (`{feature_count}` encoded X features)
- **Prediction Horizon:** `T - {metadata.get('prediction_horizon_hours', 3)} hours` prior to scheduled departure
- **Training Cutoff Date:** `{metadata.get('training_cutoff')}`
- **Approval Status:** `{metadata.get('status', 'Candidate')}` (`approved={metadata.get('approved', False)}`)

## Intended Use & Limitations
- Pre-flight disruption triage and delay duration estimation across Vietnam aviation routes (`SGN`, `HAN`, `DAD`, `CXR`, `PQC`).
- Causal historical rates strictly use completed outcomes observed prior to `prediction_time`.
- **Known Limitations:** Public timetables omit tail-number turnaround state and crew rosters; rare `cancelled` class (`~0.85%`) limits standalone recall on short holdout windows; `18` regional spokes represent `<0.5%` traffic.

## Headline Holdout Evaluation Summary
- **Accuracy:** `{clf_sel.get('accuracy', 'N/A')}` | **Balanced Acc:** `{clf_sel.get('balanced_accuracy', 'N/A')}` | **Macro-F1:** `{clf_sel.get('f1_macro', 'N/A')}`
- **ROC-AUC:** `{clf_sel.get('roc_auc', 'N/A')}` | **Log Loss:** `{clf_sel.get('log_loss', 'N/A')}` | **Brier (Disrupted):** `{clf_sel.get('brier_disrupted', 'N/A')}`
- **Delayed Recall:** `{clf_sel.get('delayed_recall', 'N/A')}` | **Cancelled Recall:** `{clf_sel.get('cancelled_recall', 'N/A')}`
- **Delay Regression MAE:** `{reg_sel.get('MAE_minutes', 'N/A')} min` | **MedAE:** `{reg_sel.get('MedAE_minutes', 'N/A')} min` | **RMSE:** `{reg_sel.get('RMSE_minutes', 'N/A')} min`

## Subgroup Airports Evaluated
{len(subgroups.get('by_airport', {}))} airports audited in `evaluation.json`.
"""
    docs_card_dir = ROOT_DIR / "docs" / "model_cards"
    docs_card_dir.mkdir(parents=True, exist_ok=True)
    (docs_card_dir / f"{metadata.get('model_name', 'model')}_{metadata.get('version', 'v1')}.md").write_text(
        card, encoding="utf-8"
    )
    return card
