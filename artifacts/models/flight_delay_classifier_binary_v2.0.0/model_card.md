# Model Card: flight_delay_classifier_binary (v2.0.0)

- **Generated:** 2026-11-27 14:08:17
- **Algorithm:** `CatBoost`
- **Task:** `binary_disrupted`
- **Dataset Version:** `2026.11.02-h3-4716bcd4`
- **Feature Schema Version:** `2.0` (`66` encoded X features)
- **Prediction Horizon:** `T - 3 hours` prior to scheduled departure
- **Training Cutoff Date:** `2026-10-31`
- **Approval Status:** `Production` (`approved=True`)

## Intended Use & Limitations
- Pre-flight disruption triage and delay duration estimation across Vietnam aviation routes (`SGN`, `HAN`, `DAD`, `CXR`, `PQC`).
- Causal historical rates strictly use completed outcomes observed prior to `prediction_time`.
- **Known Limitations:** Public timetables omit tail-number turnaround state and crew rosters; rare `cancelled` class (`~0.85%`) limits standalone recall on short holdout windows; `18` regional spokes represent `<0.5%` traffic.

## Headline Holdout Evaluation Summary
- **Accuracy:** `0.7605` | **Balanced Acc:** `0.6214` | **Macro-F1:** `0.6227`
- **ROC-AUC:** `0.7103` | **Log Loss:** `0.4939` | **Brier (Disrupted):** `0.163`
- **Delayed Recall:** `N/A` | **Cancelled Recall:** `N/A`
- **Delay Regression MAE:** `N/A min` | **MedAE:** `N/A min` | **RMSE:** `N/A min`

## Subgroup Airports Evaluated
5 airports audited in `evaluation.json`.
