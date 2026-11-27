# Model Card: flight_delay_regressor_mae (v2.0.0)

- **Generated:** 2026-11-27 14:07:30
- **Algorithm:** `HistGradientBoosting_MAE`
- **Task:** `delay_regression_mae`
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
- **Accuracy:** `N/A` | **Balanced Acc:** `N/A` | **Macro-F1:** `N/A`
- **ROC-AUC:** `N/A` | **Log Loss:** `N/A` | **Brier (Disrupted):** `N/A`
- **Delayed Recall:** `N/A` | **Cancelled Recall:** `N/A`
- **Delay Regression MAE:** `14.68 min` | **MedAE:** `6.11 min` | **RMSE:** `30.96 min`

## Subgroup Airports Evaluated
5 airports audited in `evaluation.json`.
