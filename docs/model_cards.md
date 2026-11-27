# Model Cards & Governance Approval Registry

Managed via `src/flight_delay/governance/` and persisted in `artifacts/manifests/model_registry.json`. All production bundles enforce `approved == True` and verified SHA-256 `checksums.sha256`.

---

## 1. Production Classifier — `flight_delay_classifier_3class:v2.0.0`

- **Algorithm:** `CatBoostClassifier` (`iterations=300, lr=0.06, depth=6, auto_class_weights="Balanced"`)
- **Task:** 3-Class ICAO Pre-Flight Outcome (`0: on_time`, `1: delayed`, `2: cancelled`) at $T-3\text{h}$
- **Dataset Version:** `2026.11.02-h3-4716bcd4` (`9,503` Train / `1,353` Val / `2,714` Test, 66 clean pre-flight features)
- **Registry Status:** `Production` (`approved: true`, gated by automated quality & leakage audit)
- **Holdout Performance:** Accuracy `0.7498`, Balanced Accuracy `0.4186`, Macro-F1 `0.4154`, ROC-AUC OvR `0.8082`, PR-AUC Macro `0.5402`, `delayed_recall = 0.4236`, `cancelled_recall = 0.0000`, LogLoss `0.5170`.
- **Known Limitations:** Extreme sparsity of cancellations (`2` out of `2,714` holdout flights = `0.07%`) yields zero recall at argmax threshold; use `probabilities.cancelled` as a continuous alert signal rather than hard argmax alone. Validated only for the 5 primary hubs (`SGN, HAN, DAD, CXR, PQC`).

---

## 2. Production Regressor — `flight_delay_regressor_mae:v2.0.0`

- **Algorithm:** `HistGradientBoostingRegressor` (`loss="absolute_error", max_iter=180, lr=0.06, max_leaf_nodes=31`)
- **Task:** Continuous `delay_min` magnitude estimation (minutes) on **completed flights only**
- **Dataset Policy:** Cancelled, unknown, and diverted flights excluded (`delay_min = NaN`)
- **Registry Status:** `Production` (`approved: true`)
- **Holdout Performance:** MAE `14.68 min`, MedAE `6.11 min`, RMSE `30.96 min`, $R^2 = 0.0334$.
- **Known Limitations:** Tail delays exceeding 120 minutes driven by unmeasured air-traffic flow control or inbound aircraft rotation cannot be fully resolved from schedule + public weather alone.

---

## 3. Binary Ablation Classifier — `flight_delay_classifier_binary:v2.0.0`

- **Algorithm:** `HistGradientBoostingClassifier` *(selected by Val Macro-F1)* / `CatBoost` runner-up
- **Task:** Binary Ablation (`0: on_time` vs `1: disrupted = delayed | cancelled`)
- **Registry Status:** `Production` (ablation comparison bundle under `artifacts/models/flight_delay_classifier_binary_v2.0.0`)
- **Holdout Performance:** `HistGradientBoosting` Test BalAcc `0.6489`, Macro-F1 `0.6154`, ROC-AUC `0.7132`; `CatBoost` achieves `0.6617` BalAcc / `0.6234` Macro-F1 / `0.7304` ROC-AUC.
