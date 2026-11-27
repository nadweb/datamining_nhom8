# CRISP-DM-Inspired Methodology & Anti-Leakage Engineering

> **Terminology Note:** Original CRISP-DM defines 6 phases; FlightRisk VN organizes work into a **7-stage engineering pipeline** (`01_ingestion` → `07_monitoring`) paired with **8 self-contained notebooks (`00`–`07`)** so Business Understanding, Unsupervised Discovery, and Deployment/Monitoring remain clearly separated and reproducible.

## 1. Problem Formulation

- **Problem A (Primary Classification):** Predict 3-class ICAO status (`0: on_time`, `1: delayed >= 15m`, `2: cancelled`) **$H = 3$ hours prior to scheduled departure (`prediction_time = scheduled_dt - 3h`)**.
  - **Binary Ablation:** `0: on_time` vs `1: disrupted (delayed | cancelled)` evaluated alongside 3-class to measure separability trade-offs under extreme cancellation rarity (`0.70%` overall, `0.07%` in Holdout Test).
  - **Horizon Ablation:** $H \in \{1\text{h}, 3\text{h}, 6\text{h}\}$ comparing causal historical lookback sensitivity.
- **Problem B (L1/MAE Regression):** Predict expected `delay_min` strictly on **completed flights** (`on_time` or `delayed`). Cancelled flights enforce `delay_min = NaN` and are excluded from regression (never coerced to `0`).
- **Problem C (Unsupervised Pattern Discovery):**
  - `K-Means` ($k=4$, validated $k=2..8$ + 5-seed ARI) + `PCA (3D)` to uncover operational delay regimes.
  - `Isolation Forest` (3% contamination) broken down by airport & airline for anomaly auditing.
  - `FP-Growth` with subset redundancy pruning + Holdout Test retention check to discover actionable risk rules.

---

## 2. Anti-Leakage & Integrity Policies

1. **Temporal Chronological Split (Zero Random Shuffle):**
   - Sorted strictly by `(flight_date, scheduled_time, leg_key)` with atomic timestamp grouping so simultaneous flights never split across boundaries.
   - Partition sizes: **Train 70% (`9,503`)**, **Validation 10% (`1,353`)**, **Holdout Test 20% (`2,714`)**.
2. **Validation-Only Model Selection:**
   - Expanding 3-fold temporal CV runs inside Train+Val; hyperparameter/algorithm winners are selected strictly by **Validation `f1_macro`** (classification) and **Validation `MAE_minutes`** (regression). Holdout Test is evaluated once.
3. **Strictly Point-in-Time Historical Rates (`CausalHistoryStore` + `leakage_guard.py`):**
   - *Naming Clarification:* `CausalHistoryStore` implements **point-in-time historical feature aggregation** (`bisect_left` over `outcome_known_at < prediction_time`, fit strictly on Train with $m=20$ Bayesian smoothing) so future outcomes cannot leak into earlier flights — it does **not** perform econometric causal inference.
   - Alongside `data/gold/causal_history_store.json` (~1.18 MB, suitable for JSON lookup at current scale), `data/gold/causal_history_metadata.json` records schema/row metadata and notes Parquet migration (`causal_history_store.parquet`) once scale reaches $\ge 100{,}000$ records.
   - Every dataset build runs `leakage_guard.assert_no_future_leakage()` blocking forbidden post-flight columns, non-monotonic splits, and $H < 3\text{h}$ violations; serving rejects `prediction_time > scheduled_dt` with `InvalidPredictionTimeError` (HTTP 422).
4. **Excluded Outcomes Policy:**
   - Flights with ambiguous statuses (`diverted = 11`, `unknown = 29`) are segregated into `data/silver/excluded_outcomes.csv` (`40` rows) rather than forced into valid labels.
5. **Single Feature Pipeline Parity & Drift Separation:**
   - Both batch training (`build_dataframe`) and single-flight serving (`build_single`) invoke `FeatureBuilder` and `FlightPreprocessor` identically.
   - Monitoring (`src/flight_delay/monitoring/`) separates **Distribution Drift** (`data_drift.py`: PSI & 2-sample KS-test on numerical inputs + $\chi^2$ on categories) from **Concept Drift** (`concept_drift.py`: requires ground-truth post-flight outcomes to detect shifts in the feature-target disruption rate).
