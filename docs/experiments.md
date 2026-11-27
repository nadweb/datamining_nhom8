# Observed Experimental Benchmarks (`dataset_version: 2026.11.02-h3-4716bcd4`)

All metrics below were recorded verbatim from the final verified run against the deduplicated `13,570` Silver flights (`Train = 9,503`, `Val = 1,353`, `Test = 2,714`; matrix $X = 66$ pre-flight features).

---

## 1. Primary Task: 3-Class ICAO Classification (`T-3h`)

Holdout Test distribution: `on_time = 2,169 (79.92%)`, `delayed = 543 (20.01%)`, `cancelled = 2 (0.07%)`.

### Validation Selection Table (`n = 1,353`)
| Algorithm | Accuracy | Balanced Acc | Macro-F1 | Delayed Recall | Cancelled Recall | ROC-AUC OvR | PR-AUC Macro | LogLoss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MajorityClassifier | 0.7672 | 0.3333 | 0.2894 | 0.0000 | 0.0000 | 0.5000 | 0.3333 | 8.3915 |
| HeuristicBaseline | 0.7036 | 0.3673 | 0.3674 | 0.2607 | 0.0000 | 0.5425 | 0.3634 | 0.6280 |
| HistGradientBoosting | 0.7088 | 0.3835 | 0.3838 | 0.3201 | 0.0000 | 0.6502 | 0.4352 | 0.6351 |
| XGBoost | 0.7081 | 0.3972 | 0.3953 | 0.3795 | 0.0000 | 0.6601 | 0.4466 | 0.5972 |
| **CatBoost** *(Selected)* | 0.6970 | 0.4080 | **0.4006** | 0.4455 | 0.0000 | 0.6957 | 0.4270 | 0.6105 |

### Holdout Test Results (`n = 2,714`)
| Algorithm | Accuracy | BalAcc | Macro-F1 | Delayed Recall | Cancelled Recall | ROC-AUC OvR | PR-AUC Macro | LogLoss | CV F1 (mean±std) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MajorityClassifier | 0.7992 | 0.3333 | 0.2961 | 0.0000 | 0.0000 | 0.5000 | 0.3333 | 7.2379 | -- |
| HeuristicBaseline | 0.7369 | 0.3713 | 0.3733 | 0.2560 | 0.0000 | 0.7058 | 0.3666 | 0.5833 | -- |
| HistGradientBoosting | 0.7668 | 0.3994 | 0.4040 | 0.3186 | 0.0000 | 0.8012 | 0.7516 | 0.4898 | `0.4623 ± 0.1198` |
| XGBoost | 0.7568 | 0.4187 | 0.4174 | 0.4125 | 0.0000 | 0.8045 | 0.7557 | 0.5026 | `0.4853 ± 0.1266` |
| **CatBoost** *(Prod)* | **0.7498** | **0.4186** | **0.4154** | **0.4236** | **0.0000** | **0.8082** | **0.5402** | **0.5170** | `0.4330 ± 0.0561` |

### Statistical Significance Tests (`CatBoost`)
- **Permutation Test (200 rounds):** BalAcc **`41.86%`** vs chance **`33.35% ± 0.61%`**, **`p = 0.0000`** (significant).
- **Vs HeuristicBaseline:** Paired Bootstrap `ΔF1 = +0.0455` (`95% CI: [0.0292, 0.0734]`, significant), McNemar `p = 0.1203`.
- **Vs XGBoost:** Paired Bootstrap `ΔF1 = -0.0022` (`95% CI: [-0.0114, 0.0061]`), McNemar `p = 0.1687` (comparable on holdout).
- **Top SHAP Global Features (`TreeExplainer`, n=400):** `dow_sin` (`0.49192`), `historical_airline_delay_rate` (`0.21009`), `hour_sin` (`0.19623`), `airport_congestion` (`0.16278`), `airline_code_VN` (`0.14229`), `historical_airport_delay_rate` (`0.14193`).

---

## 2. Binary Ablation (`on_time` vs `disrupted = delayed | cancelled`)

| Algorithm | Accuracy | BalAcc | Macro-F1 | Disrupted Recall | ROC-AUC | PR-AUC (Disrupted) | Brier | LogLoss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MajorityClassifier | 0.7992 | 0.5000 | 0.4442 | 0.0000 | 0.5000 | 0.2008 | 0.2008 | 7.2379 |
| HeuristicBaseline | 0.7369 | 0.5565 | 0.5596 | 0.2550 | 0.5985 | 0.2547 | 0.1948 | 0.5771 |
| HistGradientBoosting | 0.7712 | 0.6164 | 0.6226 | 0.3578 | 0.7103 | 0.3710 | 0.1589 | 0.4844 |
| XGBoost | 0.7679 | 0.6240 | 0.6275 | 0.3835 | 0.7139 | 0.3883 | 0.1591 | 0.4839 |
| **CatBoost** *(Selected)* | **0.7605** | **0.6214** | **0.6227** | **0.3890** | **0.7103** | **0.3820** | **0.1630** | **0.4939** |

---

## 3. Prediction Horizon Sensitivity (`T-1h` vs `T-3h` vs `T-6h`)

| Horizon | Best Algorithm (Val) | Val Macro-F1 | Test Macro-F1 | Test BalAcc | Test Delayed Recall | Test Cancelled Recall | Test LogLoss | Test Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `T-1h` | `CatBoost` | 0.4068 | 0.4163 | 0.4241 | 0.4622 | 0.0000 | 0.5314 | 0.1767 |
| **`T-3h`** *(Primary)* | `CatBoost` | 0.4006 | 0.4154 | 0.4186 | 0.4236 | 0.0000 | 0.5170 | 0.1702 |
| `T-6h` | `CatBoost` | 0.4067 | 0.4164 | 0.4213 | 0.4401 | 0.0000 | 0.5254 | 0.1740 |

Because the frozen `CausalHistoryStore` ends at `train_cutoff` (`T-1h/T-3h/T-6h` alter causal aggregations strictly inside the training period), offline Holdout evaluation sensitivity across horizons remains modest (`0.4154`–`0.4164` Macro-F1); genuine `T-1h` operational improvements require live intraday FIDS telemetry updates.

---

## 4. Regression L1/MAE (`delay_min`, Completed Flights Only)

Train `n = 9,414` (`89` excluded), Val `n = 1,339` (`14` excluded), Test `n = 2,711` (`3` excluded).

| Regressor | Test MAE (min) | Test MedAE (min) | Test RMSE (min) | Test $R^2$ | Fit Time (ms) | Infer Time (ms) |
|---|---:|---:|---:|---:|---:|---:|
| MedianBaseline | 15.15 | 5.00 | 32.16 | -0.0425 | 38.84 | 0.08 |
| **HistGradientBoosting_MAE** *(Prod)* | **14.68** | **6.11** | **30.96** | **0.0334** | 1542.44 | 14.63 |
| XGBoost_MAE | 15.06 | 7.05 | 30.39 | 0.0692 | 846.76 | 7.30 |
| CatBoost_MAE | 14.68 | 6.16 | 30.91 | 0.0367 | 1311.03 | 4.22 |

**Statistical Comparison (`HistGradientBoosting_MAE` vs `MedianBaseline`):** Paired Bootstrap `ΔMAE = -0.48 min` (`95% CI: [-0.6425, -0.3257]`, **`significant_005 = True`**).

---

## 5. Unsupervised Discovery & Drift Monitoring

- **K-Means ($k=4$ on 10 operational+weather features):** Sample Silhouette = `0.2090` (Full = `0.1959`), Davies-Bouldin = `1.6748`, Calinski-Harabasz = `548.0`, **5-Seed ARI = `0.7958`** (`min = 0.4894`).
- **PCA (3D):** Explains **`52.40%`** cumulative variance (`23.32% + 16.56% + 12.52%`).
- **Isolation Forest (`contamination=0.03`):** Flags **`408` flights (`3.01%`)**, which exhibit **`33.33%`** disruption rate vs **`26.82%`** for normal flights.
- **FP-Growth Association Rules:** `98` raw candidate rules pruned to **`34` non-redundant rules**; **`29/30` (96.7%)** `Status=delayed` rules validated on Holdout Test (`4` `Status=cancelled` rules observed low holdout count due to only `2` cancellations in Test window).
- **4-Tier Drift Monitor (Train+Val vs Holdout Test):**
  - Overall = **`CRITICAL`**, Max PSI = **`4.6229`** on causal expanding rate features (`historical_airline_delay_rate = 4.6229`, `historical_airport_delay_rate = 4.5591` due to early-train cold-start rates vs matured holdout priors; operational weather `wind_gusts_10m` PSI = `0.1869` `WARNING`, `visibility` `0.0760` `STABLE`, `precipitation` `0.0182` `STABLE`).
  - Concept Drift = **`ALERT`** (`-8.67` pp disruption shift from `28.75%` to `20.08%`, triggering `retrain_recommended = True`).
