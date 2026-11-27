# FlightRisk VN — Hệ thống Khai Phá Dữ Liệu & Dự Báo Chậm / Hủy Chuyến Bay Việt Nam

[![Python 3.13+](https://img.shields.io/badge/Python-3.13%2B-0d5c58.svg)](#)
[![Architecture](https://img.shields.io/badge/CRISP--DM-7%20Steps-0284c7.svg)](#)
[![Data Lake](https://img.shields.io/badge/Medallion-Raw%20%E2%86%92%20Bronze%20%E2%86%92%20Silver%20%E2%86%92%20Gold-0f766e.svg)](#)
[![Serving](https://img.shields.io/badge/API-FastAPI%20%2B%20Uvicorn-059669.svg)](#)

**FlightRisk VN** là nền tảng **Khai phá Dữ liệu & Dự báo Trước Giờ Bay ($T - 3\text{h}$)** cho mạng đường bay Việt Nam (`SGN`, `HAN`, `DAD`, `CXR`, `PQC`...), xây dựng theo quy trình **7 bước CRISP-DM**, kiến trúc dữ liệu **Medallion (`raw` $\to$ `bronze` $\to$ `silver` $\to$ `gold`)**, cơ chế **Causal Feature Store** chống rò rỉ tương lai, **Model Governance SHA-256**, và phục vụ qua **FastAPI**.

---

## 1. Hiểu nhanh dự án trong 60 giây (5 câu hỏi cốt lõi)

### 1.1. Dữ liệu đến từ đâu?
Tất cả dữ liệu gốc thu thập từ 3 nguồn thực tế được lưu bất biến tại `data/raw/` và `data/debug_data/`:
1. **Airport FIDS (Playwright):** 10 file CSV lịch bay đi/đến tại 5 cảng lớn (`SGN`, `HAN`, `DAD`, `CXR`, `PQC`) + ảnh chụp HTML gốc (`data/debug_data/playwright/`).
2. **Aviationstack REST API:** Dữ liệu đối chứng & danh mục hủy chuyến (`data/raw/aviationstack/aviationstack_api_202610.csv` + JSON gốc tại `data/debug_data/aviationstack/`).
3. **Open-Meteo Hourly Weather:** Chuỗi thời tiết theo giờ tại tọa độ từng sân bay (`precipitation`, `wind_gusts_10m`, `visibility`, `weather_code`) lưu tại `data/raw/weather/weather_hourly.csv` + JSON gốc tại `data/debug_data/open_meteo/`.

### 1.2. Dữ liệu đi qua những tầng nào? (Project Architecture)

```text
+-----------------------------------------------------------------------------------+
| 01. DATA INGESTION                                                                |
|   data/raw/{fids,aviationstack,weather}/ (58 mixed cols, 92,150 raw rows)         |
|   --> data/bronze/standardized_snapshots.csv (12 FIELDS, 93,081 rows, raw status) |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
| 02. DATA ENGINEERING & CONTRACTS (Modulo 1440 · SHA-256 3-tier Dedup · Weather)   |
|   --> data/silver/flights_silver.csv (24 CLEANED_FIELDS, 13,570 valid legs)       |
|   --> data/silver/excluded_outcomes.csv (40 rows: diverted/unknown segregated)    |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
| 03. DATA MINING, EDA & CAUSAL FEATURE STORE (Chronological Split 70/10/20 @ T-3h) |
|   CausalHistoryStore (bisect_left < prediction_time) · FeatureBuilder Parity      |
|   --> data/gold/flights_gold.csv + train/val/test.csv (66 clean X features)       |
+-----------------------------------------------------------------------------------+
                                        |
          +-----------------------------+-----------------------------+
          v                                                           v
+---------------------------------------+   +---------------------------------------+
| 04. MODELING & UNSUPERVISED           |   | 05–06. EVALUATION & GOVERNANCE        |
|  - Classification (3-Class ICAO T-3h) |   |  - Temporal CV · Bootstrap · McNemar  |
|  - Ablations (Binary & T-1h/3h/6h)    |   |  - Permutation (200x) · SHAP · Slices |
|  - Regression L1/MAE (Completed only) |   |  - Registry + SHA-256 Checksum Gate   |
|  - K-Means (ARI) · PCA · IsoForest    |   |  --> artifacts/ & reports/            |
|  - FP-Growth (Holdout Validated)      |   +---------------------------------------+
+---------------------------------------+                       |
                                                                v
                                            +---------------------------------------+
                                            | 07. DEPLOYMENT & 4-TIER MONITORING    |
                                            |  - FastAPI (/api/predict, /, /docs)   |
                                            |  - Fail-closed predict_risk()         |
                                            |  - PSI + KS-test + Concept Drift      |
                                            +---------------------------------------+
```

### 1.3. Notebook nào thực hiện công việc gì?

| Notebook | Giai đoạn | Nhiệm vụ chính | Dữ liệu vào &rarr; Kết quả ra |
|---|---|---|---|
| [00_business_understanding.ipynb](notebooks/00_business_understanding.ipynb) | Lập kế hoạch | Xác định mục tiêu 3 lớp ICAO (`T-3h`), hồi quy `delay_min` & KPI | `configs/*.yaml` &rarr; `artifacts/business/{project_scope,evaluation_plan}.json` |
| [01_data_ingestion_bronze.ipynb](notebooks/01_data_ingestion_bronze.ipynb) | Bước 1: Ingestion | Kiểm kê 11 file thô (`58` cột), giữ nguyên `status` gốc về **12 `FIELDS`** | `data/raw/*` &rarr; `data/bronze/standardized_snapshots.csv` + `bronze_manifest.json` |
| [02_data_engineering_silver.ipynb](notebooks/02_data_engineering_silver.ipynb) | Bước 2: Silver | Modulo 1440, SHA-256 dedup 3 tầng, ghép Open-Meteo, tách `diverted`/`unknown` | `data/bronze/*` &rarr; `data/silver/{flights_silver,excluded_outcomes}.csv` |
| [03_eda_and_feature_engineering_gold.ipynb](notebooks/03_eda_and_feature_engineering_gold.ipynb) | Bước 3: Gold | EDA trực quan, Pearson/Spearman/Cramér's V, `CausalHistoryStore`, Split 70/10/20 | `data/silver/*` &rarr; `data/gold/{flights_gold,train,val,test}.csv` + `dataset_manifest.json` |
| [04_modeling_classification_regression.ipynb](notebooks/04_modeling_classification_regression.ipynb) | Bước 4: Supervised | Huấn luyện Baselines, HGB, XGBoost, CatBoost (3-Class, Binary, Horizons, MAE) | `data/gold/*` &rarr; `reports/experiments/{classification,regression}_results.json` |
| [05_unsupervised_discovery.ipynb](notebooks/05_unsupervised_discovery.ipynb) | Bước 4: Unsupervised | K-Means (`k=2..8` + 5-seed ARI), PCA 3D, Isolation Forest, FP-Growth Holdout | `data/gold/*` &rarr; `data/gold/mined_association_rules.csv` + `unsupervised_results.json` |
| [06_evaluation_and_governance.ipynb](notebooks/06_evaluation_and_governance.ipynb) | Bước 5–6: Eval & Gov | Permutation/Bootstrap/McNemar, Subgroups, xác minh Model Registry & SHA-256 | `artifacts/` &rarr; `evaluation_report.json` + `model_registry.json` + `checksums.sha256` |
| [07_deployment_monitoring.ipynb](notebooks/07_deployment_monitoring.ipynb) | Bước 7: Serve & Drift | Kiểm chứng `Training-Serving Parity`, `predict_risk()` fail-closed, 4-Tier PSI/KS | `artifacts/` + `data/gold/` &rarr; `reports/monitoring/drift_report.json` |

### 1.4. Kết quả được lưu ở đâu? (Phân biệt `data/` vs `models/` vs `artifacts/` vs `reports/`)

| Thư mục | Vai trò chuyên biệt | Nội dung chính |
|---|---|---|
| `data/` | **Data Lake 4 tầng (`raw`/`bronze`/`silver`/`gold`)** | Dữ liệu thô (`raw/`), chuẩn 12 cột (`bronze/`), sạch 24 cột (`silver/`), ma trận ML 66 cột & tập luật (`gold/`), manifests (`manifests/`). |
| `models/` | **Serving Runtime Artifacts** | Model `.joblib` đang dùng trực tiếp trên FastAPI (`best_classifier.joblib`, `best_regressor.joblib`, `kmeans_model.joblib`, `isolation_forest.joblib`) + `LOCK_pipeline_manifest.json`. |
| `artifacts/` | **Model Governance & Assets** | Các bundle theo phiên bản (`artifacts/models/<name>_v2.0.0/` gồm `checksums.sha256`, `model_card.md`, `requirements.lock`, `evaluation.json`), `encoders/`, `scalers/`, `manifests/{model_registry.json,data_lineage.json}`, `figures/` (7 biểu đồ PNG), `business/`. |
| `reports/` | **Báo cáo Thực nghiệm & Giám sát** | JSON thống kê có cấu trúc: `data_quality/`, `eda/`, `experiments/` (`classification_results.json`, `regression_results.json`, `evaluation_report.json`, `horizon_ablation_results.json`, `unsupervised_results.json`), `monitoring/` (`drift_report.json`). |
| `experiments/` | **Experiment Tracking Logs** | Log chi tiết từng lần chạy mô hình & workspace thí nghiệm chân trời (`horizon_ablations/`). |

### 1.5. Làm sao kiểm tra tính đúng đắn và tái lập kết quả?
1. **Tái lập toàn bộ pipeline 7 bước (Seed cố định `42`):**
   ```powershell
   .\venv\Scripts\python.exe main.py all
   ```
2. **Kiểm tra tự động toàn bộ công đoạn (30+ automated tests — Leakage, Temporal, Contracts, Parity, Governance SHA-256, API & E2E):**
   ```powershell
   .\venv\Scripts\pytest.exe -v
   .\venv\Scripts\ruff.exe check src scripts tests main.py
   ```
3. **Xác minh tính toàn vẹn SHA-256 của Production Bundles:**
   - Mỗi bundle trong `artifacts/models/<model>_v2.0.0/` chứa `checksums.sha256`; [model_loader.py](src/flight_delay/inference/model_loader.py) kiểm tra `approved == True` và đối chiếu SHA-256 **trước khi nạp** (tự động từ chối fail-closed nếu bị chỉnh sửa hoặc thiếu phê duyệt).

---

## 2. Folder Structure

```text
ScrapeData/
├── configs/                        # Cấu hình YAML tập trung (project, sources, airports, airlines, schema, features, experiments, monitoring)
├── data/
│   ├── raw/                        # Tầng Raw gốc: fids/, aviationstack/, weather/
│   ├── debug_data/                 # Kho lưu trữ nguyên thủy (2,923+ captures Playwright/Open-Meteo/Aviationstack)
│   ├── bronze/                     # Tầng Bronze: standardized_snapshots.csv (12 FIELDS, giữ status gốc)
│   ├── silver/                     # Tầng Silver: flights_silver.csv (24 CLEANED_FIELDS) + excluded_outcomes.csv
│   ├── gold/                       # Tầng Gold: flights_gold.csv, train/val/test.csv (66 X), causal_history_store.json, mined_association_rules.csv
│   └── manifests/                  # bronze_manifest.json, dataset_manifest.json
├── src/flight_delay/               # Mã nguồn Python chuẩn src-layout
│   ├── common/                     # Constants, exceptions, SHA-256 hashing, logging
│   ├── ingestion/                  # Bước 1: Collectors & Source Registry
│   ├── data_engineering/           # Bước 2: Modulo 1440, Dedup SHA-256, Contracts, Quality
│   ├── data_mining/                # Bước 3: Profiling, EDA, FeatureBuilder, CausalHistoryStore, Encoding, Datasets
│   ├── modeling/                   # Bước 4: Baselines, Classification (3-Class/Binary/Horizon), Regression L1, Unsupervised, SHAP
│   ├── evaluation/                 # Bước 5: Imbalanced Metrics, Temporal CV, Bootstrap, McNemar, Permutation, Subgroups, Error Analysis
│   ├── governance/                 # Bước 6: Experiment Tracker, Model Registry, Checksums, Model Cards, Lineage
│   ├── inference/                  # Bước 7: Fail-Closed Model Loader, Validation, predict_risk(), Audit Builder
│   ├── monitoring/                 # Bước 7: Data Quality, PSI + KS-test, Concept Drift, Performance Alerts
│   ├── visualization/              # Sinh biểu đồ Matplotlib/Seaborn & Slide thuyết trình HTML
│   └── api/                        # FastAPI + Pydantic REST & Dashboard Server
├── notebooks/                      # 8 Notebooks chuẩn CRISP-DM (00..07) kèm sẵn bảng và biểu đồ
├── scripts/                        # Scripts chạy độc lập từng công đoạn + .bat launchers
├── tests/                          # Bộ tests phân tầng: unit, data_quality, leakage, integration, modeling, evaluation, api, regression
├── artifacts/                      # Approved model bundles, checksums SHA-256, encoders, scalers, registry manifests, figures, business
├── models/                         # Serving runtime .joblib & LOCK_pipeline_manifest.json
├── reports/                        # Báo cáo JSON (data_quality, eda, experiments, monitoring)
├── experiments/                    # Experiment run logs & horizon ablation outputs
├── docs/                           # architecture.md, data_dictionary.md, methodology.md, experiments.md, model_cards.md, api.md
├── web/                            # Giao diện Dashboard tương tác & Slide 7 bước CRISP-DM
├── main.py                         # CLI thống nhất
├── pyproject.toml                  # Metadata gói flightrisk-vn, cấu hình pytest & ruff
├── Dockerfile & docker-compose.yml # Triển khai container hóa
└── .gitignore                      # Chặn venv, caches, .env/secrets, raw data & file binary .joblib
```

---

## 3. Tóm tắt Kết Quả Thực Nghiệm (`dataset v2026.11.02-h3-4716bcd4`)

- **Dữ liệu (`2026-10-08` &rarr; `2026-11-02`):** `92,150` dòng thô &rarr; `93,081` Bronze snapshots &rarr; `13,570` Silver legs hợp lệ (`on_time: 72.98%`, `delayed: 26.32%`, `cancelled: 0.70%`) + `40` ca `diverted`/`unknown` tách riêng. Split theo thời gian: `Train 9,503` / `Val 1,353` / `Test 2,714` (`66` biến ML).
- **Phân loại 3 lớp ICAO (`T-3h`):** `CatBoost` (tuyển trên Validation) đạt Test **`ROC-AUC OvR = 0.8082`**, **`Macro-F1 = 0.4154`**, **`PR-AUC Macro = 0.5402`**, `delayed_recall = 42.36%`, Permutation Test `p = 0.0000` *(lưu ý Test chỉ có 2 ca hủy = `0.07%`)*.
- **Binary Disrupted Ablation:** `CatBoost` đạt **`BalAcc = 62.14%`**, **`Macro-F1 = 0.6227`**, **`ROC-AUC = 0.7103`**.
- **Hồi quy L1/MAE (chuyến hoàn tất):** `HistGradientBoosting_MAE` đạt **`MAE = 14.68 phút`**, **`MedAE = 6.11 phút`** (cải thiện có ý nghĩa thống kê so với `MedianBaseline 15.15m`: Paired Bootstrap `p < 0.05`).
- **Unsupervised & Monitoring:** K-Means `k=4` (5-Seed `ARI = 0.7958`), PCA 3D (`52.40%` variance), Isolation Forest (`3.01%`), FP-Growth (`34` luật phi trùng lặp, `29/30` luật trễ holdout-validated), Drift Monitor cảnh báo `CRITICAL` (`max PSI = 4.6229` trên causal historical rates; Concept Drift `-8.67` điểm % $\to$ `retrain_recommended = True`).

---

## 4. Khởi chạy Nhanh

```powershell
# Chạy toàn bộ pipeline 7 bước
.\venv\Scripts\python.exe main.py all

# Khởi chạy FastAPI Dashboard & Swagger Docs tại http://127.0.0.1:5000
.\venv\Scripts\python.exe main.py serve

# Chạy toàn bộ kiểm thử & kiểm tra chất lượng mã
.\venv\Scripts\pytest.exe -v
.\venv\Scripts\ruff.exe check src scripts tests main.py
```

Tài liệu chi tiết: [Architecture](docs/architecture.md) · [Data Dictionary & Contracts](docs/data_dictionary.md) · [Methodology](docs/methodology.md) · [Experiments](docs/experiments.md) · [Model Cards](docs/model_cards.md) · [API Reference](docs/api.md).
