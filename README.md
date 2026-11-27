# FlightRisk VN — Hệ thống Khai Phá Dữ Liệu & Dự Báo Chậm / Hủy Chuyến Bay Việt Nam

[![Python 3.13+](https://img.shields.io/badge/Python-3.13%2B-0d5c58.svg)](#)
[![Architecture](https://img.shields.io/badge/Workflow-CRISP--DM--Inspired%20(7%20Stages)-0284c7.svg)](#)
[![Data Lake](https://img.shields.io/badge/Medallion-Raw%20%E2%86%92%20Bronze%20%E2%86%92%20Silver%20%E2%86%92%20Gold-0f766e.svg)](#)
[![Serving](https://img.shields.io/badge/API-FastAPI%20%2B%20Uvicorn-059669.svg)](#)

---

## 1. Project Overview

**FlightRisk VN** (`ScrapeData`) là nền tảng **Khai phá Dữ liệu & Dự báo Trước Giờ Bay ($T - 3\text{h}$)** cho mạng lưới hàng không Việt Nam (`SGN`, `HAN`, `DAD`, `CXR`, `PQC`...), được tổ chức theo chuẩn package `src-layout` (`src/flight_delay/`), kiến trúc **Medallion (`raw` $\to$ `bronze` $\to$ `silver` $\to$ `gold`)**, cơ chế kiểm soát rò rỉ dữ liệu (`leakage_guard.py` + `CausalHistoryStore`), quản trị mô hình **SHA-256** và phục vụ qua **FastAPI**.

> **Lưu ý thuật ngữ:** Quy trình **CRISP-DM** nguyên bản gồm 6 giai đoạn; dự án thiết kế một luồng **CRISP-DM-inspired workflow** gồm **7 công đoạn kỹ thuật** (`01_data_ingestion` $\to$ `07_deployment_monitoring`) và **8 notebooks tự chạy độc lập (`00`–`07`)** để tách bạch rõ khâu hiểu bài toán, khai phá không giám sát và giám sát triển khai.

---

## 2. Business Problem

- Tình trạng chậm/hủy chuyến gây thiệt hại lớn cho hãng bay, cảng hàng không và hành khách, nhưng dữ liệu bảng điện tử sân bay (**FIDS**) thực tế có cấu trúc không đồng nhất, nhiều ảnh chụp lặp theo thời gian (`T-24h` $\to$ thực tế), lỗi trừ giờ qua nửa đêm và thường trộn lẫn các ca `diverted`/`unknown`.
- Các mô hình ngây thơ dùng trạng thái hậu kiểm (`actual_time`, `delay_min`) hoặc tỷ lệ trễ tính trên toàn tập sẽ gây **rò rỉ thông tin tương lai (Data Leakage)** và sụp đổ khi dự báo thực tế trước giờ khởi hành.

---

## 3. Project Objectives

1. **Chuẩn hóa dữ liệu đa nguồn về 4 tầng Medallion** từ dữ liệu thực tế thu thập được mà không làm mất trạng thái gốc (`Bronze 12 FIELDS` $\to$ `Silver 24 CLEANED_FIELDS` $\to$ `Gold 66 X + 3 targets`).
2. **Đảm bảo Zero Future Leakage tại mốc dự báo $T - 3\text{h}$:** khóa cứng đầu vào trước giờ bay, dùng kho lịch sử theo điểm thời gian (`CausalHistoryStore`: `outcome_known_at < prediction_time`) và chặn mọi biến hậu kiểm bằng `leakage_guard.py`.
3. **Giải quyết 3 bài toán khai phá & dự báo:**
   - *Phân loại 3 lớp ICAO (`target`):* `on_time (0)`, `delayed >= 15m (1)`, `cancelled (2)` (+ *Binary Disrupted Ablation* và *Horizon Ablations* $T-1\text{h}, T-3\text{h}, T-6\text{h}$).
   - *Hồi quy độ trễ (`delay_min`, L1/MAE):* chỉ tính trên các chuyến hoàn tất (`cancelled` giữ `NaN`, không gán `0`).
   - *Khai phá không giám sát (`artifacts/unsupervised/`):* phân cụm chế độ khai thác (`K-Means`), giảm chiều (`PCA`), phát hiện bất thường (`Isolation Forest`) và khai phá luật kết hợp (`FP-Growth` kiểm chứng trên Holdout Test).
4. **Đánh giá nghiêm ngặt theo chuỗi thời gian** (`Chronological 70/10/20`, `Expanding Temporal CV`, `Paired Bootstrap CI`, `McNemar`, `Permutation Test`) và phục vụ **Fail-Closed FastAPI + SHA-256 Verification**.

---

## 4. System Architecture

### 4.1. Luồng xử lý 7 công đoạn (`src/flight_delay/`)

```text
+-----------------------------------------------------------------------------------+
| 01. DATA INGESTION (src/flight_delay/ingestion/)                                  |
|   data/raw/{fids,aviationstack,weather}/ (58 mixed cols, 92,150 raw rows)         |
|   data/debug_data/{playwright,open_meteo,aviationstack}/ (debug captures only)    |
|   --> data/bronze/standardized_snapshots.csv (12 FIELDS, 93,081 rows)             |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
| 02. DATA ENGINEERING & CONTRACTS (src/flight_delay/data_engineering/)             |
|   Modulo 1440 · SHA-256 3-tier Dedup · Open-Meteo Merge · SilverFlightContract    |
|   --> data/silver/flights_silver.csv (24 CLEANED_FIELDS, 13,570 valid legs)       |
|   --> data/silver/excluded_outcomes.csv (40 rows: diverted/unknown segregated)    |
+-----------------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
| 03. DATA MINING, EDA & LEAKAGE GUARD (src/flight_delay/data_mining/)              |
|   EDA · Statistical Tests · CausalHistoryStore · FeatureBuilder · leakage_guard   |
|   --> data/gold/{flights_gold,train,val,test}.csv (66 X + 3 targets)              |
|   --> data/gold/causal_history_{store,metadata}.json · data/manifests/            |
+-----------------------------------------------------------------------------------+
                                        |
          +-----------------------------+-----------------------------+
          v                                                           v
+---------------------------------------+   +---------------------------------------+
| 04. MODELING & UNSUPERVISED           |   | 05–06. EVALUATION & GOVERNANCE        |
|  - Classification (3-Class, Binary, H)|   |  - Temporal CV · Bootstrap · McNemar  |
|  - Regression L1/MAE (Completed only) |   |  - Permutation (200x) · SHAP · Slices |
|  - K-Means · PCA · IsoForest          |   |  - Registry + SHA-256 Checksum Gate   |
|  - FP-Growth -> artifacts/unsupervised|   |  --> artifacts/ & reports/            |
+---------------------------------------+   +---------------------------------------+
                                                                |
                                                                v
                                            +---------------------------------------+
                                            | 07. DEPLOYMENT & MONITORING           |
|                                           |  - models/*.joblib + LOCK manifest    |
|                                           |  - FastAPI (/api/predict, /, /docs)   |
|                                           |  - Data/Feature Drift (PSI, KS, Chi2) |
|                                           |  - Concept Drift (Ground-Truth Label) |
+-------------------------------------------+---------------------------------------+
```

### 4.2. Giải thích từng thư mục (Folder Defense Guide)

| Thư mục | Tại sao cần? | Dữ liệu đầu vào | Sản phẩm đầu ra | Cách kiểm tra | Notebook / Module dùng |
|---|---|---|---|---|---|
| `configs/` | Tập trung hóa toàn bộ tham số, đường dẫn, schema & ngưỡng | YAML chỉnh tay (`8` files) | Hằng số cấu hình `PROJECT_CFG`, `SCHEMA_CFG`... | `tests/unit/test_config_and_contracts.py` | Toàn bộ `src/flight_delay/` |
| `data/raw/` | Kho nguồn thô chính thức, bất biến để tái xử lý | FIDS CSV, Aviationstack CSV, Open-Meteo CSV | Đầu vào cho Ingestion (`92,150` dòng) | Kiểm tra số dòng & file nguồn | `ingestion/`, NB `#01` |
| `data/debug_data/` | Lưu vết HTML/JSON lỗi khi cào để gỡ lỗi scraper | Playwright HTML, Open-Meteo & API JSON gốc | Không đưa thẳng vào Bronze nếu chưa kiểm nguồn | Kiểm tra đối chiếu khi scraper lỗi | Chỉ phục vụ debug (`collect`) |
| `data/bronze/` | Đồng bộ đa nguồn về đúng `12 FIELDS`, giữ nguyên `status` thô | `data/raw/` | `standardized_snapshots.csv` (`93,081` dòng) | `RawSnapshotContract`, `test_ingestion_and_cleaning.py` | `ingestion/`, `data_engineering/`, NB `#01` |
| `data/silver/` | Làm sạch, khử trùng 3 tầng SHA-256, ghép thời tiết, tách nhãn nhiễu | `data/bronze/` + `weather` | `flights_silver.csv` (`13,570`), `excluded_outcomes.csv` (`40`) | `SilverFlightContract`, `silver_quality_report.json` | `data_engineering/`, NB `#02` |
| `data/gold/` | Dữ liệu phân tích & ma trận ML không rò rỉ tại $T-3\text{h}$ | `data/silver/` | `flights_gold.csv`, `train/val/test.csv` (`66 X + 3 targets`), `causal_history_{store,metadata}.json` | `leakage_guard.py`, `GoldDatasetContract`, `tests/leakage/` | `data_mining/`, `modeling/`, NB `#03–#05` |
| `data/manifests/` | Truy xuất nguồn gốc & phiên bản dữ liệu | `collector.py`, `dataset_builder.py` | `bronze_manifest.json`, `dataset_manifest.json` | Kiểm tra `dataset_version` & phân bố nhãn | `governance/`, NB `#01`, `#03` |
| `src/flight_delay/` | Mã nguồn lõi Python theo chuẩn `src-layout` (`11` subpackages) | `configs/`, `data/` | Logic xử lý & API tái sử dụng cho mọi bước | `ruff check`, `pytest` (`33` tests) | `main.py`, `scripts/`, `notebooks/` |
| `artifacts/` | Lưu kết quả huấn luyện, quản trị & khai phá (`unsupervised/`) | `modeling/`, `governance/`, `visualization/` | `models/` (versioned bundles + `checksums.sha256`), `encoders/`, `scalers/`, `manifests/`, `figures/`, `business/`, `unsupervised/` | `test_modeling_and_governance.py` | `governance/`, `visualization/`, NB `#04–#06` |
| `models/` | Chỉ chứa model runtime đã phê duyệt để FastAPI phục vụ | `artifacts/models/` + `runner.py` | `best_classifier.joblib`, `best_regressor.joblib`, `kmeans_model.joblib`, `isolation_forest.joblib`, `LOCK_pipeline_manifest.json` | Xác minh SHA-256 fail-closed trước khi nạp | `inference/model_loader.py`, `api/`, NB `#06–#07` |
| `reports/` | Báo cáo số liệu JSON có cấu trúc theo từng giai đoạn | `data_engineering/`, `eda/`, `evaluation/`, `monitoring/` | `data_quality/`, `eda/`, `experiments/`, `monitoring/drift_report.json` | `test_evaluation_and_monitoring.py`, API routes | `api/`, `visualization/`, NB `#02–#07` |
| `experiments/` | Nhật ký từng lần chạy mô hình & thử nghiệm ablation | `classification/`, `regression/` | Log run JSON & `horizon_ablations/` | Kiểm tra lịch sử tham số & metric | `governance/experiment_tracker.py` |
| `tests/` | Đảm bảo chất lượng tự động theo từng công đoạn | `src/`, `data/`, `artifacts/` | `unit`, `data_quality`, `leakage`, `integration`, `modeling`, `evaluation`, `api`, `regression` | `.\venv\Scripts\pytest.exe -v` | CI `.github/workflows/tests.yml` |
| `web/` & `.github/` | Giao diện Dashboard/Slide FastAPI & CI tự động | `reports/`, `models/`, `src/`, `tests/` | `index.html`, `presentation.html`, CI `lint.yml` & `tests.yml` | `test_browser_e2e.py`, GitHub Actions | `api/`, NB `#07` |

---

## 5. Data Sources

Toàn bộ dữ liệu thực tế thu thập trong giai đoạn **`2026-10-08` $\to$ `2026-11-02`** chia rạch ròi làm 2 vùng:
1. **`data/raw/` (Nguồn dữ liệu chính thức để tái xử lý):**
   - **Airport FIDS (`data/raw/fids/`):** 10 tệp CSV lịch bay đi/đến tại 5 cảng hàng không (`SGN`, `HAN`, `DAD`, `CXR`, `PQC`).
   - **Aviationstack REST API (`data/raw/aviationstack/`):** Dữ liệu đối chứng và chuyến bay hủy (`aviationstack_api_202610.csv`).
   - **Open-Meteo Weather (`data/raw/weather/`):** Thời tiết theo giờ tại tọa độ từng sân bay (`precipitation`, `wind_gusts_10m`, `visibility`, `weather_code`).
2. **`data/debug_data/` (Chỉ phục vụ gỡ lỗi & kiểm tra scraper):**
   - Chứa ảnh chụp HTML/JSON thô (`playwright/`, `open_meteo/`, `aviationstack/`) để truy vết khi parser gặp lỗi; **không** nạp thẳng vào `Bronze` như dữ liệu chính thức nếu chưa qua chuẩn hóa và kiểm tra nguồn gốc.

---

## 6. Bronze–Silver–Gold Architecture

- **Bronze (`data/bronze/standardized_snapshots.csv`):** `92,150` dòng thô (`58` cột hỗn hợp) $\to$ `93,081` bản ghi chuẩn **12 `FIELDS`** (`source`, `airport`, `direction`, `scraped_at`, `flight_date`, `airline_code`, `flight_no`, `other_airport`, `scheduled_time`, `estimated_time`, `actual_time`, `status`). Giữ nguyên chuỗi `status` gốc.
- **Silver (`data/silver/`):** Chuẩn hóa ngày giờ **Modulo 1440** qua nửa đêm, khử trùng 3 tầng **SHA-256** theo `leg_key`, ghép thời tiết theo giờ, phân loại nhãn chuẩn và tách riêng ca nhiễu:
  - `flights_silver.csv`: **`13,570` chặng bay hợp lệ** (`24 CLEANED_FIELDS`: `on_time = 9,904` [`72.98%`], `delayed = 3,571` [`26.32%`], `cancelled = 95` [`0.70%`]).
  - `excluded_outcomes.csv`: **`40` bản ghi `diverted`/`unknown`** được tách riêng, không ép vào 3 lớp hợp lệ.
- **Gold (`data/gold/` — `dataset v2026.11.02-h3-4716bcd4`):**
  - `flights_gold.csv`: bảng đặc trưng đầy đủ tại mốc dự báo $T - 3\text{h}$.
  - `train.csv` (`9,503`), `val.csv` (`1,353`), `test.csv` (`2,714`): tách chặt chẽ **`66` biến đầu vào `X`** (`39` One-Hot + `13` liên tục chuẩn hóa bằng thống kê Train + `14` cờ nhị phân) với **3 biến mục tiêu** (`target`, `target_binary`, `delay_min`).
  - `causal_history_store.json` + `causal_history_metadata.json`: kho lịch sử theo điểm thời gian (`bisect_left` trên `outcome_known_at < prediction_time`). *Lưu ý:* tên gọi `CausalHistoryStore` chỉ kho lịch sử chống rò rỉ tương lai theo thời gian, **không phải** mô hình suy luận nhân quả kinh tế lượng; hiện dung lượng nhỏ (`~1.18 MB`) phù hợp JSON và có metadata khuyến nghị chuyển sang `.parquet` khi vượt $\ge 100{,}000$ bản ghi.
- **Analytical Outputs (`artifacts/unsupervised/`):** `clustering_results.csv`, `anomaly_results.csv`, và `mined_association_rules.csv` được tách khỏi `data/gold/` để phân biệt rõ *dữ liệu đầu vào* với *kết quả khai phá*.

Chi tiết đầy đủ xem tại [Data Dictionary & Contracts](docs/data_dictionary.md).

---

## 7. Notebook Execution Order

Tất cả 8 notebooks trong `notebooks/` được đánh số thống nhất `00` $\to$ `07`, chạy độc lập theo đúng thứ tự (không phụ thuộc biến còn sót trong bộ nhớ của notebook khác):

| Thứ tự | Notebook | Nhiệm vụ | Đầu vào $\to$ Đầu ra |
|---:|---|---|---|
| 0 | [00_business_understanding.ipynb](notebooks/00_business_understanding.ipynb) | Xác định mục tiêu 3 lớp ICAO ($T-3\text{h}$), hồi quy `delay_min` & KPI | `configs/*.yaml` $\to$ `artifacts/business/{project_scope,evaluation_plan}.json` |
| 1 | [01_data_ingestion_bronze.ipynb](notebooks/01_data_ingestion_bronze.ipynb) | Kiểm kê nguồn thô `data/raw/`, giữ nguyên `status` về **12 `FIELDS`** | `data/raw/*` $\to$ `data/bronze/standardized_snapshots.csv` + `bronze_manifest.json` |
| 2 | [02_data_engineering_silver.ipynb](notebooks/02_data_engineering_silver.ipynb) | Modulo 1440, SHA-256 dedup 3 tầng, ghép Open-Meteo, tách `excluded` | `data/bronze/*` $\to$ `data/silver/{flights_silver,excluded_outcomes}.csv` |
| 3 | [03_eda_and_feature_engineering_gold.ipynb](notebooks/03_eda_and_feature_engineering_gold.ipynb) | EDA thống kê, `CausalHistoryStore`, `leakage_guard`, Split 70/10/20 | `data/silver/*` $\to$ `data/gold/{flights_gold,train,val,test}.csv` + `causal_history_metadata.json` |
| 4 | [04_modeling_classification_regression.ipynb](notebooks/04_modeling_classification_regression.ipynb) | Huấn luyện Baselines, HGB, XGBoost, CatBoost (3-Class, Binary, Horizons, MAE) | `data/gold/*` $\to$ `reports/experiments/{classification,regression}_results.json` |
| 5 | [05_unsupervised_discovery.ipynb](notebooks/05_unsupervised_discovery.ipynb) | K-Means (`k=2..8`, 5-seed ARI), PCA 3D, Isolation Forest, FP-Growth | `data/gold/*` $\to$ `artifacts/unsupervised/*` + `unsupervised_results.json` |
| 6 | [06_evaluation_and_governance.ipynb](notebooks/06_evaluation_and_governance.ipynb) | Temporal CV, Bootstrap/McNemar/Permutation, Registry & SHA-256 Lock | `artifacts/` & `models/` $\to$ `evaluation_report.json` + `model_registry.json` |
| 7 | [07_deployment_monitoring.ipynb](notebooks/07_deployment_monitoring.ipynb) | Kiểm chứng Parity, `predict_risk()` fail-closed, Data Drift vs Concept Drift | `models/` & `data/gold/` $\to$ `reports/monitoring/drift_report.json` |

---

## 8. Model Development

- **Supervised Classification (Primary 3-Class @ $T-3\text{h}$):** So sánh `MajorityClassifier`, `HeuristicBaseline`, `HistGradientBoosting`, `XGBoost`, `CatBoost` (tuyển chọn mô hình tốt nhất dựa trên Validation `Macro-F1` với `class_weight`/`sample_weight` cân bằng lớp hiếm).
- **Binary Disrupted Ablation (`on_time` vs `delayed | cancelled`) & Horizon Ablations ($T-1\text{h}, T-3\text{h}, T-6\text{h}$):** Đo đánh đổi giữa 3 lớp chi tiết và dự báo gián đoạn gộp.
- **Supervised Regression (L1/MAE):** Chỉ huấn luyện và đánh giá trên các chuyến bay hoàn tất (`MedianBaseline`, `HistGradientBoosting_MAE`, `XGBoost_MAE`, `CatBoost_MAE`).
- **Unsupervised Pattern Discovery (`artifacts/unsupervised/`):** K-Means ($k=4$, 5-seed ARI = `0.7958`), PCA 3D (`52.40%` phương sai giải thích), Isolation Forest (`3.01%` bất thường), FP-Growth (`34` luật phi trùng lặp, `29/30` luật trễ tái lập trên Holdout Test).

---

## 9. Evaluation Metrics & Monitoring

- **Phân loại (3 lớp mất cân bằng nghiêm trọng):**
  - `CatBoost` đạt trên Holdout Test (`2,714` chuyến): **`ROC-AUC OvR = 0.8082`**, **`Macro-F1 = 0.4154`**, **`PR-AUC Macro = 0.5402`**, **`Delayed Recall = 42.36%`**, **`On-Time Precision = 85.14%`**, **`Accuracy = 74.98%`**, Permutation Test ($200\times$) `p = 0.0000`.
  - *Đánh giá theo chuỗi thời gian (`src/flight_delay/evaluation/temporal_backtesting.py`):* Expanding 3-fold Temporal CV đạt **`Macro-F1 = 0.4330 ± 0.0561`**, tách biệt rõ với đánh giá tĩnh thông thường.
  - *Binary Disrupted Ablation:* `CatBoost` đạt **`BalAcc = 62.14%`**, **`Macro-F1 = 0.6227`**, **`ROC-AUC = 0.7103`**.
- **Hồi quy độ trễ:** `HistGradientBoosting_MAE` đạt **`MAE = 14.68 phút`**, **`MedAE = 6.11 phút`** (cải thiện có ý nghĩa thống kê so với `MedianBaseline 15.15 phút`, Paired Bootstrap `p < 0.05`).
- **Giám sát Drift (`src/flight_delay/monitoring/`):** Phân tách **Distribution Drift** (`data_drift.py`: PSI + KS-test 2 mẫu + $\chi^2$, ghi nhận `Max PSI = 4.6229` trên tỷ lệ lịch sử tích lũy) và **Concept Drift** (`concept_drift.py`: cần nhãn thực tế hậu kiểm, ghi nhận độ lệch tỷ lệ gián đoạn **`8.67` điểm %** $\to$ `retrain_recommended = True`).

---

## 10. Installation & Setup

Yêu cầu **Python 3.11+** (khuyến nghị `Python 3.13`):

```powershell
# 1. Tạo và kích hoạt môi trường ảo
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Cài đặt toàn bộ thư viện (Pipeline + ML + API + Kiểm thử)
pip install -r requirements.txt
pip install -r requirements-ml.txt
pip install pytest httpx ruff playwright
```

---

## 11. How to Run

```powershell
# Chạy toàn bộ pipeline 7 bước (Steps 2 -> 7, cập nhật datasets, models, SHA-256 lock & reports)
.\venv\Scripts\python.exe main.py all

# Khởi chạy FastAPI Dashboard, Slide thuyết trình (/presentation) & Swagger Docs (/docs)
.\venv\Scripts\python.exe main.py serve

# (Tùy chọn) Triển khai bằng Docker Compose nếu chạy môi trường container
docker compose up --build

# Kiểm tra lint & toàn bộ kiểm thử tự động (bao gồm leakage_guard, contracts, parity, governance & E2E)
.\venv\Scripts\ruff.exe check --no-cache src scripts tests main.py
.\venv\Scripts\pytest.exe -v
```

Dự án có sẵn CI GitHub Actions tại [.github/workflows/lint.yml](.github/workflows/lint.yml) và [.github/workflows/tests.yml](.github/workflows/tests.yml) để kiểm tra tự động Ruff, import và các unit/contract/leakage tests độc lập dữ liệu trên mỗi lần push/PR.

---

## 12. Project Limitations

1. **Độ dài chuỗi thời gian ngắn (`26` ngày: `2026-10-08` $\to$ `2026-11-02`):** Dữ liệu chưa bao phủ biến động mùa vụ dài hạn (Tết, bão hè) nên có dịch chuyển phân phối (`Max PSI = 4.6229`) khi tỷ lệ trễ lũy tiến tích lũy dần từ Train sang Holdout Test.
2. **Độ hiếm cực đoan của lớp `cancelled` trên tập Test:** Tập dữ liệu có `95` chuyến hủy (`0.70%`), nhưng do phân chia theo trục thời gian, phần Holdout Test cuối kỳ chỉ chứa **`2` chuyến hủy (`0.07%`)**, khiến `Macro-F1` 3 lớp chịu phạt mạnh ở lớp `cancelled` và cần bổ sung dữ liệu hủy chuyến khi vận hành dài hạn.
3. **Lưu trữ lịch sử khi mở rộng quy mô:** `data/gold/causal_history_store.json` (`~1.18 MB`) phù hợp cho tập dữ liệu hiện tại; khi dữ liệu vượt $\ge 100{,}000$ bản ghi cần chuyển sang định dạng `causal_history_store.parquet` như đã ghi chú trong `data/gold/causal_history_metadata.json`.
4. **Hạn chế nguồn FIDS công khai:** Thiếu biến chuyên sâu về lịch quay vòng thân tàu bay (*tail-number rotation*), slot không lưu (ATC) và bảo trì kỹ thuật theo thời gian thực.

---

Tài liệu kỹ thuật chi tiết: [Architecture](docs/architecture.md) · [Data Dictionary & Contracts](docs/data_dictionary.md) · [Methodology](docs/methodology.md) · [Experiments](docs/experiments.md) · [Model Cards](docs/model_cards.md) · [API Reference](docs/api.md).
