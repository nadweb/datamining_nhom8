# Sổ Tay Hướng Dẫn Thực Hiện Nhiệm Vụ Cho Từng Thành Viên
> **Dự án:** Khai phá dữ liệu & Dự báo rủi ro chuyến bay Việt Nam (FlightRisk VN)  
> **Cấu trúc phân công:**
> - Thành viên 1: Part 1 - Data Cleaning & Integration
> - Thành viên 2: Part 2 - Data Transformation & Validation
> - Thành viên 3: Part 3 - Feature Engineering
> - Thành viên 4: Part 4 - Encoding & Data Normalization
> - Thành viên 5: Part 5 - Machine Learning (Data Splitting, Supervised & Unsupervised Learning)
> - **3 Thành viên cuối làm Web:**
>   - Thành viên 6: Part 6A - Web Backend API & Model Serving
>   - Thành viên 7: Part 6B - Web Frontend UI & Interactive Simulator
>   - Thành viên 8: Part 6C - Web Analytics Dashboard, Client Logic & System Integration

---

## Bản Đồ 6 Bước Quy Trình Modern Data Mining (CRISP-DM / KDD)

| Bước | Tên bước chuẩn | Câu hỏi trọng tâm | Sản phẩm đầu ra cụ thể của nhóm |
| :---: | :--- | :--- | :--- |
| **1** | **Business Understanding** | Cần giải quyết việc gì? Đo bằng gì? | Tài liệu mục tiêu, chỉ số Macro F1, Hợp đồng dữ liệu `docs/schema.md` |
| **2** | **Data Understanding** | Dữ liệu ở đâu? Có đặc điểm gì? | 12,830 dòng snapshot từ 5 bảng FIDS sân bay + Aviationstack + Weather API *(Xem chi tiết tại `notebooks/00_data_collection_and_eda.ipynb`)* |
| **3** | **Data Preprocessing** | Làm sao để sạch và máy học hiểu được? | `flights_cleaned_global.csv`, `flights_features.csv`, bộ transformer `scaler.joblib`, `encoder.joblib` |
| **4** | **Data Mining & Modeling** | Quy luật và mô hình dự báo là gì? | `best_classifier.joblib`, `best_regressor.joblib`, `kmeans_model.joblib`, `isolation_forest.joblib` |
| **5** | **Evaluation & Interpretation** | Mô hình tốt cỡ nào? Tri thức gì mới? | `supervised_metrics.json`, `cluster_profiles.json`, Ma trận nhầm lẫn, Top Feature Importances |
| **6** | **Deployment & Serving** | Người dùng sử dụng thế nào? | Hệ thống Web tương tác `http://127.0.0.1:5000/`, REST API, Báo cáo & Slide thuyết trình |

---

## Nơi Tìm Tệp Dữ Liệu & Bản Đồ Dòng Chảy Dữ Liệu (Data Lineage)

> [!IMPORTANT]
> **Quy định quan trọng về dữ liệu cho tất cả thành viên:**
> 1. **Dữ liệu mẫu chạy nhanh (Sample Data - 300 dòng):** Nằm tại [`data/flights_sample.csv`](file:///c:/Users/LENOVO/Documents/ScrapeData/data/flights_sample.csv).
> 2. **Dữ liệu đầy đủ (Full Data - 12,830 dòng):** Nằm tại [`data/flights_cleaned_global.csv`](file:///c:/Users/LENOVO/Documents/ScrapeData/data/flights_cleaned_global.csv).

### Bảng tra cứu dữ liệu đầu vào & đầu ra cho từng thành viên:

| Thành viên | Nhiệm vụ | File dữ liệu ĐẦU VÀO cần lấy | Vị trí thư mục | File sản phẩm ĐẦU RA sinh ra | Bàn giao cho ai? |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **TV 1** | Data Cleaning & Integration | `data/raw/*.csv` hoặc `flights_cleaned_global.csv` | Thư mục `data/raw/` hoặc `data/` | Báo cáo tích hợp, bảng so sánh nguồn | Cho TV2 |
| **TV 2** | Transformation & Validation | `flights_cleaned_global.csv` (hoặc `flights_sample.csv`) | Thư mục `data/` | `flights_cleaned_global.csv` (24 cột chuẩn) | Cho TV3 |
| **TV 3** | Feature Engineering | `data/flights_cleaned_global.csv` | Thư mục `data/` | `data/processed/flights_features.csv` (46 cột) | Cho TV4 & TV5 |
| **TV 4** | Encoding & Normalization | `data/processed/flights_features.csv` | Thư mục `data/processed/` | `scaler.joblib`, `encoder.joblib`, `preprocessing_metadata.json` (5 leakage columns guarded, train median imputation) | Cho TV5 & TV6 |
| **TV 5** | Data Splitting & ML Modeling | `data/processed/flights_features.csv` & bộ scaler/encoder của TV4 | `data/processed/` | • `train.csv` (11,172 dòng), `test.csv` (2,793 dòng)<br>• `models/best_classifier.joblib`<br>• `models/best_regressor.joblib`<br>• `models/kmeans_model.joblib`<br>• `models/isolation_forest.joblib`<br>• `models/*.json` | Cho TV6, TV7, TV8 |
| **TV 6** | Web Backend API | Các mô hình `.joblib` & metadata `.json` của TV5 | Thư mục `models/` & `data/processed/` | Endpoint REST API `/api/predict`, `/api/stats` | Cho TV7 & TV8 |
| **TV 7** | Web Frontend Simulator | Giao diện HTML/CSS kết nối API | Thư mục `web/templates/` & `web/static/` | Giao diện tương tác mô phỏng chuyến bay | Cho TV8 & Người dùng |
| **TV 8** | Web Dashboard & UAT | REST API từ TV6 & file `data/mined_association_rules.csv` | `http://127.0.0.1:5000/` & `data/` | Bảng điều khiển phân tích cụm, luật kết hợp | Hoàn thiện sản phẩm |

### Cấu trúc cây thư mục dữ liệu (`data/` & `models/`):
```text
ScrapeData/
├── data/
│   ├── flights_cleaned_global.csv     <-- TẬP DỮ LIỆU SẠCH 24 CỘT (12,830 dòng, TV1 & TV2 hoàn thiện)
│   ├── flights_sample.csv             <-- DỮ LIỆU MẪU 300 DÒNG (Đã commit Git để test nhanh)
│   ├── weather_hourly.csv             <-- Dữ liệu thời tiết theo giờ từ Open-Meteo
│   ├── mined_association_rules.csv    <-- 15+ luật kết hợp khai phá bởi FP-Growth
│   ├── api_usage.json                 <-- Nhật ký số lượt gọi Aviationstack API
│   ├── raw/                           <-- 11 tệp CSV thô thu thập từ FIDS sân bay và API
│   │   ├── aviationstack_api_202610.csv
│   │   ├── guide_sgn_202610.csv, guide_sgn_202611.csv
│   │   └── guide_han_*.csv, guide_dad_*.csv, guide_pqc_*.csv, guide_cxr_*.csv
│   ├── debug_data/                    <-- Tệp gốc nguyên bản (1,400+ file HTML và JSON)
│   │   ├── playwright/ (bản gốc FIDS sân bay)
│   │   ├── aviationstack/ (bản gốc JSON API)
│   │   └── open_meteo/ (bản gốc thời tiết)
│   └── processed/                     <-- DỮ LIỆU ĐÃ QUA BIẾN ĐỔI (Do TV3, TV4, TV5 sinh ra)
│       ├── flights_features.csv       <-- 46 cột đặc trưng (TV3 sinh ra)
│       ├── encoder.joblib             <-- Bộ OneHotEncoder đã fit (TV4 sinh ra)
│       ├── scaler.joblib              <-- Bộ StandardScaler đã fit (TV4 sinh ra)
│       ├── preprocessing_metadata.json<-- Danh sách tên đặc trưng & nhãn (TV4 sinh ra, 64 features)
│       ├── train.csv                  <-- Tập Train 80% (11,172 dòng, TV5 sinh ra)
│       └── test.csv                   <-- Tập Test 20% (2,793 dòng, TV5 sinh ra)
└── models/                            <-- CÁC MÔ HÌNH HỌC MÁY ĐÃ HUẤN LUYỆN (TV5 sinh ra)
    ├── best_classifier.joblib         <-- Mô hình phân loại 3 lớp tốt nhất (Random Forest, macro-F1 = 0.419, ROC-AUC = 0.810)
    ├── best_regressor.joblib          <-- Mô hình hồi quy dự báo phút trễ (HistGradientBoosting)
    ├── kmeans_model.joblib            <-- Mô hình phân cụm 4 nhóm rủi ro (Data-derived names)
    ├── pca_model.joblib               <-- Mô hình giảm chiều không gian 2D/3D (3 components)
    ├── isolation_forest.joblib        <-- Mô hình phát hiện 392 chuyến bay dị thường (2.81%)
    ├── supervised_metrics.json        <-- Toàn bộ chỉ số CV TimeSeriesSplit, Accuracy, F1, ROC-AUC, MAE, Grouped Importances
    ├── cluster_profiles.json          <-- Bảng đặc tả chi tiết 4 cụm rủi ro K-Means suy ra từ dữ liệu thực tế
    └── unsupervised_metrics.json      <-- Chỉ số PCA, Isolation Forest (z-score σ), FP-Growth sorted by lift
```

---

## Hướng Dẫn Chung Trước Khi Bắt Đầu
1. **Môi trường:** Kích hoạt môi trường ảo `venv` trước khi làm việc:
   - Windows: `venv\Scripts\activate`
   - macOS / Linux: `source venv/bin/activate`
2. **Quy trình Git:**
   - Tạo nhánh theo đúng tên được phân công: `git checkout -b <tên-nhánh>`
   - Luôn cập nhật code mới nhất từ `main`: `git pull --rebase origin main`
   - Chỉ chỉnh sửa các file thuộc trách nhiệm của mình, không sửa file của bạn khác.
   - **Xóa output của Jupyter Notebook trước khi commit** (vào menu *Kernel $\rightarrow$ Restart & Clear Output* hoặc chạy `nbstripout`).
   - Mở Pull Request (PR) trên GitHub và nhờ trưởng nhóm duyệt trước khi merge vào `main`.

---

## THÀNH VIÊN 1: Data Cleaning & Integration Audit (Part 1)
- **Nhánh Git:** `feature/cleaning-integration`
- **Tệp phụ trách chính:** `pipeline/processor.py`, `pipeline/collector.py`
- **Notebook nộp:** `notebooks/01_member1_cleaning_integration.ipynb`
- **Dữ liệu cần để chạy:** `data/flights_cleaned_global.csv` (hoặc mẫu `data/flights_sample.csv`)

### 1. Bạn cần làm gì và chỉnh sửa những gì?
- **Khảo sát dữ liệu đa nguồn:** Tích hợp dữ liệu từ 5 website bảng FIDS sân bay (Playwright) và Aviationstack REST API.
- **Khử trùng lặp đa cấp:**
  - *Cấp 1 (Cùng phía sân bay):* Giữ bản ghi mới nhất, giàu thông tin nhất tại cùng `(airport, direction, flight_date, flight_no, scheduled_time)`.
  - *Cấp 2 (Cấp chuyến bay leg):* Ghép nối bản ghi ở bảng đi (Departures) và bảng đến (Arrivals) của cùng 1 chuyến bay thành đúng 1 dòng duy nhất.
- **Tính toán hệ số Cohen's Kappa:** Đo lường độ nhất quán gán nhãn giữa các nguồn đối với các sự kiện chuyến bay trùng nhau.

### 2. Yêu cầu sản phẩm phải nộp:
1. File mã nguồn `pipeline/processor.py` chứa logic hàm `prepare_dataset()`, `label_and_report()`, `airline_table()`, `source_agreement()`.
2. File notebook `notebooks/01_member1_cleaning_integration.ipynb` đã hoàn thiện các bảng thống kê và biểu đồ cột so sánh các hãng bay.
3. Tạo Pull Request (PR) trên GitHub từ nhánh `feature/cleaning-integration` vào `main`.

### 3. Cách tự kiểm tra (Self-check):
Chạy lệnh trong Terminal:
```bash
python -m pipeline.processor
```
**Kết quả mong đợi:**
- Tổng mẫu hợp lệ đạt $\ge 10,000$ chuyến bay (thực tế đạt **12,830 chuyến**).
- Thấy rõ 6 nguồn dữ liệu: `guide_sgn` (4,850), `guide_han` (3,761), `guide_dad` (1,541), `guide_pqc` (1,103), `aviationstack_api` (914), `guide_cxr` (661).
- Bảng so sánh OTP các hãng hiển thị rõ: Vietjet trễ 39.1%, Vietnam Airlines trễ 23.9%, Sun PhuQuoc trễ 13.7%.

### 4. Đóng góp vào Báo cáo Word / Slide:
- Viết **Chương 2: Thu thập, Làm sạch và Tích hợp dữ liệu**.
- Đưa biểu đồ phân bố nguồn dữ liệu và bảng đối chiếu tỷ lệ OTP thực tế so với số liệu của Cục Hàng không (CAAV) vào báo cáo.

---

## THÀNH VIÊN 2: Data Transformation & Validation (Part 2)
- **Nhánh Git:** `feature/transformation`
- **Tệp phụ trách chính:** `pipeline/transformation.py`, `pipeline/check_data.py`
- **Notebook nộp:** `notebooks/02_member2_data_transformation.ipynb`
- **Dữ liệu cần để chạy:** `data/flights_cleaned_global.csv`, `data/flights_sample.csv`

### 1. Bạn cần làm gì và chỉnh sửa những gì?
- **Tính số phút trễ (`calculate_delay`):** Xử lý chính xác trường hợp chuyến bay cất cánh/hạ cánh qua nửa đêm (Midnight rollover).
  - *Ví dụ:* Lịch bay 23:50, thực tế 00:20 hôm sau $\rightarrow$ Phải tính ra trễ **+30 phút**, không được ra số âm.
- **Gán nhãn đích (`label_status`):** Phân chia nhãn `status_clean` thành 3 lớp (`on_time`, `delayed`, `cancelled`) theo ngưỡng 15 phút.
  - *Ràng buộc bắt buộc:* Chuyến bay bị hủy (`cancelled`) phải để trống số phút trễ (`delay_min = NaN`).
- **Xây dựng công cụ kiểm định (`check_data.py`):** Kiểm tra 8 ràng buộc khắt khe của Hợp đồng dữ liệu (Data Contract).

### 2. Yêu cầu sản phẩm phải nộp:
1. File `pipeline/transformation.py` chứa các hàm xử lý thời gian, gán nhãn, ghép thời tiết.
2. File `pipeline/check_data.py` hỗ trợ chạy CLI kiểm tra tính toàn vẹn của dữ liệu.
3. Notebook `notebooks/02_member2_data_transformation.ipynb` ghi lại kết quả 8 bài test kiểm định.
4. Tạo Pull Request từ `feature/transformation` vào `main`.

### 3. Cách tự kiểm tra (Self-check):
Chạy 2 lệnh sau trong Terminal:
```bash
python main.py check-data
python -m unittest tests.test_data_validation tests.test_label_rules tests.test_iata_mapping
```
**Kết quả mong đợi:**
- Màn hình in thông báo: `ALL DATA QUALITY CHECKS PASSED SUCCESSFULLY. DATA CONTRACT IS VALID.`
- Tất cả các bài unit test liên quan đến nửa đêm, ngưỡng 15/30 phút và mã IATA đều báo `ok`.

### 4. Đóng góp vào Báo cáo Word / Slide:
- Viết **Chương 3 (Mục 3.1 & 3.2): Tiền xử lý & Biến đổi dữ liệu**.
- Trình bày bảng quy tắc gán nhãn, giải thích công thức xử lý nửa đêm và đưa bảng kiểm định chất lượng 8 chỉ tiêu vào báo cáo.

---

## THÀNH VIÊN 3: Feature Engineering (Part 3)
- **Nhánh Git:** `feature/feature-engineering`
- **Tệp phụ trách chính:** `features/features.py`
- **Notebook nộp:** `notebooks/03_member3_feature_engineering.ipynb`
- **Dữ liệu cần để chạy:** `data/flights_cleaned_global.csv`

### 1. Bạn cần làm gì và chỉnh sửa những gì?
- **Sáng tạo các đặc trưng phi tuyến từ dữ liệu gốc:**
  - *Chu kỳ thời gian:* Tính hàm $\sin, \cos$ cho giờ bay trong ngày (chu kỳ 24h) và thứ trong tuần (chu kỳ 7 ngày) để mô hình học tính liên tục qua nửa đêm.
  - *Chỉ số nguy cơ thời tiết (`weather_hazard_score`):* Kết hợp đa biến lượng mưa, gió giật, tầm nhìn và dông sét thành 1 thang điểm liên tục từ 0.0 đến 1.0.
  - *Tải sân bay & Giờ cao điểm:* Tạo cờ `is_peak_hour` (sáng 06h-09h30, tối 17h-20h30) và mức độ ùn ứ `airport_load`.
  - *Tuyến bay trục vàng:* Đánh dấu `is_golden_triangle` cho các chuyến bay kết nối giữa 3 trung tâm lớn SGN - HAN - DAD.
  - *Tương tác đặc trưng:* `load_x_weather` (sân bay quá tải kết hợp thời tiết xấu sẽ nhân đôi rủi ro trễ).
- **Quy tắc vàng:** Tuyệt đối không được dùng 4 cột rò rỉ (`delay_min`, `actual_time`, `status`, `status_detail`).

### 2. Yêu cầu sản phẩm phải nộp:
1. File `features/features.py` chứa hàm `build_engineered_features()` và `extract_single_flight_features()`.
2. Chạy lệnh để sinh ra file dữ liệu: `data/processed/flights_features.csv` (12,830 dòng x 46 cột).
3. Notebook `notebooks/03_member3_feature_engineering.ipynb` trực quan hóa biểu đồ tương quan giữa các đặc trưng mới và tỷ lệ trễ.
4. Tạo Pull Request từ `feature/feature-engineering` vào `main`.

### 3. Cách tự kiểm tra (Self-check):
Chạy lệnh trong Terminal:
```bash
python main.py features
python -m unittest tests.test_features
```
**Kết quả mong đợi:**
- Terminal thông báo: `Successfully engineered features: 12,830 rows x 46 columns`.
- Sinh ra đủ 22 cột đặc trưng mới.
- Các bài unit test kiểm tra miền giá trị $[-1, 1]$ của $\sin/\cos$ đều báo `ok`.

### 4. Đóng góp vào Báo cáo Word / Slide:
- Viết **Chương 3 (Mục 3.3): Kỹ nghệ đặc trưng (Feature Engineering)**.
- Trình bày công thức tính chu kỳ thời gian $\sin/\cos$, công thức điểm nguy cơ thời tiết, và biểu đồ ma trận tương quan (Correlation heatmap) giữa các biến mới và biến mục tiêu.

---

## THÀNH VIÊN 4: Encoding & Data Normalization (Part 4)
- **Nhánh Git:** `feature/encoding-normalization`
- **Tệp phụ trách chính:** `modeling/encoding.py`
- **Notebook nộp:** `notebooks/04_member4_encoding_normalization.ipynb`
- **Dữ liệu cần để chạy:** `data/processed/flights_features.csv`

### 1. Bạn cần làm gì và chỉnh sửa những gì?
- **Cô lập rò rỉ thông tin (Data Leakage Guard):** Loại bỏ hoàn toàn 5 cột rò rỉ (`delay_min`, `actual_time`, `status`, `status_detail`, `is_delayed_for_analysis`) và các mã định danh không mang tính dự báo (`leg_key`, `flight_no`, `origin`, `destination`...) khỏi ma trận đặc trưng $X$.
- **Xử lý giá trị khuyết thiếu an toàn:** Lưu và điền giá trị trung vị từ tập Train (`train_medians`) cho các biến liên tục bị thiếu, tránh data leakage.
- **Mã hóa biến phân loại danh mục (Categorical Encoding):**
  - Dùng `OneHotEncoder(handle_unknown="ignore")` cho 5 biến danh mục (`airline_code`, `airport`, `direction`, `route_type`, `carrier_category`).
  - *Ý nghĩa:* Khi ứng dụng chạy trong thực tế gặp hãng hàng không mới hoặc mã sân bay mới chưa từng thấy, preprocessor sẽ không bị crash mà tự động gán toàn bộ vector 0 cho danh mục đó.
- **Chuẩn hóa đặc trưng liên tục (Continuous Feature Scaling):**
  - Dùng `StandardScaler` cho 10 biến số liên tục (`airport_load`, `precipitation`, `wind_gusts_10m`, `visibility`, `weather_hazard_score`, `load_x_weather`, `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`).
  - Đưa toàn bộ các thang đo khác biệt (tầm nhìn 10,000m, gió 60km/h, mưa 20mm) về cùng phân phối chuẩn $\mathcal{N}(0, 1)$, giúp các thuật toán tối ưu hội tụ nhanh và không bị thiên lệch.
- **Bảo toàn các cờ nhị phân (Binary Flags Passthrough):** Duy trì nguyên trạng giá trị 0/1 cho 14 biến cờ (`is_peak_hour`, `is_heavy_traffic`, `is_rain`, `is_severe_wind`, `is_storm`, `is_golden_triangle`...).
- **Đóng gói đối tượng tiền xử lý (`FlightDataPreprocessor`):** Cung cấp các phương thức `fit(X)`, `transform(X)`, `fit_transform(X)`, `save(target_dir)`, `load(target_dir)` chuẩn hóa scikit-learn.
- **Lưu trữ các Transformer Artifacts:** Xuất các file `scaler.joblib`, `encoder.joblib`, `preprocessing_metadata.json` vào thư mục `data/processed/`.
- *(Lưu ý: Nhiệm vụ chia tập dữ liệu Train/Test theo chuỗi ngày bay được thực hiện phối hợp với Thành viên 5 trong Part 5).*

### 2. Yêu cầu sản phẩm phải nộp:
1. File mã nguồn `modeling/encoding.py` hoàn chỉnh class `FlightDataPreprocessor`.
2. Bộ file transformer sẵn sàng phục vụ huấn luyện và web inference:
   - `data/processed/scaler.joblib`
   - `data/processed/encoder.joblib`
   - `data/processed/preprocessing_metadata.json`
3. Notebook `notebooks/04_member4_encoding_normalization.ipynb` thực thi quy trình fit/transform, đối chiếu số chiều trước/sau One-Hot và kiểm định Zero Data Leakage.
4. Tạo Pull Request (PR) từ nhánh `feature/encoding-normalization` vào `main`.

### 3. Cách tự kiểm tra (Self-check):
Chạy các lệnh trong Terminal:
```bash
python main.py encode
python -m unittest tests.test_modeling.TestModelingPipeline.test_zero_leakage_in_training_features
```
**Kết quả mong đợi:**
- Terminal in thông báo lưu thành công các artifact: `scaler.joblib`, `encoder.joblib`, `preprocessing_metadata.json`.
- Kích thước ma trận đặc trưng sau One-Hot mở rộng đầy đủ 40+ chiều.
- Bài unit test `test_zero_leakage_in_training_features` báo `ok`, xác nhận không còn bất kỳ cột rò rỉ nào.

### 4. Đóng góp vào Báo cáo Word / Slide:
- Viết **Chương 4 (Mục 4.1): Kỹ thuật Mã hóa biến Phân loại và Chuẩn hóa Thang đo Đặc trưng**.
- Trình bày bảng so sánh danh mục biến trước và sau khi One-Hot Encoding, công thức toán học của `StandardScaler` ($z = \frac{x - \mu}{\sigma}$), và giải thích cơ chế phòng ngừa lỗi crash trong sản xuất với `handle_unknown="ignore"`.

---

## THÀNH VIÊN 5: Data Splitting, Supervised & Unsupervised Machine Learning (Part 5)
- **Nhánh Git:** `feature/ml-modeling` (hoặc `feature/ml-insight-analysis`)
- **Tệp phụ trách chính:** `modeling/train.py`, `modeling/unsupervised.py`
- **Notebook nộp:** `notebooks/05_member5_machine_learning.ipynb`
- **Dữ liệu cần để chạy:** `data/processed/flights_features.csv`, các transformer từ TV4 trong `data/processed/`

### 1. Bạn cần làm gì và chỉnh sửa những gì?
Thành viên 5 là vị trí then chốt về Machine Learning của dự án, phụ trách trọn gói từ **Phân chia dữ liệu**, **Học có giám sát (Supervised)** đến **Học không giám sát & Khai phá tri thức (Unsupervised)**:

#### Phần 1: Phân chia Dữ liệu theo Chuỗi Thời gian (Chronological Train/Test Split)
- **Sắp xếp thời gian:** Sắp xếp tập dữ liệu theo thứ tự ngày bay `flight_date` và giờ bay `scheduled_time`.
- **Phân chia 80% Train / 20% Test:** Áp dụng mốc cắt ngày `2026-10-31` (80% các ngày đầu làm Train: 11,172 dòng, 20% các ngày sau làm Test: 2,793 dòng trên toàn bộ 13,965 dòng dữ liệu đặc trưng).
- **Nguyên tắc vàng - Chống Lookahead Bias:**
  - Tuyệt đối **không dùng shuffle ngẫu nhiên**. Trong bài toán thời gian thực của hàng không, việc shuffle ngẫu nhiên sẽ làm mô hình "học lỏm" trước xu hướng thời tiết và tải sân bay của tương lai để dự báo quá khứ (Data Leakage theo thời gian).
  - Mô hình chỉ được học trên quá khứ (Train) và đánh giá trên các ngày tương lai độc lập (Test).
- **Áp dụng Preprocessor từ TV4:** Fit bộ tiền xử lý `FlightDataPreprocessor` CHỈ TRÊN TẬP TRAIN, sau đó biến đổi (transform) trên cả Train và Test.
- **Xuất tệp dữ liệu phân chia:** Lưu `data/processed/train.csv` (11,172 dòng) và `data/processed/test.csv` (2,793 dòng).

#### Phần 2: Huấn luyện & Đánh giá Mô hình Học Có Giám Sát (Supervised Learning - `modeling/train.py`)
- **Huấn luyện 3 mô hình phân loại đa lớp (3-Class Classification) + Dummy Baseline:**
  - Dự báo nhãn `status_clean` của chuyến bay: `0 = on_time`, `1 = delayed`, `2 = cancelled`.
  - *Mô hình 1 (Baseline Linear):* **Logistic Regression** với tham số `class_weight='balanced'` để chống mất cân bằng lớp.
  - *Mô hình 2 (Ensemble Bagging):* **Random Forest Classifier** (150 cây quyết định, `max_depth=12`, `class_weight='balanced'`).
  - *Mô hình 3 (Gradient Boosting):* **HistGradientBoosting Classifier** (`max_iter=150`).
  - *Baseline So Sánh:* **DummyClassifier(strategy='most_frequent')** để đối chiếu với tỷ lệ đoán mù.
- **Lựa chọn mô hình chuẩn khoa học (Model Selection via TimeSeriesSplit CV):**
  - Đánh giá bằng Cross-Validation phân đoạn thời gian 3-folds (`TimeSeriesSplit`) trên tập **Train**. Tập **Test** chỉ được chấm duy nhất 1 lần để báo cáo khách quan, không được dùng để lựa chọn mô hình.
  - Lựa chọn mô hình tốt nhất theo **macro-F1 CV**: **Random Forest** đạt macro-F1 CV = 0.4189 (vượt trội so với baseline 0.2764) và ROC-AUC Test = 0.8103.
- **Trích xuất Đặc trưng Quan trọng (Permutation Importance & Grouped Importances):**
  - Dùng `permutation_importance` trên tập Test (tránh lỗi rỗng của HistGradientBoosting), đồng thời gộp các biến one-hot về biến gốc (`hour`, `airport_load`, `day_of_week`, `visibility`, `route_type`...).
- **Huấn luyện mô hình Hồi quy dự báo độ trễ (Delay Regression):**
  - Xây dựng mô hình **HistGradientBoosting Regressor** tối ưu MAE trên tập các chuyến bay trễ để dự báo số phút trễ liên tục (`delay_min`). Đánh giá MAE $\approx 18.2$ phút, $R^2 \approx 0.05$.
- **Đóng gói mô hình:** Lưu `models/best_classifier.joblib`, `models/best_regressor.joblib` và `models/supervised_metrics.json`.

#### Phần 3: Học Không Giám Sát & Khai phá Tri thức (Unsupervised Learning - `modeling/unsupervised.py`)
- **Phân cụm K-Means ($k=4$ cụm rủi ro vận hành với tên động suy ra từ dữ liệu):**
  - Quét kiểm định $k \in [2, 8]$ lưu toàn bộ Inertia và Silhouette scores (cố định random_state).
  - Tên cụm được suy luận tự động từ trung bình giờ bay và điều kiện thời tiết thực tế:
    - *Cụm 0 (Morning, Clear weather):* Chuyến sáng, thời tiết tốt (5,833 chuyến ~ 41.8%, tỷ lệ trễ 23.9% - rủi ro Vừa).
    - *Cụm 1 (Afternoon, Windy/Reduced visibility):* Chuyến chiều, gió lớn/tầm nhìn giảm (3,070 chuyến ~ 22.0%, tỷ lệ trễ 30.4% - rủi ro Cao).
    - *Cụm 2 (Evening, Clear weather):* Chuyến tối cao điểm, thời tiết tốt (4,816 chuyến ~ 34.5%, tỷ lệ trễ 30.3% - rủi ro Cao).
    - *Cụm 3 (Afternoon, Rain/Storm):* Chuyến chiều gặp mưa bão bất lợi (246 chuyến ~ 1.8%, tỷ lệ trễ 39.0% - rủi ro Cao).
  - Lưu `models/kmeans_model.joblib` và `models/cluster_profiles.json`.
- **Giảm chiều dữ liệu PCA (Principal Component Analysis):**
  - Giảm chiều 2D và 3D, giải thích tổng cộng $\approx 43.66\%$ phương sai dữ liệu. Lưu `models/pca_model.joblib`.
- **Phát hiện Bất thường với Isolation Forest (Anomaly Detection):**
  - Với tham số $contamination = 3\%$, phát hiện **392 chuyến bay bất thường** (2.81%).
  - Báo cáo rõ đặc trưng sai lệch nhiều nhất so với bình thường: `is_storm` ($+3.88\sigma$), `precipitation` ($+3.85\sigma$), `is_heavy_rain` ($+3.75\sigma$).
  - Lưu `models/isolation_forest.joblib`.
- **Khai phá Luật kết hợp FP-Growth (Association Rules Mining):**
  - Lọc và sắp xếp giảm dần theo Lift các luật kết hợp từ `data/mined_association_rules.csv`.
  - Lưu `models/unsupervised_metrics.json`.

### 2. Yêu cầu sản phẩm phải nộp:
1. Hai file mã nguồn: `modeling/train.py` và `modeling/unsupervised.py`.
2. Năm file mô hình trong thư mục `models/`:
   - `models/best_classifier.joblib`
   - `models/best_regressor.joblib`
   - `models/kmeans_model.joblib`
   - `models/pca_model.joblib`
   - `models/isolation_forest.joblib`
3. Ba file kết quả phân tích JSON:
   - `models/supervised_metrics.json`
   - `models/cluster_profiles.json`
   - `models/unsupervised_metrics.json`
4. Hai file dữ liệu đã phân chia Train/Test:
   - `data/processed/train.csv` (10,264 dòng)
   - `data/processed/test.csv` (2,566 dòng)
5. Notebook `notebooks/05_member5_machine_learning.ipynb` chạy đầy đủ kết quả thực nghiệm cả 3 phần.
6. Tạo Pull Request (PR) từ nhánh `feature/ml-modeling` vào `main`.

### 3. Cách tự kiểm tra (Self-check):
Chạy các lệnh trong Terminal:
```bash
python main.py encode
python main.py train
python main.py unsupervised
python -m unittest discover tests -v
```
**Kết quả mong đợi:**
- Lệnh `python main.py train` in ra:
  - Mốc cắt ngày: `2026-10-31`. Train: 11,172 dòng | Test: 2,793 dòng.
  - Chọn mô hình bằng TimeSeriesSplit CV trên Train: Random Forest đạt macro-F1 CV = 0.4189 (baseline Dummy 0.2764), Test ROC-AUC = 0.8103.
  - Mô hình hồi quy báo MAE $\approx 18.2$ phút.
  - Top permutation features: `hour_sin` (~10%), `airport_load` (~8.8%), `hour_cos` (~7.5%), `dow_sin` (~7.4%), `visibility` (~5.6%).
- Lệnh `python main.py unsupervised` in ra:
  - Quét k=2..8 Silhouette Score và Inertia của K-Means ($k=4$).
  - Bảng hồ sơ 4 cụm rủi ro vận hành với tên cụm tự động suy ra từ giờ và thời tiết.
  - PCA 3 components in tỷ lệ phương sai tích lũy giải thích được $\approx 43.66\%$.
  - Isolation Forest báo phát hiện **392 chuyến bay bất thường** (2.81%).
- Toàn bộ **52 bài test tự động** trong project đều báo `OK`.

### 4. Đóng góp vào Báo cáo Word / Slide:
- Viết **Chương 4 (Mục 4.2 & 4.3): Chiến lược Phân chia Dữ liệu & Huấn luyện Mô hình Học Có Giám Sát**.
  - Trình bày chiến lược Chronological Split chống Lookahead Bias so với Random Split.
  - Bảng đối chiếu hiệu năng 3 mô hình phân loại và mô hình hồi quy.
  - Đồ thị Ma trận nhầm lẫn (Confusion Matrix), đường cong ROC-AUC và biểu đồ cột Top 10 Feature Importance.
- Viết **Chương 5: Khai phá Tri thức & Mô hình Học Không Giám Sát (K-Means, PCA, Isolation Forest, FP-Growth)**.
  - Biểu đồ Elbow và Silhouette Score chứng minh lựa chọn $k=4$ cụm.
  - Bảng đặc tả hồ sơ 4 cụm rủi ro vận hành (Cluster Profiles) kèm khuyến nghị thực tiễn cho ngành hàng không.
  - Biểu đồ phân tán PCA 2D trực quan hóa không gian dữ liệu chuyến bay.
  - Phân tích thống kê 385 chuyến bay dị biệt do Isolation Forest phát hiện.
  - Bảng Top 15 luật kết hợp có Lift cao nhất trích xuất từ FP-Growth.

---

## THÀNH VIÊN 6: Web Backend API & Model Serving (Part 6A)
- **Nhánh Git:** `feature/web-backend`
- **Tệp phụ trách chính:** `modeling/predict.py`, `web/app.py`
- **Sản phẩm chính:** Dịch vụ REST API suy luận và nạp mô hình
- **Dữ liệu cần để chạy:** Các file mô hình `.joblib` trong `models/` (do TV5 cung cấp) và preprocessor trong `data/processed/`

### 1. Bạn cần làm gì và chỉnh sửa những gì?
- **Hoàn thiện hàm suy luận trung tâm `predict_risk()` (`modeling/predict.py`):**
  - Nhận các tham số chuyến bay người dùng gửi lên từ form (hãng bay, giờ bay, thời tiết, tải sân bay).
  - Gọi Preprocessor chuẩn hóa đầu vào $\rightarrow$ Gọi mô hình phân loại (TV5) tính xác suất trễ/hủy $\rightarrow$ Gọi mô hình hồi quy (TV5) tính số phút trễ $\rightarrow$ Gọi K-Means (TV5) gán cụm rủi ro $\rightarrow$ Khớp tập luật FP-Growth (TV5).
  - Trả về cấu trúc JSON đúng 100% Hợp đồng dữ liệu `docs/schema.md`.
- **Xây dựng hệ thống REST API trong `web/app.py`:**
  - `POST /api/predict`: Nhận JSON payload, gọi `predict_risk()`, trả về kết quả dự báo thời gian thực kèm xử lý lỗi (Exception handling).
  - `GET /api/stats`: Đọc và trả về kết quả đánh giá mô hình phân loại & hồi quy từ `models/supervised_metrics.json`.
  - `GET /api/clusters`: Đọc và trả về hồ sơ 4 cụm K-Means từ `models/cluster_profiles.json`.
  - `GET /api/rules`: Đọc và trả về danh sách luật kết hợp từ `data/mined_association_rules.csv`.
  - `GET /health`: Endpoint kiểm tra trạng thái hoạt động của máy chủ backend.
- **Tối ưu hiệu năng:** Đảm bảo thời gian phản hồi (latency) của API dưới 50ms cho mỗi yêu cầu dự báo.

### 2. Yêu cầu sản phẩm phải nộp:
1. File `modeling/predict.py` và `web/app.py` hoàn chỉnh, xử lý lỗi tốt, có tài liệu chú thích rõ ràng.
2. Biên bản kiểm thử các API endpoint, đo thời gian phản hồi (latency benchmark < 50ms) và xác thực tính toàn vẹn của JSON schema.
3. Tạo Pull Request từ `feature/web-backend` vào `main`.

### 3. Cách tự kiểm tra (Self-check):
Chạy trong Terminal:
```bash
python -m modeling.predict
python -m unittest tests.test_modeling.TestModelingPipeline.test_predict_risk_response_schema
```
**Kết quả mong đợi:**
- In ra chuỗi JSON đẹp mắt có đủ: `prediction`, `risk_score`, `risk_level`, `estimated_delay_min`, `probabilities`, `cluster_profile`, `matched_rules`.
- Tốc độ suy luận đạt dưới 50ms/request.

### 4. Đóng góp vào Báo cáo Word / Slide:
- Viết **Chương 6 (Mục 6.1): Kiến trúc Hệ thống & Thiết kế Dịch vụ API Backend**.
- Đưa sơ đồ kiến trúc luồng dữ liệu API (Data Pipeline Architecture) và bảng đặc tả các endpoint REST API vào báo cáo.

---

## THÀNH VIÊN 7: Web Frontend UI & Interactive Simulator (Part 6B)
- **Nhánh Git:** `feature/web-frontend`
- **Tệp phụ trách chính:** `web/templates/index.html`, `web/static/style.css`
- **Sản phẩm chính:** Giao diện người dùng mô phỏng rủi ro chuyến bay
- **Dữ liệu cần để chạy:** Trình duyệt web và máy chủ cục bộ

### 1. Bạn cần làm gì và chỉnh sửa những gì?
- **Thiết kế giao diện bảng điều khiển mô phỏng (`web/templates/index.html`):**
  - *Form thông số chuyến bay:* Thiết kế dropdown chọn hãng bay (7 hãng VN), sân bay quan sát (5 sân bay lớn), sân bay đi/đến, hướng bay (Đi/Đến), loại tuyến (Nội địa/Quốc tế), ngày bay và giờ bay.
  - *Bộ điều khiển thời tiết & bối cảnh:* Thiết kế thanh trượt (sliders) kéo chỉnh trực quan lượng mưa (0-25mm), sức gió giật (5-80 km/h), tầm nhìn (500-10,000m), hiện tượng thời tiết WMO và mật độ chuyến bay tại sân bay (Traffic load).
  - *Hộp kết quả rủi ro (Risk Card):* Thiết kế khung hiển thị kết quả trực quan với nhãn dự báo lớn, đồng hồ đo mức rủi ro tự đổi màu theo cấp độ (Xanh: LOW, Vàng: MEDIUM, Đỏ: HIGH).
  - *Thanh xác suất đa lớp (Probability Bars):* Thiết kế 3 thanh tiến trình (progress bars) hiển thị tỷ lệ % của `on_time`, `delayed`, `cancelled`.
- **Tối ưu giao diện & Phong cách (`web/static/style.css`):**
  - Viết CSS theo chuẩn Responsive Design (hiển thị đẹp trên cả màn hình máy tính và thiết bị di động).
  - Thiết kế bảng điều hướng Tab mượt mà: Tab 1 (Dự báo mô phỏng), Tab 2 (Mô hình Supervised), Tab 3 (Phân cụm K-Means), Tab 4 (Tập luật FP-Growth).

### 2. Yêu cầu sản phẩm phải nộp:
1. Tệp giao diện `web/templates/index.html` và file giao diện `web/static/style.css`.
2. Bộ ảnh chụp màn hình giao diện (UI screenshots), bố cục Wireframe và các thành phần trực quan phục vụ báo cáo và slide thuyết trình.
3. Tạo Pull Request từ `feature/web-frontend` vào `main`.

### 3. Cách tự kiểm tra (Self-check):
Khởi chạy web:
```bash
python main.py web
```
- Mở trình duyệt tại `http://127.0.0.1:5000/`.
- Kiểm tra độ sắc nét của giao diện, chuyển đổi giữa các tab, kéo thử các thanh trượt slider xem số hiển thị bên cạnh có nhảy mượt mà hay không.

### 4. Đóng góp vào Báo cáo Word / Slide:
- Viết **Chương 6 (Mục 6.2): Thiết kế Giao diện Người dùng (User Interface Design)**.
- Đưa hình ảnh thiết kế giao diện, bố cục Wireframe, bảng màu nhận diện thương hiệu và giải thích các thành phần trực quan vào báo cáo.

---

## THÀNH VIÊN 8: Web Analytics Dashboard, Client Logic & Integration (Part 6C)
- **Nhánh Git:** `feature/web-dashboard` (hoặc cùng nhánh `feature/web-frontend`)
- **Tệp phụ trách chính:** `web/static/app.js`
- **Sản phẩm chính:** Mã xử lý Client AJAX, Dashboard phân tích & Kịch bản UAT Demo
- **Dữ liệu cần để chạy:** Máy chủ backend và API (`http://127.0.0.1:5000/`)

### 1. Bạn cần làm gì và chỉnh sửa những gì?
- **Lập trình logic xử lý phía Client (`web/static/app.js`):**
  - Viết JavaScript bắt sự kiện submit form: Thu thập giá trị các ô nhập và thanh trượt $\rightarrow$ Gửi request AJAX `POST /api/predict` $\rightarrow$ Nhận kết quả JSON và cập nhật DOM động mà không cần tải lại trang.
  - Cập nhật màu sắc của Risk Badge và thanh phân bố xác suất tương ứng.
  - Hiển thị thông tin hồ sơ cụm K-Means tương ứng và danh sách các luật kết hợp cảnh báo.
- **Tải dữ liệu phân tích động cho các Tab phụ:**
  - Viết hàm `loadModelMetrics()`: Gọi `GET /api/stats` để vẽ bảng so sánh Accuracy, F1, Recall, ROC-AUC của 3 mô hình phân loại và bảng Top 10 đặc trưng quan trọng nhất.
  - Viết hàm `loadClusterProfiles()`: Gọi `GET /api/clusters` để render bảng hồ sơ 4 cụm K-Means và thông số dị thường của Isolation Forest.
  - Viết hàm `loadAssociationRules()`: Gọi `GET /api/rules` để render bảng 15 luật khai phá FP-Growth có Lift cao nhất.
- **Kịch bản Demo & Kiểm thử người dùng (UAT):** Chuẩn bị kịch bản biểu diễn demo thực tế cho buổi báo cáo đồ án của nhóm.

### 2. Yêu cầu sản phẩm phải nộp:
1. File mã nguồn JavaScript `web/static/app.js` hoàn chỉnh, không có lỗi console.
2. Biên bản kiểm thử UAT (User Acceptance Testing) và 2 kịch bản demo đối chiếu trực tiếp (Trời đẹp đúng giờ vs Mưa bão trễ chuyến).
3. Tạo Pull Request vào `main`.

### 3. Cách tự kiểm tra (Self-check):
Chạy web:
```bash
python main.py web
```
- Truy cập `http://127.0.0.1:5000/`.
- **Kịch bản test 1 (Chuyến bay an toàn):** Chọn Vietnam Airlines (`VN`), giờ bay `08:00`, lượng mưa `0mm`, gió `10km/h`, tải sân bay `15` $\rightarrow$ Bấm nút: Kết quả phải hiện **MÀU XANH (RỦI RO: LOW)**, xác suất đúng giờ $>75\%$.
- **Kịch bản test 2 (Nguy cơ trễ cao):** Chọn Vietjet (`VJ`), giờ bay `19:30`, lượng mưa `15mm`, gió `45km/h`, dông sét mạnh (WMO 95), tải sân bay `35` $\rightarrow$ Bấm nút: Kết quả lập tức chuyển sang **MÀU ĐỎ (RỦI RO: HIGH)**, xác suất trễ tăng lên $\approx 79\%$, dự báo trễ $\approx 35$ phút và hiện luật cảnh báo liên quan.
- Chuyển sang các tab khác: Kiểm tra bảng so sánh mô hình, hồ sơ cụm K-Means và tập luật FP-Growth đều tải dữ liệu thành công.

### 4. Đóng góp vào Báo cáo Word / Slide:
- Viết **Chương 6 (Mục 6.3): Kịch bản Kiểm thử Chấp nhận (UAT) & Ứng dụng Thực tế**.
- Chụp ảnh các kịch bản demo đối chiếu thực tế và chuẩn bị slide thuyết trình phần Demo sản phẩm trước hội đồng.
