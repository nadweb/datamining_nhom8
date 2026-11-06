# Vietnam Flight Delay & Risk Intelligence Platform (FlightRisk VN)

> Hệ thống khai phá dữ liệu, học máy có giám sát (Supervised) & không giám sát (Unsupervised), cùng nền tảng web dự báo rủi ro trễ/hủy chuyến bay tại 5 sân bay lớn của Việt Nam (SGN, HAN, DAD, CXR, PQC).

---



### 1. Clone mã nguồn & Thiết lập môi trường ảo
```bash
git clone https://github.com/your-org/ScrapeData.git
cd ScrapeData

# Khởi tạo môi trường ảo Python
python -m venv venv

# Kích hoạt venv
# Trên Windows (cmd / PowerShell):
venv\Scripts\activate
# Trên macOS / Linux:
source venv/bin/activate
```

### 2. Cài đặt thư viện theo phân nhóm công việc
Không cần cài toàn bộ nếu bạn chỉ phụ trách một phần:

```bash
# Lựa chọn A - Dành cho Nhóm Dữ liệu (Parts 1-2):
pip install -r requirements-pipeline.txt
playwright install chromium

# Lựa chọn B - Dành cho Nhóm Mô hình (Parts 3-5):
pip install -r requirements-ml.txt

# Lựa chọn C - Dành cho Nhóm Web (Part 6):
pip install -r requirements-web.txt

# Lựa chọn D - Cài đặt toàn bộ (Team lead / Fullstack / Testing):
pip install -r requirements.txt
```

### 3. Thiết lập biến môi trường
Tạo file `.env` từ file mẫu:
```bash
cp .env.example .env
# Chỉnh sửa khóa AVIATIONSTACK_KEY trong file .env nếu cần thu thập thêm API
```

### 4. Dữ liệu & Tự kiểm tra (Self-Check)
Dữ liệu mẫu (`data/flights_sample.csv`, 300 dòng) đã được commit sẵn trong Git để bạn có thể chạy thử nghiệm và test ngay lập tức mà không cần xin file dữ liệu lớn từ máy chủ:

```bash
# Kiểm tra tính toàn vẹn của dữ liệu mẫu hoặc dữ liệu thật
python main.py check-data

# Chạy toàn bộ 21 automated unit tests
python main.py test
```

---

##  Cấu Trúc Dự Án (Project Architecture)

```
ScrapeData/
├── .env.example                    # Biến môi trường mẫu (chứa AVIATIONSTACK_KEY, port)
├── requirements.txt                # Yêu cầu chung toàn dự án
├── requirements-pipeline.txt       # Nhóm 1-2: Playwright, bs4, requests, pandas
├── requirements-ml.txt             # Nhóm 3-5: scikit-learn, mlxtend, joblib, matplotlib
├── requirements-web.txt            # Nhóm 6: flask, jinja2, requests
├── requirements-dev.txt            # Công cụ kiểm thử: pytest, nbstripout
├── config.py                       # Cấu hình chung, tọa độ, mã IATA, danh sách hãng
├── main.py                         # CLI thống nhất toàn bộ dự án
│
├── data/
│   ├── flights_cleaned_global.csv  # Bộ dữ liệu hoàn chỉnh (12,830 dòng x 24 cột)
│   ├── flights_sample.csv          # Mẫu 300 dòng phục vụ test độc lập trên Git
│   ├── data_version.txt            # Thông tin phiên bản, ngày cập nhật, mã băm SHA-256
│   └── processed/                  # Dữ liệu train/test, scaler.joblib, encoder.joblib
│
├── docs/
│   └── schema.md                   # Hợp đồng dữ liệu (Data Contract), chống rò rỉ (leakage)
│
├── pipeline/                       # PHẦN 1 & 2: THU THẬP, LÀM SẠCH & BIẾN ĐỔI DỮ LIỆU
│   ├── collector.py                # Cào FIDS sân bay & Aviationstack REST API
│   ├── weather.py                  # Thu thập thời tiết Open-Meteo theo giờ
│   ├── transformation.py           # Tính số phút trễ, xử lý nửa đêm, gán nhãn
│   ├── processor.py                # Khử trùng đa cấp, đối chiếu nguồn, điều phối
│   └── check_data.py               # Lệnh kiểm định ràng buộc schema 24 cột
│
├── features/                       # PHẦN 3: KỸ NGHỆ ĐẶC TRƯNG (FEATURE ENGINEERING)
│   └── features.py                 # Chu kỳ giờ/ngày sin-cos, chỉ số thời tiết, tải sân bay
│
├── modeling/                       # PHẦN 4 & 5: MÃ HÓA & MÔ HÌNH HỌC MÁY
│   ├── encoding.py                 # One-Hot, StandardScaler, chia tập theo thời gian
│   ├── train.py                    # Supervised: Phân loại 3 lớp & Hồi quy độ trễ
│   ├── unsupervised.py             # Unsupervised: K-Means (k=4), PCA, Isolation Forest
│   └── predict.py                  # Giao diện suy luận chuẩn predict_risk() cho Web
│
├── web/                            # PHẦN 6: GIAO DIỆN WEB & REST API
│   ├── app.py                      # Máy chủ API (/api/predict, /api/stats, /api/clusters)
│   ├── static/                     # CSS hiện đại & JS xử lý AJAX thời gian thực
│   └── templates/index.html        # Bảng điều khiển mô phỏng rủi ro chuyến bay
│
├── notebooks/                      # NOTEBOOKS RIÊNG CHO TỪNG THÀNH VIÊN (ĐÃ BỎ OUTPUT)
│   ├── 01_member1_cleaning_integration.ipynb
│   ├── 02_member2_data_transformation.ipynb
│   ├── 03_member3_feature_engineering.ipynb
│   ├── 04_member4_encoding_normalization.ipynb
│   ├── 05_member5_machine_learning.ipynb
│   ├── 06_member6_web_backend_api.ipynb
│   ├── 07_member7_web_frontend_ui.ipynb
│   └── 08_member8_web_dashboard_integration.ipynb
│
├── tests/                          # TẬP KIỂM THỬ TỰ ĐỘNG (21 UNIT TESTS)
│   ├── test_data_validation.py     # Kiểm định 24 cột, khóa duy nhất, chuyến hủy
│   ├── test_label_rules.py         # Kiểm định qua nửa đêm, ngưỡng trễ 15/30 phút
│   ├── test_iata_mapping.py        # Kiểm định ánh xạ tên thành phố sang mã IATA
│   ├── test_features.py            # Kiểm định đặc trưng tuần hoàn và nguy cơ thời tiết
│   └── test_modeling.py            # Kiểm định chống rò rỉ dữ liệu & predict_risk
│
└── models/                         # LƯU TRỮ CÁC MÔ HÌNH ĐÃ HUẤN LUYỆN
    ├── best_classifier.joblib      # Mô hình phân loại tốt nhất (HistGradientBoosting)
    ├── best_regressor.joblib       # Mô hình hồi quy độ trễ
    ├── kmeans_model.joblib         # Mô hình phân cụm K-Means
    ├── isolation_forest.joblib     # Mô hình phát hiện dị thường
    └── supervised_metrics.json     # Báo cáo kết quả đánh giá mô hình
```

---

##  Danh Sách Lệnh CLI (`main.py`)

| Nhóm | Lệnh | Ý nghĩa |
|---|---|---|
| **Kiểm tra dữ liệu** | `python main.py check-data` | Kiểm định 8 ràng buộc dữ liệu theo Data Contract |
| **Kiểm thử tự động** | `python main.py test` | Chạy toàn bộ 21 test tự động (pytest / unittest) |
| **Kỹ nghệ đặc trưng** | `python main.py features` | Chạy Part 3: Tạo 22 đặc trưng mới (tuần hoàn, thời tiết, tải sân bay) |
| **Mã hóa & Chuẩn hóa**| `python main.py encode` | Chạy Part 4: One-Hot, StandardScaler, chia Train/Test theo ngày |
| **Supervised ML** | `python main.py train` | Chạy Part 5A: Huấn luyện phân loại 3 lớp & hồi quy số phút trễ |
| **Unsupervised ML** | `python main.py unsupervised` | Chạy Part 5B: K-Means (k=4), PCA 2D/3D, Isolation Forest |
| **Giao diện Web** | `python main.py web` | Khởi chạy máy chủ Web UI Dashboard tại `http://127.0.0.1:5000/` |
| **Thu thập dữ liệu** | `python main.py loop --every 120` | Thu thập FIDS định kỳ (chạy đa nền tảng Windows/macOS/Linux) |

---

##  Quy Tắc Làm Việc Nhóm (Team Collaboration Rules)

1. **Ngăn chặn xung đột (Zero Merge Conflict):**
   - Tuyệt đối không chỉnh sửa file của người khác. Mỗi nhánh chức năng (`feature/*`) chỉ làm việc trên thư mục và tệp được phân công theo bảng phân công nhóm.
   - **Xóa output của Jupyter Notebook trước khi commit:** Chạy lệnh `nbstripout notebooks/*.ipynb` hoặc dùng tính năng *Clear All Outputs* trong Jupyter.
2. **Quy trình Git & Pull Request:**
   - Bảo vệ nhánh `main`: Yêu cầu mở Pull Request (PR) và có ít nhất 1 phê duyệt (approval) từ trưởng nhóm trước khi merge.
   - Luôn chạy lệnh đồng bộ trước khi đẩy mã:
     ```bash
     git pull --rebase origin main
     ```
   - Chạy `python main.py test` trên máy cá nhân trước khi tạo PR.
3. **Tuân thủ Hợp đồng dữ liệu (`docs/schema.md`):**
   - Tuyệt đối không đưa các biến rò rỉ (`delay_min`, `actual_time`, `status`, `status_detail`) vào tập đặc trưng $X$ khi huấn luyện mô hình.
   - Nhóm Web bắt buộc tích hợp qua hàm `predict_risk()` trong `modeling/predict.py`.

