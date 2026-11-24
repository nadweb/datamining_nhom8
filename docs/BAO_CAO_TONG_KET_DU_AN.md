# BÁO CÁO TỔNG KẾT DỰ ÁN KHAI PHÁ DỮ LIỆU (DATA MINING FINAL REPORT)
## Đề tài: Khai Phá Dữ Liệu & Dự Báo Rủi Ro Trễ, Hủy Chuyến Bay Tại Việt Nam (FlightRisk VN)
> **Môn học:** Khai phá dữ liệu (Data Mining)  
> **Nhóm thực hiện:** Nhóm 8 (8 Thành viên)  
> **Quy mô dữ liệu:** 13,965 chuyến bay ghi nhận thực tế từ 5 sân bay FIDS & Aviationstack  
> **Quy trình áp dụng:** Chuẩn CRISP-DM / Modern KDD Process  
> **Thời gian hoàn thành:** Tháng 11/2026  

---

## MỤC LỤC TỔNG QUAN
1. [TỔNG QUAN DỰ ÁN & MỤC TIÊU NGHIÊN CỨU](#1-tổng-quan-dự-án--mục-tiêu-nghiên-cứu)
2. [KIẾN TRÚC HỆ THỐNG & PHÂN CÔNG TRÁCH NHIỆM](#2-kiến-trúc-hệ-thống--phân-công-trách-nhiệm)
3. [GIAI ĐOẠN 1 & 2: THU THẬP & TÍCH HỢP ĐA NGUỒN (TV1)](#3-giai-đoạn-1--2-thu-thập--tích-hợp-đa-nguồn-tv1)
4. [GIAI ĐOẠN 3: LÀM SẠCH, CHUYỂN ĐỔI & HỢP ĐỒNG DỮ LIỆU (TV2)](#4-giai-đoạn-3-làm-sạch-chuyển-đổi--hợp-đồng-dữ-liệu-tv2)
5. [GIAI ĐOẠN 4: KỸ NGHỆ ĐẶC TRƯNG MIỀN HÀNG KHÔNG (TV3)](#5-giai-đoạn-4-kỹ-nghệ-đặc-trưng-miền-hàng-không-tv3)
6. [GIAI ĐOẠN 5: MÃ HÓA, CHUẨN HÓA & BẢO VỆ CHỐNG RÒ RỈ (TV4)](#6-giai-đoạn-5-mã-hóa-chuẩn-hóa--bảo-vệ-chống-rò-rỉ-tv4)
7. [GIAI ĐOẠN 6: HỌC MÁY CÓ GIÁM SÁT (SUPERVISED LEARNING - TV5)](#7-giai-đoạn-6-học-máy-có-giám-sát-supervised-learning---tv5)
8. [GIAI ĐOẠN 7: KHAI PHÁ TRI THỨC HỌC KHÔNG GIÁM SÁT (UNSUPERVISED LEARNING - TV5)](#8-giai-đoạn-7-khai-phá-tri-thức-học-không-giám-sát-unsupervised-learning---tv5)
9. [GIAI ĐOẠN 8: XÂY DỰNG HỆ THỐNG WEB & PHỤC VỤ DỰ BÁO (TV6, TV7, TV8)](#9-giai-đoạn-8-xây-dựng-hệ-thống-web--phục-vụ-dự-báo-tv6-tv7-tv8)
10. [ĐÁNH GIÁ TRUNG THỰC KHOA HỌC, HẠN CHẾ & BÀI HỌC KINH NGHIỆM](#10-đánh-giá-trung-thực-khoa-học-hạn-chế--bài-học-kinh-nghiệm)
11. [KẾT LUẬN & HƯỚNG PHÁT TRIỂN](#11-kết-luận--hướng-phát-triển)

---

## 1. TỔNG QUAN DỰ ÁN & MỤC TIÊU NGHIÊN CỨU

### 1.1. Bối cảnh thực tiễn ngành hàng không Việt Nam
Vận tải hàng không tại Việt Nam có tốc độ tăng trưởng nhanh bậc nhất khu vực Đông Nam Á, tập trung cao độ trên trục "Tam giác vàng" Hà Nội (HAN) – Đà Nẵng (DAD) – TP. Hồ Chí Minh (SGN). Tuy nhiên, tình trạng quá tải hạ tầng sân bay kết hợp với diễn biến thời tiết nhiệt đới gió mùa phức tạp (mưa dông, mây giông chiều tối, gió giật ven biển) gây ra tỷ lệ chậm chuyến và hủy chuyến đáng kể, ảnh hưởng trực tiếp đến trải nghiệm hành khách và chi phí vận hành hàng tỷ đồng của các hãng bay.

### 1.2. Mục tiêu kỹ thuật & Sản phẩm đầu ra
Dự án **FlightRisk VN** giải quyết bài toán khai phá dữ liệu toàn diện theo vòng đời chuẩn CRISP-DM:
1. **Thu thập và tích hợp tự động:** Kết hợp dữ liệu FIDS thời gian thực từ 5 sân bay chủ lực, đối soát chéo với radar toàn cầu Aviationstack API và dữ liệu khí tượng Open-Meteo.
2. **Khai phá tri thức không giám sát (Unsupervised Mining):**
   - Phân cụm $k$-Means xác định 4 hồ sơ rủi ro vận hành bay thực tế.
   - Giảm chiều PCA 3D trực quan hóa không gian trạng thái chuyến bay.
   - Thuật toán Isolation Forest phát hiện các chuyến bay dị thường/cực đoan.
   - Thuật toán FP-Growth khai phá các luật kết hợp nhân quả giữa yếu tố hãng bay, khung giờ và điều kiện thời tiết.
3. **Mô hình dự báo có giám sát (Supervised Learning):**
   - Phân loại 3 lớp: Đúng giờ (`on_time`), Trễ (`delayed`), Hủy (`cancelled`).
   - Hồi quy số phút trễ liên tục (`delay_min`).
   - Đảm bảo tính minh bạch khoa học: **Zero Data Leakage** và kiểm định bằng **Chronological TimeSeriesSplit**.
4. **Hệ thống ứng dụng hoàn chỉnh (Full-stack Web Platform):**
   - Cung cấp giao diện mô phỏng tương tác, tính toán rủi ro tức thì và bảng điều khiển trực quan hóa tri thức cho người dùng.

---

## 2. KIẾN TRÚC HỆ THỐNG & PHÂN CÔNG TRÁCH NHIỆM

```
[TV1: Thu thập & Tích hợp] ──> data/raw/*.csv
                                   │
[TV2: Làm sạch & Chuyển đổi] ──> flights_cleaned_global.csv (24 cột chuẩn)
                                   │
[TV3: Kỹ nghệ đặc trưng] ──────> flights_features.csv (46 cột đặc trưng)
                                   │
[TV4: Mã hóa & Chuẩn hóa] ─────> encoder.joblib, scaler.joblib, train.csv, test.csv
                                   │
[TV5: Học máy & Khai phá] ─────> models/*.joblib, models/*.json (Supervised & Unsupervised)
                                   │
        ┌──────────────────────────┴──────────────────────────┐
        ▼                                                     ▼
[TV6: Web Backend API]                                 [TV7: Web Frontend UI]
(REST API, Defensive Lineage)                           (HTML5/CSS3 Simulator)
        │                                                     │
        └──────────────────────────┬──────────────────────────┘
                                   ▼
                        [TV8: Web Dashboard & UAT]
                        (Tích hợp K-Means & FP-Growth)
```

| Thành viên | Nhiệm vụ chính | Trách nhiệm sản phẩm | Tệp mã nguồn & Artifact |
| :--- | :--- | :--- | :--- |
| **TV 1** | Data Cleaning & Integration | Thu thập FIDS 5 sân bay + API, khử trùng đa nguồn | `pipeline/collector.py`, `notebooks/01_...` |
| **TV 2** | Transformation & Validation | Xây dựng Data Contract 24 cột, kiểm định tự động | `pipeline/transformation.py`, `check_data.py`, `docs/schema.md` |
| **TV 3** | Feature Engineering | Tạo 22 đặc trưng mới: chu kỳ thời gian, chỉ số dông bão, tải sân bay | `features/features.py`, `notebooks/03_...` |
| **TV 4** | Encoding & Normalization | One-Hot, StandardScaler, Train median imputation, Chặn 5 biến rò rỉ | `modeling/encoding.py`, `tests/test_encoding.py` |
| **TV 5** | Machine Learning & Mining | Cắt chuỗi thời gian, TimeSeriesSplit CV, Random Forest, K-Means, PCA, IsoForest | `modeling/train.py`, `unsupervised.py`, `tests/test_ml.py` |
| **TV 6** | Web Backend & Serving | Thiết kế REST API `/api/predict`, Kiểm tra tương thích phiên bản | `web/app.py`, `modeling/predict.py`, `tests/test_web_backend.py` |
| **TV 7** | Web Frontend Simulator | Giao diện tương tác, 3 kịch bản UAT, bảng màu rủi ro responsive | `web/templates/index.html`, `web/static/style.css` |
| **TV 8** | Dashboard & Integration | Kết nối bất đồng bộ, trực quan hóa K-Means & FP-Growth, UAT End-to-End | `web/static/app.js`, `notebooks/08_...`, `tests/test_web_dashboard.py` |

---

## 3. GIAI ĐOẠN 1 & 2: THU THẬP & TÍCH HỢP ĐA NGUỒN (TV1)

- **FIDS Scraping:** Sử dụng Playwright thu thập trực tiếp từ bảng thông tin chuyến bay điện tử (FIDS) của 5 sân bay trọng điểm: SGN, HAN, DAD, CXR, PQC.
- **Aviationstack API:** Bổ sung chuyến bay bị hủy (`status = cancelled`) để giải quyết mất cân bằng nhãn.
- **Open-Meteo Weather API:** Ghép nối thời tiết theo giờ dựa trên tọa độ sân bay.
- **Khử trùng (Deduplication):** Khóa `flight_date + flight_no + scheduled_time + origin + dest`.
- **Tổng số bản ghi:** Sau khử trùng đạt **13,965 chuyến bay sạch**.

---

## 4. GIAI ĐOẠN 3: LÀM SẠCH, CHUYỂN ĐỔI & HỢP ĐỒNG DỮ LIỆU (TV2)

- **Data Contract 24 cột chuẩn ([`docs/schema.md`](file:///c:/Users/LENOVO/Documents/ScrapeData/docs/schema.md)):** Khóa chính `leg_key`, định danh hành trình, giờ lịch, thời tiết và nhãn.
- **Ngưỡng trễ quốc tế:** Chuyến bay trễ khi $\text{delay\_min} \ge 15$ phút.
- **Nhãn mục tiêu 3 lớp:** `on_time` (0), `delayed` (1), `cancelled` (2).
- **Kiểm định chất lượng tự động (`check_data.py`):** Ràng buộc khóa duy nhất, 7 hãng bay Việt Nam, và chuyến bay hủy bắt buộc `delay_min` là `NaN`.

---

## 5. GIAI ĐOẠN 4: KỸ NGHỆ ĐẶC TRƯNG MIỀN HÀNG KHÔNG (TV3)

- **Chu kỳ thời gian lượng giác:** $\sin/\cos$ của 24 giờ và 7 ngày trong tuần, phản ánh tính tuần hoàn liên tục.
- **Điểm nguy cơ thời tiết (Weather Hazard Score):**
  $$\text{Hazard} = 0.4 \times \frac{\text{precip}}{20} + 0.35 \times \frac{\text{wind}}{50} + 0.25 \times \left(1 - \frac{\text{vis}}{10000}\right)$$
- **Tải sân bay & Biến tương tác:** `airport_load` và `load_x_weather = airport_load \times \text{weather\_hazard\_score}` mô phỏng rủi ro bùng phát khi sân bay đông đúc gặp thời tiết xấu.

---

## 6. GIAI ĐOẠN 5: MÃ HÓA, CHUẨN HÓA & BẢO VỆ CHỐNG RÒ RỈ (TV4)

- **Zero Data Leakage Guard:** Loại bỏ hoàn toàn 5 cột rò rỉ (`delay_min`, `actual_time`, `status`, `status_detail`, `is_delayed_for_analysis`).
- **OneHotEncoder(`handle_unknown="ignore"`):** Tránh dừng ứng dụng khi gặp hãng bay hoặc sân bay mới lạ lúc triển khai.
- **StandardScaler & Train Median Imputation:** Fit tham số và điền khuyết thiếu CHỈ trên tập Train.
- **Không gian đặc trưng:** Mở rộng thành **64 chiều**.

---

## 7. GIAI ĐOẠN 6: HỌC MÁY CÓ GIÁM SÁT (SUPERVISED LEARNING - TV5)

- **Chronological Split:** Cắt chuỗi thời gian tại `2026-10-31` (Train: 11,172 dòng, Test: 2,793 dòng) nhằm loại bỏ Lookahead Bias.
- **Lựa chọn mô hình:** `TimeSeriesSplit` 3-folds trên Train. **Random Forest** được chọn nhờ `CV Macro-F1 = 0.4152` (vượt xa Baseline `0.2764`), `Test ROC-AUC = 0.8086`.
- **Nghịch lý Accuracy:** Baseline đoán mù `on_time` đạt 80.02% nhưng vô dụng (Recall trễ = 0%). Random Forest bắt được **55.4% các chuyến trễ thực tế**.
- **Hồi quy phút trễ:** `HistGradientBoostingRegressor` đạt `MAE = 15.13 phút`.
- **Top Đặc trưng quan trọng:** Giờ bay (17.7%), Hãng hàng không (9.3%), Phân khúc (9.2%), Tải sân bay (8.8%), Ngày trong tuần (7.2%).

---

## 8. GIAI ĐOẠN 7: KHAI PHÁ TRI THỨC HỌC KHÔNG GIÁM SÁT (UNSUPERVISED LEARNING - TV5)

- **K-Means ($k=4$):** Tên cụm tự động suy diễn từ dữ liệu:
  - Cụm 0 (Morning, Clear weather): 41.8%, trễ 23.9% (MEDIUM).
  - Cụm 1 (Afternoon, Windy/Reduced vis): 22.0%, trễ 30.4% (HIGH).
  - Cụm 2 (Evening, Clear weather): 34.5%, trễ 30.3% (HIGH).
  - Cụm 3 (Afternoon, Rain/Storm): 1.8%, trễ 39.0% (HIGH).
- **PCA (3 chiều):** Giải thích **43.66%** tổng phương sai dữ liệu.
- **Isolation Forest:** Phát hiện **392 chuyến bất thường** (2.81%), tỷ lệ trễ vọt lên 45.4%, lệch mạnh nhất ở cờ dông bão ($+3.88\sigma$).
- **FP-Growth:** Luật tiêu biểu `VJ + SGN + Tối muộn -> Trễ` với **Lift = 1.97**.

---

## 9. GIAI ĐOẠN 8: XÂY DỰNG HỆ THỐNG WEB & PHỤC VỤ DỰ BÁO (TV6, TV7, TV8)

- **Full-stack Architecture:** Python Micro-server + REST API 5 endpoints + HTML5/CSS3 Semantic Dashboard.
- **3 Kịch bản UAT Presets:**
  - ☀️ Kịch bản 1: Sáng trời quang (VN • SGN $\rightarrow$ HAN) $\rightarrow$ **LOW RISK**.
  - 🌬️ Kịch bản 2: Chiều gió mạnh (QH • DAD $\rightarrow$ SGN) $\rightarrow$ **MEDIUM RISK**.
  - ⛈️ Kịch bản 3: Tối mưa dông & quá tải (VJ • SGN $\rightarrow$ HAN) $\rightarrow$ **HIGH RISK (Trễ 18 phút, Xác suất trễ 71.9%)**.

---

## 10. ĐÁNH GIÁ TRUNG THỰC KHOA HỌC, HẠN CHẾ & BÀI HỌC KINH NGHIỆM

1. **Lệch phân phối thời gian:** Tháng 10 trễ nhiều hơn tháng 11, cần bảng hiệu chuẩn xác suất.
2. **Lớp Hủy chuyến:** Chỉ có 2 mẫu trong Test nên độ nhiễu thống kê cao.
3. **Hồi quy số phút trễ ($R^2 \approx 0.07$):** Phản ánh việc thiếu dữ liệu nội bộ hãng bay (kỹ thuật, phi hành đoàn, khách trễ). MAE 15 phút mang tính tham khảo khung thời gian.
4. **Phòng thủ phiên bản:** Cơ chế `_check_training_lineage` ngăn ngừa xung đột phiên bản scikit-learn.

---

## 11. KẾT LUẬN & HƯỚNG PHÁT TRIỂN
Dự án đã hoàn thành trọn vẹn 100% mục tiêu, đạt đầy đủ 52 bài test kiểm định tự động, xây dựng nền tảng từ cào dữ liệu đến web serving hoàn chỉnh. Hướng phát triển tiếp theo là bổ sung theo dõi số hiệu đuôi tàu bay (Tail Number Turnaround) và mở rộng phạm vi ra các sân bay Đông Nam Á.
