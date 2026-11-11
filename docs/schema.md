# Data Contract: Vietnam Flight Delay & Cancellation Mining

## 1. Mục tiêu và Phạm vi
Tài liệu này xác lập bản giao ước dữ liệu (Data Contract) giữa 3 phân nhóm:
- **Nhóm Dữ liệu (Parts 1-2):** Đảm bảo tính toàn vẹn và chất lượng của `data/flights_cleaned_global.csv`.
- **Nhóm Mô hình (Parts 3-5):** Đảm bảo feature engineering, encoding và mô hình hóa không bị rò rỉ dữ liệu (data leakage).
- **Nhóm Web (Part 6):** Sử dụng đúng giao diện suy luận `predict_risk()` và không gọi trực tiếp các cột bị cấm.

---

## 2. Schema chuẩn 24 cột (`flights_cleaned_global.csv`)
Mỗi dòng đại diện cho đúng **1 chặng bay duy nhất (flight leg)** được theo dõi tại một thời điểm hoặc một đầu sân bay.

| STT | Tên cột | Kiểu dữ liệu | Mô tả | Giá trị hợp lệ / Ví dụ | Ràng buộc |
|:---:|---|:---:|---|---|---|
| 1 | `leg_key` | String | Khóa duy nhất cấp chuyến bay | `VN1234\|2026-10-28\|SGN>HAN` | **Primary Key**: Duy nhất, không rỗng |
| 2 | `flight_date` | Date (YYYY-MM-DD) | Ngày bay theo lịch | `2026-10-28` | ISO 8601, không rỗng |
| 3 | `flight_no` | String | Số hiệu chuyến bay đã chuẩn hóa | `VN1234`, `VJ162` | 2 ký tự mã hãng + số |
| 4 | `airline_code` | String | Mã IATA của hãng hàng không | `VN`, `VJ`, `QH`, `BL`, `VU`, `9G`, `0V` | Thuộc danh sách hãng VN |
| 5 | `origin` | String | Mã sân bay đi (IATA 3 chữ cái) | `SGN`, `HAN`, `DAD`, `CXR`, `PQC`,... | IATA 3 ký tự |
| 6 | `destination` | String | Mã sân bay đến (IATA 3 chữ cái) | `HAN`, `SGN`, `PQC`, `ICN`, `NRT`,... | IATA 3 ký tự |
| 7 | `route_type` | String | Loại đường bay | `domestic` hoặc `international` | Không được để trống |
| 8 | `airport` | String | Sân bay đặt trạm quan sát | `SGN`, `HAN`, `DAD`, `CXR`, `PQC` | 1 trong 5 sân bay FIDS |
| 9 | `direction` | String | Hướng chuyến bay tại trạm quan sát | `DEPARTURES` hoặc `ARRIVALS` | In hoa |
| 10 | `scheduled_time` | String (HH:MM) | Giờ bay dự kiến theo lịch ban đầu | `08:30`, `23:45` | Định dạng 24h HH:MM |
| 11 | `actual_time` | String (HH:MM) | Giờ thực tế cất cánh hoặc hạ cánh | `08:45`, `00:10` | *Cột rò rỉ thông tin!* |
| 12 | `delay_min` | Float | Số phút trễ so với giờ lịch | `15.0`, `-5.0`, `NaN` | *Cột rò rỉ! Phải là NaN khi hủy* |
| 13 | `time_slot` | Categorical | Khung giờ bay trong ngày | `00h-06h`, `06h-12h`, `12h-18h`, `18h-24h` | Phân đoạn 6 tiếng |
| 14 | `day_type` | Categorical | Loại ngày trong tuần | `weekday` hoặc `weekend` | Thứ 2-6 / Thứ 7-CN |
| 15 | `is_holiday` | Boolean | Ngày lễ quốc gia Việt Nam | `True` hoặc `False` | Theo thư viện `holidays` (VN) |
| 16 | `airport_load` | Integer | Số chuyến bay trong cùng khung 1 giờ tại sân bay | `15`, `42` | $\ge 0$ |
| 17 | `precipitation` | Float | Lượng mưa tại sân bay lúc bay (mm) | `0.0`, `3.2` | $\ge 0$ (từ Open-Meteo) |
| 18 | `wind_gusts_10m` | Float | Gió giật độ cao 10m (km/h) | `12.5`, `45.0` | $\ge 0$ |
| 19 | `visibility` | Float | Tầm nhìn xa quan sát được (mét) | `10000.0`, `4500.0` | $\ge 0$ |
| 20 | `weather_code` | Integer | Mã hiện tượng thời tiết WMO | `0` (quang), `61` (mưa), `95` (dông) | WMO Code |
| 21 | `status` | String | Trạng thái hiển thị gốc đã chuẩn hóa | `Departed On-Time`, `Cancelled` | *Cột rò rỉ thông tin!* |
| 22 | `status_detail` | Categorical | Trạng thái phân loại chi tiết | `departed_on_time`, `departed_late`, `cancelled` | *Cột rò rỉ thông tin!* |
| 23 | `status_clean` | Categorical | **Nhãn mục tiêu (Target variable)** | `on_time`, `delayed`, `cancelled` | 3 lớp chuẩn |
| 24 | `source` | String | Tên nguồn dữ liệu thu thập | `guide_sgn`, `aviationstack_api` | Chứng minh đa nguồn |

---

## 3. Quy định nghiêm ngặt về Rò rỉ Thông tin (Data Leakage)

> [!CAUTION]
> **Tuyệt đối không được đưa các cột sau vào tập biến độc lập ($X$) khi huấn luyện mô hình dự báo:**
> 1. `delay_min` (Số phút trễ thực tế)
> 2. `actual_time` (Giờ thực tế cất/hạ cánh)
> 3. `status` (Trạng thái chi tiết sau khi bay)
> 4. `status_detail` (Chi tiết nhãn sau khi bay)
>
> Bất kỳ mô hình nào sử dụng 4 cột này đều coi như **không hợp lệ (Invalid)** vì trong thực tế, tại thời điểm hành khách hoặc điều hành cần dự báo rủi ro (trước giờ bay), các giá trị này **chưa hề xảy ra**.

### Các đặc trưng đầu vào hợp lệ ($X$ Features):
- **Thông tin chuyến bay:** `flight_no` (prefix/number), `airline_code`, `origin`, `destination`, `route_type`, `airport`, `direction`.
- **Thời gian & Lịch trình:** `scheduled_time`, `flight_date`, `time_slot`, `day_type`, `is_holiday`, cyclical hour/day sine & cosine.
- **Tải sân bay & Hoạt động:** `airport_load`, `load_level`.
- **Thời tiết dự báo:** `precipitation`, `wind_gusts_10m`, `visibility`, `weather_code`, `rain_level`, `wind_level`, `vis_level`, `is_storm`.

---

## 4. Quy tắc chia tập dữ liệu theo Thời gian (Time-based Split Rule)

> [!IMPORTANT]
> **KHÔNG SỬ DỤNG `train_test_split(shuffle=True)` ngẫu nhiên.**
> Dữ liệu chuyến bay có tính chuỗi thời gian (time-series). Nếu xáo trộn ngẫu nhiên, mô hình sẽ học dữ liệu của "ngày tương lai" để dự đoán "ngày quá khứ" (Lookahead Bias).

### Quy định phân chia:
1. Sắp xếp toàn bộ dữ liệu theo thứ tự thời gian tăng dần: `flight_date`, `scheduled_time`.
2. Điểm cắt thời gian (Cutoff Date): **80% dữ liệu thời gian đầu làm Train**, **20% dữ liệu thời gian sau làm Test**.
3. Các transformer (như `StandardScaler`, `OneHotEncoder`, Imputer) **chỉ được `fit()` trên tập Train**, sau đó chỉ dùng `transform()` trên tập Test.

---

## 5. Giao diện chuẩn cho Nhóm Web: Hàm `predict_risk()`

Mọi thành viên nhóm Web và API suy luận phải gọi thông qua hàm chuẩn duy nhất trong `modeling/predict.py`:

### Chữ ký hàm (Signature):
```python
def predict_risk(
    airline_code: str,
    airport: str,
    direction: str,
    route_type: str,
    scheduled_time: str,     # "HH:MM"
    flight_date: str,        # "YYYY-MM-DD"
    precipitation: float = 0.0,
    wind_gusts_10m: float = 10.0,
    visibility: float = 10000.0,
    weather_code: int = 0,
    airport_load: int = 20
) -> dict:
```

### Cấu trúc kết quả trả về (Return Schema):
```json
{
  "prediction": "on_time",           // "on_time" | "delayed" | "cancelled"
  "risk_score": 0.28,                // Xác suất trễ hoặc hủy (0.00 -> 1.00)
  "risk_level": "LOW",               // "LOW" | "MEDIUM" | "HIGH"
  "estimated_delay_min": 0,          // Dự đoán số phút trễ (từ mô hình hồi quy)
  "probabilities": {
    "on_time": 0.72,
    "delayed": 0.26,
    "cancelled": 0.02
  },
  "cluster_profile": {
    "cluster_id": 1,
    "cluster_name": "Standard Morning Domestic Flights",
    "cluster_risk_avg": "18.5%"
  },
  "matched_rules": [
    {
      "antecedent": "Airport=SGN & Slot=18h-24h & Load=high",
      "consequent": "Status=delayed",
      "confidence": 42.5,
      "lift": 1.62
    }
  ]
}
```
Mọi thay đổi đối với contract này cần thông báo và được sự đồng ý của cả 3 nhóm.

