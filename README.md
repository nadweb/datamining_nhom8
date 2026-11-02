# Thu thập và khai phá dữ liệu chuyến bay Việt Nam (đúng giờ / trễ / hủy)

Pipeline thu thập dữ liệu chuyến bay từ nhiều nguồn, lưu **bản gốc** (HTML/JSON), dựng CSV thô từ bản gốc, tích hợp thời tiết rồi khai phá luật kết hợp (FP-Growth) cho nhãn `delayed` và `cancelled`.

## Nguồn dữ liệu
| Nguồn | Cách lấy | Bản gốc được lưu |
|---|---|---|
| `guide_sgn`, `guide_han`, `guide_dad`, `guide_cxr`, `guide_pqc` | Playwright, cào bảng FIDS (chuyến đến và đi) của 5 sân bay | HTML + JSON bảng |
| `aviationstack_api` | Aviationstack API (cần biến môi trường `AVIATIONSTACK_KEY`, gói free giới hạn request) | JSON phản hồi nguyên bản |
| Open-Meteo | Thời tiết theo giờ tại 5 sân bay (Forecast API cho 90 ngày gần nhất, Archive cho phần cũ hơn) | JSON phản hồi nguyên bản |

## Luồng dữ liệu
```
Website ─► debug_data/playwright/html/*.html     (trang gốc)
        └► debug_data/playwright/*.json          (bảng trích từ HTML)
API     ─► debug_data/aviationstack/*.json       (phản hồi gốc)
Open-Meteo ► debug_data/open_meteo/*.json        (phản hồi gốc)
                    │  rebuild_raw_csv()  (chỉ làm phẳng, giữ cột riêng của từng nguồn)
                    ▼
            data/raw/<nguồn>_<YYYYMM>.csv
                    │  tích hợp -> schema chuẩn 12 cột -> khử trùng cấp chuyến + ghép thời tiết
                    ▼
            data/flights_cleaned_global.csv  ──►  data/mined_association_rules.csv
```
JSON/HTML gốc là nguồn sự thật duy nhất. `data/raw/*.csv` luôn được dựng lại từ JSON, không append và không sửa tay (sửa lỗi thì sửa parser rồi chạy lại `build-raw`).

## Cài đặt (1 lần)
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    playwright install chromium

Đặt key Aviationstack (nếu dùng API):

    set AVIATIONSTACK_KEY=your_key_here          (chỉ terminal hiện tại)
    setx AVIATIONSTACK_KEY "your_key_here"       (lâu dài, mở terminal mới)

## Lệnh chính (chạy từng lệnh, không chép phần chú thích)
    python main.py collect                        thu thập 1 lượt từ các trang sân bay
    python main.py collect --only guide_sgn,guide_han   chỉ một số nguồn
    python main.py loop --every 120               lặp lại thu thập mỗi 120 phút
    python main.py api --status cancelled,landed --max-requests 12   thu từ Aviationstack (dùng "all" để không lọc)
    python main.py weather                        tải thời tiết cho khoảng ngày đã thu
    python main.py build-raw                      dựng lại data/raw/*.csv từ JSON gốc
    python main.py html2json                      dựng lại JSON bảng từ HTML đã lưu (thêm --overwrite để ghi đè)
    python main.py stats                          đếm snapshot và chuyến duy nhất theo nguồn
    python main.py process                        khử trùng, ghép thời tiết, gán nhãn, khai phá luật (ngưỡng trễ 15 phút)
    python main.py process --compare 30           thêm ngưỡng 30 phút để so sánh
    python main.py process --unit event           cất cánh và hạ cánh là 2 mẫu riêng (mặc định: 1 chuyến = 1 mẫu)
    python main.py process --min-age-hours 3      chỉ giữ chuyến có giờ bay cách lần cào cuối >= 3 giờ

Thứ tự khuyến nghị: `collect` và `api` (lặp nhiều ngày) -> `weather` -> `process --compare 30`.

## Cấu trúc thư mục
    data/
      debug_data/
        playwright/        JSON bảng (processor đọc)   + html/ (trang gốc)
        aviationstack/     JSON gốc của API
        open_meteo/        JSON gốc thời tiết
      raw/                 CSV thô mỗi nguồn một bộ cột riêng (dựng từ JSON)
      raw_legacy/          bản sao lưu CSV cũ (tạo ở lần dựng lại đầu tiên)
      weather_hourly.csv   cache thời tiết gộp từ JSON
      flights_cleaned_global.csv
      mined_association_rules.csv
      api_usage.json       đếm request Aviationstack theo tháng
    logs/pipeline.log

## Schema
**CSV thô (`data/raw`)**: mỗi nguồn giữ cột của chính JSON/HTML của nó, kèm `scraped_at` và `source_file` (tên file JSON mà dòng đó xuất phát).
- `aviationstack_api_*.csv`: JSON làm phẳng (`departure.iata`, `departure.delay`, `airline.name`, `flight.codeshared.*`...), giữ cả terminal và gate.
- `guide_*_*.csv`: đúng tiêu đề bảng trên trang (`Scheduled`, `Revised`, `Flight`, `From`/`To`, `Status`) cùng `source`, `airport`, `direction`, `table_index`.
- `guide_*_legacy_*.csv`: dữ liệu cũ đã parse sẵn, không còn HTML gốc.

**Schema chuẩn trung gian (12 cột)**, tạo ở bước tích hợp: `source, airport, direction, scraped_at, flight_date, airline_code, flight_no, other_airport, scheduled_time, estimated_time, actual_time, status`. Bước này bỏ terminal, gate, extra (thiếu quá nhiều) và bỏ chuyến codeshare.

**Schema đầu ra (`flights_cleaned_global.csv`, 24 cột, 1 dòng = 1 chuyến)**

| Nhóm | Cột |
|---|---|
| Định danh | `leg_key, flight_date, flight_no, airline_code` |
| Tuyến bay | `origin, destination, route_type, airport, direction` |
| Thời gian và bối cảnh | `scheduled_time, actual_time, delay_min, time_slot, day_type, is_holiday, airport_load` |
| Thời tiết (tại `airport`, đúng giờ bay) | `precipitation, wind_gusts_10m, visibility, weather_code` |
| Nhãn | `status, status_detail, status_clean` |
| Nguồn | `source` |

Nhãn đích `status_clean`: `on_time`, `delayed`, `cancelled`. Các dòng `scheduled`, `diverted`, `unknown` bị loại khỏi khai phá. `delay_min` để trống với chuyến hủy.

## Báo cáo của `process`
- Số mẫu qua từng bước khử trùng, nhãn bị loại, phân bố nhãn.
- % thiếu theo cột; cột vượt 40% được cảnh báo để cân nhắc loại khỏi schema.
- Chẩn đoán `other_airport` chưa nhận ra mã IATA (bổ sung vào `CITY_IATA` trong `config.py`).
- Tỷ lệ trễ / hủy / OTP theo hãng, đặt cạnh số chính thức (`OFFICIAL_STATS` trong `config.py`).
- Độ nhất quán nhãn giữa các nguồn: % trùng, Cohen's kappa, lệch độ trễ trung vị.
- Luật kết hợp FP-Growth cho lớp `delayed` và `cancelled`, xuất ra `data/mined_association_rules.csv`.

## Lưu ý khi diễn giải
- `api --status cancelled,landed` lấy mẫu lệch về chuyến hủy, nên tỷ lệ hủy/trễ trong mẫu không so trực tiếp với số liệu chính thức được. Luật và lift vẫn có giá trị trên mẫu.
- Dữ liệu thu trước khi lưu JSON/HTML chỉ còn CSV đã parse (nằm trong `raw_legacy/`), không khôi phục được bản gốc.
- File HTML khá nặng; xóa bớt HTML cũ khi đã chắc JSON đúng.

## Lên lịch (Windows)
    schtasks /Create /SC HOURLY /MO 2 /TN "FlightScraper" /TR "C:\Users\LENOVO\Documents\ScrapeData\run_collect.bat" /F

`run_collect.bat` cần kích hoạt venv rồi chạy `python main.py collect`.
