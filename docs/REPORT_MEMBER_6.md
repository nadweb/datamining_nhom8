# Thành viên 6: Web Backend API & Model Serving

## Phạm vi

- `modeling/predict.py`: kiểm tra đầu vào, nạp và kiểm tra artifact, tạo đặc trưng trước chuyến bay, suy luận và ghép luật.
- `web/app.py`: phục vụ API và giao diện tĩnh bằng HTTP server của Python.
- `tests/test_web_backend.py`: kiểm thử contract và HTTP với model được tạo từ dữ liệu giả lập trong thư mục tạm.
- `tests/benchmark_web_backend.py`: đo độ trễ HTTP với bộ artifact thực tế, không huấn luyện hoặc ghi model.
- `requirements-web.txt`: thêm `holidays` để khớp cách xác định ngày lễ của pipeline và `beautifulsoup4` để `main.py web` nạp được CLI.

## Cách chạy

Tại thư mục gốc của dự án trên Windows PowerShell:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-web.txt
.\venv\Scripts\python.exe main.py web
```

Mặc định giao diện ở `http://127.0.0.1:5000/`; có thể chỉ định `--host` và `--port`. Chỉ chạy cho client tin cậy trên máy cục bộ: server này chưa có xác thực hoặc TLS.

## Endpoint

| Endpoint | Đầu vào / kết quả |
| --- | --- |
| `POST /api/predict` | JSON chuyến bay; gọi `predict_risk()` và trả dự báo theo `docs/schema.md`. |
| `GET /api/stats` | Đối tượng `supervised` lấy từ `models/supervised_metrics.json`. |
| `GET /api/clusters` | `profiles`, `anomalies`, `pca` từ artifact không giám sát. |
| `GET /api/rules` | Tối đa 50 luật từ `data/mined_association_rules.csv`, sắp theo lift giảm dần. |
| `GET /health` | `200`, `ready: true` khi nạp và kiểm tra được artifact; `503`, `ready: false` khi chưa sẵn sàng. |

Ví dụ body của `POST /api/predict`:

```json
{
  "airline_code": "VJ",
  "airport": "SGN",
  "direction": "DEPARTURES",
  "route_type": "domestic",
  "scheduled_time": "19:30",
  "flight_date": "2026-11-06",
  "precipitation": 1.0,
  "wind_gusts_10m": 35,
  "visibility": 4000,
  "weather_code": 95,
  "airport_load": 34,
  "origin": "SGN",
  "destination": "HAN"
}
```

Sáu trường đầu là bắt buộc; các trường thời tiết, tải sân bay, `origin` và `destination` có giá trị mặc định để duy trì giao diện hiện hành. Nên gửi `origin` và `destination` thật: nếu bỏ qua, giá trị mặc định SGN → HAN ảnh hưởng đặc trưng tuyến bay. Thời tiết phải là thông tin có thể biết tại thời điểm dự báo. Các trường hậu chuyến bay như `delay_min`, `actual_time`, `status`, `status_detail` không được chấp nhận làm đầu vào API.

Phản hồi giữ các trường của hợp đồng và các trường frontend đã sử dụng, bao gồm `flight_summary`, `cluster_profile.name`, `cluster_profile.delay_rate_pct` và `matched_rules[].confidence_pct`. `cluster_risk_avg` là tổng tỷ lệ trễ và hủy lịch sử của cụm, tính theo phần trăm. `risk_score` là tổng xác suất trễ và hủy của classifier; đây không phải cam kết về độ chính xác hay xác suất đã hiệu chỉnh.

Các trường hợp chính: `400` cho dữ liệu đầu vào sai/thiếu hoặc trường không hỗ trợ, `413` cho body lớn hơn 16 KiB, `415` nếu không gửi JSON, `503` nếu artifact thiếu, hỏng hoặc có metadata mâu thuẫn. Sau khi thay artifact, khởi động lại server để nạp lại cache.

## Hợp đồng và giới hạn

- Server chỉ `transform` và `predict`, không tự `fit`, huấn luyện lại hoặc ghi artifact. Encoder/scaler, classifier, regressor, K-Means, hồ sơ cụm, metrics và luật là dữ liệu đầu vào.
- Khi metadata thể hiện sự khác biệt về train/test, ngày cắt hoặc phiên bản major.minor của scikit-learn, bộ artifact bị từ chối. Theo quy ước của nhóm, khác biệt bản vá cùng major.minor được cho phép, nhưng vẫn cần kiểm thử tính tương thích. Tên/thứ tự/số đặc trưng của encoder, scaler và model cũng được kiểm tra. Metadata trùng nhau **không chứng minh** các artifact có cùng nguồn gốc; cần Thành viên 4–5 xác nhận bộ bàn giao.
- `is_weekend_flag` và `is_holiday_flag` của request được suy ra từ ngày bay theo cùng quy ước ngày lễ Việt Nam của pipeline. Cột sau chuyến bay không đi vào ma trận model.
- Mọi điều kiện vế trái của một luật phải khớp. `Load=low/mid/high` được tạo bằng xếp hạng tải theo từng sân bay trong tập khai phá, nhưng artifact luật không lưu phân phối/ngưỡng tương ứng. Vì vậy không khẳng định các luật chứa `Load` là khớp cho đến khi có thông tin đó.
- Các model hiện có ghi nhận xác suất chưa chắc đã được hiệu chỉnh; tập Test trong báo cáo chỉ có hai chuyến hủy. Cần nêu giới hạn này khi diễn giải.
- Dữ liệu sạch hiện có 38 dòng có `airport` ngoài 5 sân bay của contract. Đây là vấn đề cần nhóm dữ liệu xác minh, không được backend tự sửa.

## Kiểm thử và tình trạng hiện tại

```powershell
.\venv\Scripts\python.exe -B -m unittest tests.test_web_backend -v
.\venv\Scripts\python.exe -B -m unittest discover tests -v
.\venv\Scripts\python.exe -B -m modeling.predict
.\venv\Scripts\python.exe -B -m tests.benchmark_web_backend --requests 100 --warmup 5
```

Ngày 13/11/2026, 9 bài kiểm thử backend dùng artifact giả lập đồng bộ đều đạt. Toàn bộ suite chạy 61 bài, 60 đạt; `test_modeling.TestModelingPipeline.test_predict_risk_response_schema` không chạy được với artifact thật vì metadata tiền xử lý ghi 10.264/2.566 dòng, mốc 2026-10-30, còn metrics model ghi 11.172/2.793 dòng, mốc 2026-10-31. Cần nhận bộ artifact đồng nhất từ Thành viên 4–5 rồi chạy lại; không tự sửa metadata hoặc ghi đè model để vượt kiểm tra.

**Cập nhật ngày 14/11/2026:** Bộ artifact hiện tại ghi train/test 11.172/2.793 và ngày cắt 2026-10-31 ở cả metadata tiền xử lý và báo cáo model. Encoder/scaler được lưu bằng scikit-learn 1.9.0, model và môi trường kiểm thử dùng 1.9.1. Theo quy ước major.minor của nhóm, self-check dự báo đã trả JSON và toàn bộ **62/62 test đạt**. Tuy nhiên scikit-learn vẫn phát `InconsistentVersionWarning` khi nạp artifact khác bản vá; kết quả cần được nhóm xác nhận và không đồng nghĩa bảo đảm tương thích cho mọi đầu vào.

Đo lại ngày 14/11/2026 bằng API cục bộ với artifact hiện tại: 100 request sau 5 lượt làm nóng, trung vị 43,09 ms, p95 58,00 ms, tối đa 64,54 ms; request đầu gồm nạp model là 643,51 ms. Vì có request vượt 50 ms, mục tiêu **dưới 50 ms cho mỗi request chưa đạt** trong phép đo này.

Phép đo chẩn đoán trên máy hiện tại dùng riêng một tiến trình cục bộ, tạm bỏ qua kiểm tra nguồn gốc **chỉ cho phép đo thời gian**, không sử dụng dự báo làm kết quả: 100 request sau 5 lượt làm nóng có trung vị 47,94 ms, p95 55,07 ms và tối đa 63,85 ms; lần đầu gồm nạp model khoảng 655 ms. Bộ artifact chưa được xác nhận đồng nhất nên đây không phải benchmark nghiệm thu. Mục tiêu mọi request dưới 50 ms **chưa đạt** trong phép đo này; cần đo lại và tối ưu khi có bộ chính thức. Công cụ benchmark bình thường giữ nguyên kiểm tra nguồn gốc và báo lỗi khi bundle không đồng nhất.
