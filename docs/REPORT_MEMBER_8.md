# BÁO CÁO THÀNH VIÊN 8 — PHẦN 6C

## 6.3. Dashboard phân tích và kiểm thử chấp nhận

**Phạm vi:** `web/static/app.js`, kiểm thử `tests/test_web_dashboard.py` và notebook `notebooks/08_member8_web_dashboard.ipynb`. Giao diện và các nút kịch bản do Thành viên 7 cung cấp; máy chủ và hợp đồng dự báo do Thành viên 6 cung cấp. Kết quả thực tế phụ thuộc bộ dữ liệu và model đang được nạp.

### 6.3.1. Luồng client và API

```text
Giao diện TV7 → kiểm tra đầu vào → POST /api/predict → predict_risk()
              ← JSON theo docs/schema.md ← mô hình + cụm + luật
              → cập nhật màu, xác suất, cụm và các luật phù hợp

Tab mô hình  → GET /api/stats    → chỉ số phân loại, hồi quy, 10 đặc trưng
Tab phân cụm → GET /api/clusters → hồ sơ cụm và thông số Isolation Forest
Tab tập luật → GET /api/rules    → 15 luật có Lift cao nhất do API trả về
```

- `risk_score` là xác suất trễ cộng xác suất hủy; lớp dự báo là lớp có xác suất riêng lớn nhất. Vì vậy, một chuyến có thể được phân loại `on_time` nhưng vẫn có `risk_level` HIGH khi tổng xác suất trễ/hủy từ 45% trở lên.
- Dashboard dùng thông tin mô hình được chọn, kích thước tập test và phương pháp tính độ quan trọng từ API; không tự đặt cứng các giá trị này. Bảng luật thể hiện quan hệ trong dữ liệu, không chứng minh quan hệ nhân quả.
- Khi API lỗi, quá thời gian hoặc dữ liệu dự báo không hợp lệ, giao diện xóa kết quả cũ và thông báo lỗi. Các tab phân tích có nút thử lại; không tạo dự báo, cụm hoặc số liệu thay thế.
- Khi người dùng thay đổi đầu vào hoặc đảo chiều, kết quả cũ bị xóa. Các yêu cầu chồng lấn được quản lý để phản hồi cũ không ghi đè kết quả mới.

### 6.3.2. Điều kiện UAT và kết quả ghi nhận

Đã chạy ngày **15/11/2026** trên máy cục bộ bằng Chrome không giao diện, máy chủ Python của dự án và bộ artifact đang có. `/health` trả về sẵn sàng, ba yêu cầu dự báo trả HTTP 200. Ảnh lưu trong `docs/uat_tv8/`; số liệu trong bảng là kết quả lần chạy này, không phải cam kết cho mọi bộ model.

| Kịch bản | Đầu vào chính | Phân loại | Rủi ro trễ/hủy | Trễ ước tính | Ảnh |
| --- | --- | --- | --- | --- | --- |
| Trời quang | VN, SGN → HAN, 08:00, mưa 0 mm, gió 10 km/h, tải 15 | `on_time`, 51,14% đúng giờ | **HIGH, 48,86%** | 6 phút | [safe.png](uat_tv8/safe.png) |
| Chiều gió | QH, DAD → SGN, 14:45, mưa 1 mm, gió 32 km/h, tải 26 | `on_time`, 51,48% đúng giờ | **HIGH, 48,52%** | 18 phút | [windy.png](uat_tv8/windy.png) |
| Mưa dông | VJ, SGN → HAN, 19:30, mưa 15 mm, gió 45 km/h, WMO 95, tải 35 | `delayed`, 74,54% trễ | **HIGH, 74,54%** | 26 phút | [storm.png](uat_tv8/storm.png) |

Các kịch bản đều dùng ngày 06/11/2026; tầm nhìn tương ứng là 10.000 m, 6.000 m và 2.500 m. Nút mưa dông dùng tải 35 để khớp hướng dẫn TV8.

**Đối chiếu yêu cầu:** kịch bản trời quang **chưa đạt** kết quả mong đợi trong `docs/huong_dan_thanh_vien.md` (LOW, xác suất đúng giờ trên 75%). Kịch bản mưa dông trả HIGH và phân loại trễ; giá trị đo được 74,54% và 26 phút khác với các con số xấp xỉ 79% và 35 phút trong hướng dẫn. Cần leader và Thành viên 5 xác nhận bộ artifact, cách diễn giải và tiêu chí nghiệm thu trước khi kết luận hoàn thành UAT. Không sửa đầu ra để ép khớp kịch bản.

### 6.3.3. Kiểm thử tích hợp

```powershell
.\venv\Scripts\python.exe -B -m unittest tests.test_web_dashboard tests.test_web_frontend_browser -v
```

Bộ kiểm thử trình duyệt kiểm tra: mỗi preset chỉ gửi một request; dữ liệu đầu vào bằng 0; API thất bại và thử lại; JSON không hợp lệ; kết quả cũ và phản hồi đến muộn; văn bản từ API; số liệu các tab so với API thật; và kết nối ba kịch bản với model thật. Các bài dùng phản hồi mô phỏng được phân biệt với bài gọi API thật trong mã kiểm thử. Các kiểm thử trình duyệt cần Playwright và Chrome/Chromium/Edge; nếu thiếu sẽ được đánh dấu bỏ qua.

Để chụp lại ảnh với bộ artifact hiện hành:

```powershell
.\venv\Scripts\python.exe -B -m tests.capture_tv8_uat
```

Lệnh trên ghi lại ba ảnh trong `docs/uat_tv8/` và in kết quả API; chỉ chạy khi `/health` sẵn sàng. Chạy ứng dụng cho demo bằng `.\venv\Scripts\python.exe main.py web`, sau đó mở `http://127.0.0.1:5000/`.

### 6.3.4. Giới hạn và nội dung cần xác nhận

- Encoder và scaler hiện được lưu bằng scikit-learn **1.9.0**, môi trường nạp **1.9.1**; khi nạp có cảnh báo `InconsistentVersionWarning`. Cần nhóm xác nhận hoặc cung cấp bộ artifact và môi trường tương thích. Cảnh báo này chưa chứng minh nguyên nhân của sai lệch UAT.
- Metadata thống kê hiện ghi tập test chỉ có 2 chuyến hủy và có lưu ý về hiệu chỉnh xác suất; không xem xác suất hiển thị là độ chắc chắn đã được hiệu chỉnh.
- Chưa có bằng chứng trong biên bản này để nghiệm thu độ trễ dưới 50 ms; cần chạy benchmark theo báo cáo TV6 với bộ artifact và môi trường dùng khi nghiệm thu.
- Ảnh thực tế có thể đưa vào slide; việc chọn nội dung slide, ký xác nhận UAT và tạo Pull Request cần được nhóm thực hiện theo quy trình của dự án.
