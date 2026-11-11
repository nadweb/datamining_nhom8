# Báo cáo hoàn thiện Thành viên 2 và Thành viên 3

## Thành viên 2 – Data Transformation & Validation

- `pipeline/transformation.py`: giữ các hàm chuẩn hóa IATA, tính delay, xử lý midnight rollover, gán nhãn và ghép dữ liệu thời gian/thời tiết; tăng độ an toàn khi đầu vào là `None`, `NaN`, chuỗi rỗng hoặc giờ sai định dạng.
- `pipeline/check_data.py`: kiểm định đủ 8 quy tắc Data Contract; thiếu cột được báo thành FAIL thay vì gây `KeyError`; kiểm tra khóa không rỗng/duy nhất và đối chiếu trạng thái hủy theo từ khóa.
- `notebooks/02_member2_data_transformation.ipynb`: notebook có dữ liệu thật nếu tồn tại; nếu không, dùng fixture synthetic để trình diễn các bước kiểm thử. Fixture không được ghi thành dataset chính thức.
- `tests/test_data_validation.py`: không còn phụ thuộc cứng vào `data/flights_sample.csv`; bổ sung kiểm tra thiếu cột.

## Thành viên 3 – Feature Engineering

- `features/features.py`: tạo 22 cột mới từ 24 cột chuẩn, tương ứng 46 cột đầu ra; các nhóm gồm chu kỳ thời gian, thời tiết, giờ cao điểm/tải sân bay, tuyến bay/hãng và tương tác.
- `notebooks/03_member3_feature_engineering.ipynb`: trình bày công thức/nhóm đặc trưng, thống kê, biểu đồ và tương quan với nhãn trễ; chỉ dùng `status_clean` ở phần phân tích sau khi tạo feature.
- `tests/test_features.py`: kiểm tra đủ 22 cột mới và kiểm tra thay đổi 4 cột `delay_min`, `actual_time`, `status`, `status_detail` không làm thay đổi các đặc trưng mới.

## Kiểm thử đã thực hiện

```bash
python -m unittest tests.test_data_validation tests.test_label_rules tests.test_iata_mapping tests.test_features
```

Kết quả: **23 tests passed**.

Hai notebook 02 và 03 đã được thực thi thành công trên dữ liệu synthetic do archive source không chứa thư mục `data/`.

## Giới hạn xác minh

Archive được gửi không có `data/flights_cleaned_global.csv` hoặc `data/flights_sample.csv`. Vì vậy chưa thể xác minh trên dữ liệu thật các mục tiêu như 12,830 dòng, 46 cột và thông báo CLI đạt đủ 8/8 trên dataset của nhóm. Khi đặt dataset đúng vào `data/`, chạy:

```bash
python main.py check-data
python -m unittest tests.test_data_validation tests.test_label_rules tests.test_iata_mapping
python main.py features
python -m unittest tests.test_features
```

Full test suite hiện còn một test của pipeline TV5 cần dataset thật: `tests.test_modeling.TestModelingPipeline.test_predict_risk_response_schema`. Lỗi do thiếu cả hai CSV đầu vào, không phải do test TV2/TV3.
