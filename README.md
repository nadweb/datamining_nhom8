# Thu thập và khai phá dữ liệu chuyến bay Việt Nam (chậm / hủy / đúng giờ)

## Nguồn dữ liệu
- noibai_han: noibaiairport.vn (bảng FIDS chính thức của Nội Bài)
- guide_sgn / guide_han / guide_dad / guide_cxr / guide_pqc: nhóm "Airport Guide" (có trạng thái "N minutes late")
- aviationstack_api (tùy chọn, cần AVIATIONSTACK_KEY, gói free hạn chế request)
- Open-Meteo: thời tiết theo giờ (mưa, gió giật, mây, mã thời tiết, tầm nhìn)

## Cài đặt (1 lần)
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    playwright install chromium

## Lệnh chính (chạy từng lệnh, không chép phần chú thích)
    python main.py collect                      thu thập 1 lượt tất cả nguồn
    python main.py collect --only noibai        chỉ một nguồn (nhiều nguồn cách nhau bằng dấu phẩy)
    python main.py loop --every 120             lặp lại mỗi 120 phút
    python main.py stats                        đếm snapshot theo nguồn
    python main.py weather                      tải thời tiết cho các ngày đã thu
    python main.py process                      khử trùng cấp chuyến, ghép thời tiết, khai phá luật (ngưỡng trễ 15 phút)
    python main.py process --compare 30         thêm ngưỡng 30 phút để so sánh
    python main.py process --unit event         đếm cất cánh và hạ cánh là 2 mẫu riêng (mặc định: 1 chuyến = 1 mẫu)
    python main.py process --min-age-hours 3    chỉ giữ chuyến có giờ bay cách lần cào cuối >= 3 giờ
    python main.py api --status cancelled       chuyến hủy từ Aviationstack

## Báo cáo của `process`
- Số mẫu qua từng bước khử trùng, nhãn bị loại (unknown), phân bố nhãn.
- Chẩn đoán other_airport chưa nhận ra mã IATA (bổ sung vào CITY_IATA trong config.py).
- Tỷ lệ trễ / hủy / OTP theo hãng, đặt cạnh số chính thức (OFFICIAL_STATS trong config.py).
- Độ nhất quán nhãn giữa các nguồn khác nhau: % trùng, Cohen's kappa, lệch độ trễ trung vị.
- Luật kết hợp FP-Growth cho lớp delayed và cancelled, xuất ra data/mined_association_rules.csv.

## Lên lịch (Windows)
    schtasks /Create /SC HOURLY /MO 2 /TN "FlightScraper" /TR "C:\Users\LENOVO\Documents\ScrapeData\run_collect.bat" /F
