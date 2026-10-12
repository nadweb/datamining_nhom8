# -*- coding: utf-8 -*-
from pathlib import Path

# Thư mục và file hệ thống
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
LOG_DIR = BASE_DIR / "logs"
DEBUG_DIR = BASE_DIR / "data" / "debug"
USAGE_FILE = BASE_DIR / "data" / "api_usage.json"

FILE_DEDUP = BASE_DIR / "data" / "flights_dedup.csv"
FILE_CLEANED = BASE_DIR / "data" / "flights_cleaned_global.csv"
FILE_RULES = BASE_DIR / "data" / "mined_association_rules.csv"
FILE_WEATHER = BASE_DIR / "data" / "weather_hourly.csv"

# User-Agent chuẩn cho dự án học thuật
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 DataMiningCourse/1.0")

# Các trường dữ liệu thô
FIELDS = [
    "source", "airport", "direction", "scraped_at", "flight_date",
    "airline_code", "flight_no", "other_airport", "scheduled_time",
    "estimated_time", "actual_time", "terminal", "gate", "status", "extra"
]

# Các hãng hàng không Việt Nam
VN_AIRLINES = {"VN", "VJ", "QH", "BL", "VU", "9G", "0V"}

# Danh sách cấu hình các trang FIDS
SITES = [
    {"name": "noibai_han", "kind": "noibai", "airport": "HAN", "direction": "arrivals",
     "url": "https://noibaiairport.vn/vi/arrivals"},
    {"name": "noibai_han", "kind": "noibai", "airport": "HAN", "direction": "departures",
     "url": "https://noibaiairport.vn/vi/departures"},
    {"name": "guide_sgn", "kind": "guide", "airport": "SGN", "direction": "arrivals",
     "url": "https://www.hochiminhcityairport.com/flight-status-arrivals-departures/"},
    {"name": "guide_sgn", "kind": "guide", "airport": "SGN", "direction": "departures",
     "url": "https://www.hochiminhcityairport.com/flight-status-departures/"},
    {"name": "guide_han", "kind": "guide", "airport": "HAN", "direction": "arrivals",
     "url": "https://www.hanoiairportonline.com/flight-status/"},
    {"name": "guide_han", "kind": "guide", "airport": "HAN", "direction": "departures",
     "url": "https://www.hanoiairportonline.com/flight-status-departures/"},
    {"name": "guide_dad", "kind": "guide", "airport": "DAD", "direction": "arrivals",
     "url": "https://www.danangairportonline.com/flight-status-arrivals-departures/"},
    {"name": "guide_dad", "kind": "guide", "airport": "DAD", "direction": "departures",
     "url": "https://www.danangairportonline.com/flight-status-departures/"},
    {"name": "guide_cxr", "kind": "guide", "airport": "CXR", "direction": "arrivals",
     "url": "https://www.nhatrangairport.com/flight-status-arrivals-departures/"},
    {"name": "guide_cxr", "kind": "guide", "airport": "CXR", "direction": "departures",
     "url": "https://www.nhatrangairport.com/flight-status-departures/"},
    {"name": "guide_pqc", "kind": "guide", "airport": "PQC", "direction": "arrivals",
     "url": "https://www.phuquocairport.com/flight-status-arrivals-departures/"},
    {"name": "guide_pqc", "kind": "guide", "airport": "PQC", "direction": "departures",
     "url": "https://www.phuquocairport.com/flight-status-departures/"},
]

# Ánh xạ tên hãng sang mã IATA
AIRLINE_NAME_CODE = {
    "vietjet": "VJ", "vietnam airlines": "VN", "pacific airlines": "BL",
    "sun phuquoc": "9G", "vietravel": "VU", "vasco": "0V", "bamboo": "QH"
}

# Toạ độ sân bay (để lấy thời tiết Open-Meteo)
AIRPORT_COORD = {
    "SGN": (10.8188, 106.6520), "HAN": (21.2212, 105.8072),
    "DAD": (16.0439, 108.1994), "CXR": (12.2275, 109.1920),
    "PQC": (10.1698, 103.9931),
}

# Tên thành phố/sân bay (đã bỏ dấu, viết thường, bỏ khoảng trắng) -> mã IATA.
# Dùng khi cột other_airport chỉ có tên (vd Nội Bài trả về "TP. Hồ Chí Minh").
CITY_IATA = {
    "hochiminh": "SGN", "saigon": "SGN", "tansonnhat": "SGN",
    "hanoi": "HAN", "noibai": "HAN", "danang": "DAD",
    "nhatrang": "CXR", "camranh": "CXR", "phuquoc": "PQC",
    "haiphong": "HPH", "catbi": "HPH", "vinh": "VII", "hue": "HUI", "phubai": "HUI",
    "quynhon": "UIH", "phucat": "UIH", "cantho": "VCA", "dalat": "DLI", "lienkhuong": "DLI",
    "buonmathuot": "BMV", "banmethuot": "BMV", "vandon": "VDO", "quangninh": "VDO",
    "thanhhoa": "THD", "thoxuan": "THD", "chulai": "VCL", "tamky": "VCL",
    "tuyhoa": "TBB", "pleiku": "PXU", "rachgia": "VKG", "camau": "CAH",
    "condao": "VCS", "dienbien": "DIN", "donghoi": "VDH",
}

# Số liệu chính thức để đối chiếu (đơn vị %, lấy từ báo cáo tháng 9/2026 của Cục Hàng không).
# Điền thêm hãng khác nếu bảng gốc có: "QH": {"otp": ..}, "BL": {"otp": ..}
OFFICIAL_STATS = {
    "ALL": {"delay": 22.3, "cancel": 0.6},
    "VN": {"otp": 78.0},
    "VJ": {"otp": 75.2},
}
