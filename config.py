# -*- coding: utf-8 -*-
from pathlib import Path

# Thư mục và file hệ thống
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
LOG_DIR = BASE_DIR / "logs"
DEBUG_DIR = BASE_DIR / "data" / "debug_data"
USAGE_FILE = BASE_DIR / "data" / "api_usage.json"

FILE_CLEANED = BASE_DIR / "data" / "flights_cleaned_global.csv"
FILE_RULES = BASE_DIR / "data" / "mined_association_rules.csv"
FILE_WEATHER = BASE_DIR / "data" / "weather_hourly.csv"

# User-Agent chuẩn cho dự án học thuật
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 DataMiningCourse/1.0"
)

# Schema dữ liệu thô (đã loại bỏ terminal, gate, extra do khuyết thiếu >63%-99%)
FIELDS = [
    "source", "airport", "direction", "scraped_at", "flight_date",
    "airline_code", "flight_no", "other_airport", "scheduled_time",
    "estimated_time", "actual_time", "status"
]

# Schema chuẩn đầu ra cho bài toán Khai thác dữ liệu: 1 dòng = 1 chuyến bay (leg), đã gộp thời tiết.
# Mỗi nhóm cột có vai trò riêng khi khai phá:
CLEANED_FIELDS = [
    # --- Định danh ---
    "leg_key", "flight_date", "flight_no", "airline_code",
    # --- Tuyến bay (airport/direction = phía sân bay mà bản ghi được quan sát) ---
    "origin", "destination", "route_type", "airport", "direction",
    # --- Thời gian & bối cảnh ---
    "scheduled_time", "actual_time", "delay_min",
    "time_slot", "day_type", "is_holiday", "airport_load",
    # --- Thời tiết tại `airport`, đúng giờ bay dự kiến (Open-Meteo); mức rời rạc được tính khi khai phá ---
    "precipitation", "wind_gusts_10m", "visibility", "weather_code",
    # --- Nhãn trạng thái ---
    "status",          # chuỗi chuẩn: Departed On-Time / Departed N minutes late / Arrived ... / Delayed N minutes / Cancelled
    "status_detail",   # departed_on_time | departed_late | arrived_on_time | arrived_late | on_time | delayed | cancelled
    "status_clean",    # nhãn đích: on_time | delayed | cancelled
    # --- Nguồn (chứng minh >= 2 nguồn dữ liệu) ---
    "source",
]

# Các hãng hàng không Việt Nam
VN_AIRLINES = {"VN", "VJ", "QH", "BL", "VU", "9G", "0V"}

# Danh sách cấu hình các trang FIDS
SITES = [
    {"name": "guide_sgn", "airport": "SGN", "direction": "arrivals",
     "url": "https://www.hochiminhcityairport.com/flight-status-arrivals-departures/"},
    {"name": "guide_sgn", "airport": "SGN", "direction": "departures",
     "url": "https://www.hochiminhcityairport.com/flight-status-departures/"},
    {"name": "guide_han", "airport": "HAN", "direction": "arrivals",
     "url": "https://www.hanoiairportonline.com/flight-status/"},
    {"name": "guide_han", "airport": "HAN", "direction": "departures",
     "url": "https://www.hanoiairportonline.com/flight-status-departures/"},
    {"name": "guide_dad", "airport": "DAD", "direction": "arrivals",
     "url": "https://www.danangairportonline.com/flight-status-arrivals-departures/"},
    {"name": "guide_dad", "airport": "DAD", "direction": "departures",
     "url": "https://www.danangairportonline.com/flight-status-departures/"},
    {"name": "guide_cxr", "airport": "CXR", "direction": "arrivals",
     "url": "https://www.nhatrangairport.com/flight-status-arrivals-departures/"},
    {"name": "guide_cxr", "airport": "CXR", "direction": "departures",
     "url": "https://www.nhatrangairport.com/flight-status-departures/"},
    {"name": "guide_pqc", "airport": "PQC", "direction": "arrivals",
     "url": "https://www.phuquocairport.com/flight-status-arrivals-departures/"},
    {"name": "guide_pqc", "airport": "PQC", "direction": "departures",
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

# Tên thành phố/sân bay -> mã IATA
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

# Sân bay trong nước (dùng để suy ra route_type = domestic / international)
VN_AIRPORTS = set(AIRPORT_COORD) | set(CITY_IATA.values())

# Số liệu chính thức để đối chiếu (Báo cáo Cục Hàng không)
OFFICIAL_STATS = {
    "ALL": {"delay": 22.3, "cancel": 0.6},
    "VN": {"otp": 78.0},
    "VJ": {"otp": 75.2},
}