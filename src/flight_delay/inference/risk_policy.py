# -*- coding: utf-8 -*-
"""[STEP 7] Risk Tier Mapping & Pre-Flight Operational Explanation Policy."""
from typing import Dict, List

import pandas as pd

from flight_delay.common.constants import MONITORING_CFG


def classify_risk_tier(disruption_prob: float) -> Dict[str, str]:
    pol = MONITORING_CFG["risk_policy"]
    low_max = float(pol["low_max_exclusive"])
    med_max = float(pol["medium_max_exclusive"])

    if disruption_prob < low_max:
        return {
            "risk_tier": "LOW",
            "risk_level": "Thấp",
            "risk_color": "success",
            "recommendation": "Điều kiện khai thác ổn định; tuân thủ quy trình làm thủ tục tiêu chuẩn.",
        }
    if disruption_prob < med_max:
        return {
            "risk_tier": "MEDIUM",
            "risk_level": "Trung bình",
            "risk_color": "warning",
            "recommendation": "Theo dõi bảng điện tử FIDS & cập nhật thời tiết sân bay trước giờ đóng cửa tàu bay.",
        }
    return {
        "risk_tier": "HIGH",
        "risk_level": "Cao",
        "risk_color": "danger",
        "recommendation": "Cảnh báo rủi ro gián đoạn cao: chuẩn bị phương án điều phối slot, cổng ra tàu bay và hành khách nối chuyến.",
    }


def explain_operational_factors(gold_row: pd.Series) -> List[str]:
    factors: List[str] = []
    if int(gold_row.get("is_storm", 0)) == 1:
        factors.append("Cảnh báo dông bão (WMO >= 95)")
    if int(gold_row.get("is_heavy_rain", 0)) == 1:
        factors.append(f"Mưa lớn ({float(gold_row.get('precipitation', 0)):.1f} mm/h)")
    elif int(gold_row.get("is_rain", 0)) == 1:
        factors.append(f"Có mưa ({float(gold_row.get('precipitation', 0)):.1f} mm/h)")
    if int(gold_row.get("is_high_wind", 0)) == 1:
        factors.append(f"Gió giật mạnh ({float(gold_row.get('wind_gusts_10m', 0)):.1f} km/h)")
    if int(gold_row.get("is_low_vis", 0)) == 1:
        factors.append(f"Tầm nhìn hạn chế ({float(gold_row.get('visibility', 10000)):.0f} m)")
    if int(gold_row.get("is_peak_hour", 0)) == 1:
        factors.append("Khung giờ cao điểm")
    if int(gold_row.get("is_heavy_traffic", 0)) == 1:
        factors.append(f"Mật độ sân bay cao ({int(round(float(gold_row.get('airport_load', 20))))} chuyến/giờ)")
    if float(gold_row.get("historical_airline_delay_rate", 0)) >= 0.30:
        factors.append(f"Tỷ lệ lịch sử hãng cao ({float(gold_row['historical_airline_delay_rate']) * 100:.1f}%)")
    if not factors:
        factors.append("Thời tiết quang đãng & tải khai thác ổn định")
    return factors
