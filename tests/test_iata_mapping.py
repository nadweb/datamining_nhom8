# -*- coding: utf-8 -*-
"""Automated tests for IATA code extraction, normalization, and mapping."""
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from pipeline.transformation import normalize_flight_no, to_iata


class TestIataMapping(unittest.TestCase):
    """Test suite verifying city/airport name to IATA resolution."""

    def test_city_names_mapping(self):
        """Standard Vietnamese city and airport names must resolve to official 3-letter IATA."""
        test_cases = {
            "Ho Chi Minh": "SGN",
            "TP. Hồ Chí Minh": "SGN",
            "Tân Sơn Nhất": "SGN",
            "Saigon": "SGN",
            "Hà Nội": "HAN",
            "Nội Bài": "HAN",
            "Đà Nẵng": "DAD",
            "Cam Ranh": "CXR",
            "Nha Trang": "CXR",
            "Phú Quốc": "PQC",
            "Cần Thơ": "VCA",
            "Đà Lạt": "DLI",
            "Liên Khương": "DLI",
            "Hải Phòng": "HPH",
            "Huế": "HUI",
            "Vinh": "VII",
            "Quy Nhơn": "UIH",
        }
        for city_text, expected_iata in test_cases.items():
            resolved = to_iata(city_text)
            self.assertEqual(resolved, expected_iata, f"Failed resolving '{city_text}' to '{expected_iata}' (got '{resolved}')")

    def test_parenthesized_iata(self):
        """Strings containing IATA code inside parentheses should extract the code."""
        cases = {
            "Tokyo Narita (NRT)": "NRT",
            "Seoul Incheon (ICN)": "ICN",
            "Bangkok Suvarnabhumi (BKK)": "BKK",
            "Singapore Changi (SIN)": "SIN",
        }
        for s, expected in cases.items():
            self.assertEqual(to_iata(s), expected)

    def test_raw_iata_passthrough(self):
        """A bare 3-letter uppercase code should be preserved."""
        for code in ["SGN", "HAN", "DAD", "CXR", "PQC", "HPH", "DLI"]:
            self.assertEqual(to_iata(code), code)

    def test_flight_number_normalization(self):
        """Flight numbers with spacing or leading zeros after airline code must be normalized."""
        s = pd.Series(["VN 1234", "VJ0162", "QH 0021", "BL 0005"])
        normalized = normalize_flight_no(s)
        self.assertEqual(list(normalized), ["VN1234", "VJ162", "QH21", "BL5"])


if __name__ == "__main__":
    unittest.main()

