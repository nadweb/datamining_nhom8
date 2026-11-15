# -*- coding: utf-8 -*-
"""Kiểm thử tự động cho Thành viên 7 (Part 6B: Web Frontend UI & Interactive Simulator).

Chạy độc lập:
    python -m unittest tests.test_web_frontend -v
"""
import re
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bs4 import BeautifulSoup
from config import AIRPORT_COORD, VN_AIRLINES


class TestWebFrontendUI(unittest.TestCase):
    """Kiểm định giao diện HTML/CSS của Thành viên 7 tuân thủ đúng hợp đồng với app.js (TV8) và schema.md."""

    @classmethod
    def setUpClass(cls):
        cls.html_path = ROOT_DIR / "web" / "templates" / "index.html"
        cls.css_path = ROOT_DIR / "web" / "static" / "style.css"
        cls.html_text = cls.html_path.read_text(encoding="utf-8")
        cls.css_text = cls.css_path.read_text(encoding="utf-8")
        cls.soup = BeautifulSoup(cls.html_text, "html.parser")

    def test_files_exist_and_non_empty(self):
        """Hai tệp index.html và style.css phải tồn tại và có nội dung."""
        self.assertTrue(self.html_path.exists())
        self.assertTrue(self.css_path.exists())
        self.assertGreater(len(self.html_text), 1000)
        self.assertGreater(len(self.css_text), 1000)

    def test_navigation_tabs_contract(self):
        """Phải có đủ 4 nút chuyển tab và 4 section nội dung khớp với app.js."""
        expected_tabs = ["simulator-tab", "models-tab", "clusters-tab", "rules-tab"]
        nav_btns = self.soup.select(".nav-btn[data-tab]")
        btn_targets = [b.get("data-tab") for b in nav_btns]
        self.assertEqual(btn_targets, expected_tabs)

        for tab_id in expected_tabs:
            sec = self.soup.find("section", id=tab_id)
            self.assertIsNotNone(sec, f"Thiếu section #{tab_id}")
            self.assertIn("tab-content", sec.get("class", []))

    def test_simulator_form_inputs_and_airlines(self):
        """Form dự báo phải có đủ các trường đầu vào và đủ 7 hãng hàng không Việt Nam."""
        form = self.soup.find("form", id="predict-form")
        self.assertIsNotNone(form, "Thiếu form #predict-form")

        airline_select = self.soup.find("select", id="airline")
        self.assertIsNotNone(airline_select)
        airline_values = {opt.get("value") for opt in airline_select.find_all("option")}
        self.assertEqual(airline_values, VN_AIRLINES)

        airport_select = self.soup.find("select", id="airport")
        self.assertIsNotNone(airport_select)
        airport_values = {opt.get("value") for opt in airport_select.find_all("option")}
        self.assertEqual(airport_values, set(AIRPORT_COORD.keys()))

    def test_sliders_and_live_value_readouts(self):
        """Bốn thanh trượt (precip, wind, vis, load) phải có đúng min/max và thẻ hiển thị giá trị động."""
        expected_sliders = {
            "precip": ("0", "25"),
            "wind": ("5", "80"),
            "vis": ("500", "10000"),
            "load": ("5", "50"),
        }
        for sid, (min_v, max_v) in expected_sliders.items():
            slider = self.soup.find("input", id=sid)
            self.assertIsNotNone(slider, f"Thiếu thanh trượt #{sid}")
            self.assertEqual(slider.get("type"), "range")
            self.assertEqual(slider.get("min"), min_v)
            self.assertEqual(slider.get("max"), max_v)

            val_span = self.soup.find(id=f"{sid}-val")
            self.assertIsNotNone(val_span, f"Thiếu phần tử hiển thị giá trị #{sid}-val")

    def test_result_card_and_probability_bars_contract(self):
        """Khung kết quả phải chứa đủ các ID cho Risk Badge, 3 thanh xác suất, K-Means và FP-Growth."""
        result_ids = [
            "predict-result", "res-risk-level", "res-prediction-text", "res-risk-score",
            "prob-ontime", "prob-ontime-text",
            "prob-delayed", "prob-delayed-text",
            "prob-cancelled", "prob-cancelled-text",
            "res-cluster-name", "res-cluster-rate", "res-rules-list",
        ]
        for rid in result_ids:
            self.assertIsNotNone(self.soup.find(id=rid), f"Thiếu phần tử kết quả #{rid} trong index.html")

    def test_zero_leakage_in_ui_form(self):
        """Giao diện tuyệt đối không được có ô nhập cho các biến rò rỉ (delay_min, actual_time, status)."""
        form = self.soup.find("form", id="predict-form")
        all_ids_and_names = {
            value
            for el in form.find_all(["input", "select", "textarea"])
            for value in (el.get("id"), el.get("name"))
            if value
        }
        forbidden = {"delay_min", "actual_time", "status", "status_detail"}
        self.assertTrue(all_ids_and_names.isdisjoint(forbidden))

    def test_css_risk_classes_and_responsive_media_queries(self):
        """Tệp style.css phải định nghĩa đủ 3 trạng thái màu rủi ro, thanh xác suất và responsive @media."""
        required_selectors = [
            ".result-header.risk-low",
            ".result-header.risk-medium",
            ".result-header.risk-high",
            ".bar-fill.on-time",
            ".bar-fill.delayed",
            ".bar-fill.cancelled",
            ".data-table",
            ".rule-item",
        ]
        for sel in required_selectors:
            self.assertIn(sel, self.css_text, f"Thiếu selector '{sel}' trong style.css")
        self.assertRegex(self.css_text, r"@media\s*\(max-width:")

    def test_dom_ids_are_unique_and_result_starts_neutral(self):
        ids = [element.get("id") for element in self.soup.find_all(id=True)]
        self.assertEqual(len(ids), len(set(ids)), "ID trùng có thể làm app.js cập nhật sai phần tử")
        header = self.soup.select_one("#predict-result .result-header")
        self.assertIsNotNone(header)
        self.assertFalse({"risk-low", "risk-medium", "risk-high"} & set(header.get("class", [])))
        self.assertEqual(self.soup.find(id="res-risk-score").get_text(strip=True), "--")

    def test_offline_preview_is_explicitly_labeled_as_simulation(self):
        preview = (ROOT_DIR / "docs" / "XEM_NGAY_GIAO_DIEN_TV7.html").read_text(encoding="utf-8")
        self.assertIn("BẢN XEM TRƯỚC NGOẠI TUYẾN", preview)
        self.assertIn("không lấy từ model hoặc API", preview)


if __name__ == "__main__":
    unittest.main()
