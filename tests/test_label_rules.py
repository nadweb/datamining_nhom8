# -*- coding: utf-8 -*-
"""Automated tests for labeling rules, delay calculation, and midnight rollover."""
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from pipeline.transformation import calculate_delay, label_status, parse_time_min


class TestLabelRulesAndDelays(unittest.TestCase):
    """Test suite covering delay calculations, midnight cases, and status labeling."""

    def test_midnight_rollover_positive_delay(self):
        """Flight scheduled at 23:50 and departing at 00:20 rolls past midnight (+30 min delay)."""
        diff = calculate_delay(sched="23:50", est=None, actual="00:20", status_str="")
        self.assertEqual(diff, 30.0)

    def test_midnight_rollover_early_departure(self):
        """Flight scheduled at 00:15 and departing at 23:55 departs early (-20 min)."""
        diff = calculate_delay(sched="00:15", est=None, actual="23:55", status_str="")
        self.assertEqual(diff, -20.0)

    def test_text_regex_delays(self):
        """Status string with explicit late/early text takes precedence."""
        # '35 minutes late'
        d1 = calculate_delay(sched="10:00", est=None, actual="10:00", status_str="Arrived 35 minutes late")
        self.assertEqual(d1, 35.0)

        # 'delayed 45 minutes'
        d2 = calculate_delay(sched="14:00", est=None, actual="14:00", status_str="Delayed 45 minutes")
        self.assertEqual(d2, 45.0)

        # '10 minutes early'
        d3 = calculate_delay(sched="08:00", est=None, actual="08:00", status_str="Landed 10 minutes early")
        self.assertEqual(d3, -10.0)

    def test_label_cancelled_flights(self):
        """Flights with cancelled keyword must strictly map to ('Cancelled', 'cancelled', 'cancelled')."""
        cases = [
            {"status": "Cancelled", "delay_min": np.nan, "direction": "DEPARTURES"},
            {"status": "Đã hủy", "delay_min": np.nan, "direction": "ARRIVALS"},
            {"status": "Flight cancelled by airline", "delay_min": 100.0, "direction": "DEPARTURES"},
        ]
        for c in cases:
            st, detail, clean = label_status(c)
            self.assertEqual(st, "Cancelled")
            self.assertEqual(detail, "cancelled")
            self.assertEqual(clean, "cancelled")

    def test_label_departed_on_time_and_late(self):
        """Test threshold boundary between on_time and delayed."""
        # 14 min late at threshold 15 is on_time
        row_on_time = {
            "status": "Departed", "direction": "DEPARTURES", "actual_time": "10:14", "delay_min": 14.0
        }
        st, detail, clean = label_status(row_on_time, delay_threshold=15)
        self.assertEqual(clean, "on_time")
        self.assertEqual(detail, "departed_on_time")

        # 15 min late at threshold 15 is delayed
        row_delayed = {
            "status": "Departed", "direction": "DEPARTURES", "actual_time": "10:15", "delay_min": 15.0
        }
        st, detail, clean = label_status(row_delayed, delay_threshold=15)
        self.assertEqual(clean, "delayed")
        self.assertEqual(detail, "departed_late")
        self.assertEqual(st, "Departed 15 minutes late")

    def test_delay_threshold_compare(self):
        """A 20-min delay is delayed under threshold 15, but on_time under threshold 30."""
        row = {"status": "Departed", "direction": "DEPARTURES", "actual_time": "10:20", "delay_min": 20.0}
        _, _, clean_15 = label_status(row, delay_threshold=15)
        _, _, clean_30 = label_status(row, delay_threshold=30)
        self.assertEqual(clean_15, "delayed")
        self.assertEqual(clean_30, "on_time")


if __name__ == "__main__":
    unittest.main()

