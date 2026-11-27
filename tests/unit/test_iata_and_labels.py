# -*- coding: utf-8 -*-
"""[UNIT] Modulo 1440 Midnight Rollover, IATA Resolution & Status Label Rules."""
import math

import pandas as pd

from flight_delay.data_engineering.transformation import (
    calculate_delay,
    label_status,
    normalize_flight_no,
    to_iata,
)


def test_modulo_1440_midnight_rollover():
    assert calculate_delay("23:50", "", "00:15", "Departed") == 25.0
    assert calculate_delay("00:10", "", "23:55", "Departed") == -15.0
    assert calculate_delay("08:00", "", "08:22", "Departed 22 minutes late") == 22.0


def test_iata_and_flight_number_normalization():
    assert to_iata("Ho Chi Minh City (SGN)") == "SGN"
    assert to_iata("Ha Noi") == "HAN"
    assert to_iata("DAD") == "DAD"
    s = normalize_flight_no(pd.Series(["VN 0210", "VJ0088"]))
    assert list(s) == ["VN210", "VJ88"]


def test_label_status_policies():
    # Normal on_time & delayed (<15m vs >=15m)
    assert label_status({"status": "Departed", "direction": "DEPARTURES", "delay_min": 14.0, "actual_time": "10:14"})[2] == "on_time"
    assert label_status({"status": "Departed", "direction": "DEPARTURES", "delay_min": 15.0, "actual_time": "10:15"})[2] == "delayed"
    # Cancelled & Diverted never coerced wrongly
    assert label_status({"status": "Cancelled", "direction": "DEPARTURES", "delay_min": math.nan})[2] == "cancelled"
    assert label_status({"status": "Flight Diverted", "direction": "ARRIVALS", "delay_min": math.nan})[2] == "diverted"
    # Implausible early departure (< -90m) -> unknown (excluded)
    assert label_status({"status": "Departed", "direction": "DEPARTURES", "delay_min": -120.0, "actual_time": "06:00"})[2] == "unknown"
