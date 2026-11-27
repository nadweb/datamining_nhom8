# -*- coding: utf-8 -*-
"""[E2E BROWSER] Live Uvicorn + Headless Chromium Interactive Dashboard Verification."""
import importlib.util
import socket
import threading
import time

import pytest
import uvicorn
from tests.conftest import find_chromium_binary

from flight_delay.api.app import app

HAS_BROWSER = bool(importlib.util.find_spec("playwright") and find_chromium_binary())


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.skipif(not HAS_BROWSER, reason="Playwright and Chromium/Edge required for E2E browser checks.")
def test_live_dashboard_presets_and_tabs_render_without_js_errors():
    from playwright.sync_api import sync_playwright

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait briefly for uvicorn socket startup
    deadline = time.time() + 10.0
    while not server.started and time.time() < deadline:
        time.sleep(0.1)

    url = f"http://127.0.0.1:{port}"
    page_errors = []
    console_errors = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, executable_path=find_chromium_binary())
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            page.on("pageerror", lambda err: page_errors.append(str(err)))
            page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)

            page.goto(url + "/", wait_until="networkidle")
            page.wait_for_function("document.querySelector('#prob-ontime-text').textContent !== '--'")

            # Click storm preset -> verifies 3-class ICAO bars populate
            with page.expect_response("**/api/predict"):
                page.locator('[data-preset="storm"]').click()
            assert page.locator("#prob-delayed-text").inner_text().endswith("%")
            assert page.locator("#prob-cancelled-text").inner_text().endswith("%")

            # Navigate through analytical tabs
            for tab in ("eda-tab", "models-tab", "clusters-tab", "rules-tab", "drift-tab"):
                page.locator(f'[data-tab="{tab}"]').click()
                page.wait_for_timeout(450)

            assert page.locator("#metrics-content table").count() >= 1
            assert page.locator("#rules-content table").count() >= 1
            assert page.locator("#drift-content table").count() >= 1

            # Verify presentation opens cleanly
            page.goto(url + "/presentation", wait_until="networkidle")
            assert "FlightRisk VN 2.0" in page.title()
            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=5)

    assert page_errors == [], f"JS runtime errors: {page_errors}"
    assert console_errors == [], f"Console errors: {console_errors}"
