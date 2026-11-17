# -*- coding: utf-8 -*-
"""Capture actual UAT scenarios from the local web app, if artifacts are ready.

Run from the repository root: python -m tests.capture_tv8_uat
Requires Playwright and a locally installed Chrome/Chromium/Edge.
"""
import json
import threading
from pathlib import Path
from urllib.request import urlopen

from playwright.sync_api import sync_playwright

from tests.test_web_frontend_browser import _browser_path
from web.app import FlightApiHandler, ThreadingHTTPServer


def main():
    executable = _browser_path()
    if not executable:
        raise RuntimeError("Cần cài Chrome, Chromium hoặc Edge để chụp UAT.")
    target = Path(__file__).resolve().parent.parent / "docs" / "uat_tv8"
    target.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), FlightApiHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(url + "/health", timeout=30) as response:
            if not json.load(response).get("ready"):
                raise RuntimeError("Artifact chưa sẵn sàng cho UAT.")
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, executable_path=executable)
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
                page.goto(url + "/", wait_until="networkidle", timeout=30000)
                for preset in ("safe", "windy", "storm"):
                    with page.expect_response("**/api/predict", timeout=30000) as pending:
                        page.locator(f'[data-preset="{preset}"]').click()
                    response = pending.value
                    if response.status != 200:
                        raise RuntimeError(f"API {preset}: HTTP {response.status}")
                    data = response.json()
                    page.wait_for_function(
                        "score => document.querySelector('#res-risk-score').textContent === Math.round(score * 100) + '%'",
                        arg=data["risk_score"], timeout=30000,
                    )
                    page.evaluate("window.scrollTo(0, 0)")
                    page.screenshot(path=str(target / f"{preset}.png"), full_page=True)
                    print(preset, json.dumps({key: data[key] for key in
                          ("prediction", "risk_score", "risk_level", "estimated_delay_min", "probabilities")},
                          ensure_ascii=False))
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
