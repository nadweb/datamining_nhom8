# -*- coding: utf-8 -*-
"""Browser checks for TV7 UI and the existing TV8 client.

Optional: install Playwright and have Chrome/Chromium available to run these
checks. They use mocked HTTP responses; no real model artifacts are modified.
"""
import importlib.util
import shutil
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from web.app import FlightApiHandler, ThreadingHTTPServer


def _browser_path():
    candidates = [
        shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("chromium-browser"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    return next((str(Path(path)) for path in candidates if path and Path(path).is_file()), None)


class BrowserFixture:
    @classmethod
    def setUpClass(cls):
        from playwright.sync_api import sync_playwright
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FlightApiHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(headless=True, executable_path=_browser_path())
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self):
        self.context = self.browser.new_context(viewport={"width": 1440, "height": 900})
        self.page = self.context.new_page()
        self.payloads = []
        self.page_errors = []
        self.console_errors = []
        self.page.on("pageerror", lambda error: self.page_errors.append(str(error)))
        self.page.on("console", lambda message: self.console_errors.append(message.text) if message.type == "error" else None)

        def predict(route):
            self.payloads.append(route.request.post_data_json)
            route.fulfill(status=200, json={
                "prediction": "on_time", "risk_score": 0.2, "risk_level": "LOW", "estimated_delay_min": 0,
                "probabilities": {"on_time": 0.8, "delayed": 0.19, "cancelled": 0.01},
                "cluster_profile": {"name": "Test cluster", "delay_rate_pct": 20}, "matched_rules": [],
            })

        self.page.route("**/api/predict", predict)
        metrics = {"accuracy": 0.8, "f1_macro": 0.5, "recall_macro": 0.5, "roc_auc_ovr": 0.7}
        self.page.route("**/api/stats", lambda route: route.fulfill(status=200, json={
            "supervised": {"classification": {name: metrics for name in
                           ("LogisticRegression", "RandomForest", "HistGradientBoosting")},
                           "regression": {"MAE_minutes": 12, "RMSE_minutes": 20}, "feature_importances": []}
        }))
        self.page.route("**/api/clusters", lambda route: route.fulfill(status=200, json={
            "profiles": {}, "anomalies": {"anomaly_count": 0, "anomaly_percentage": 0}
        }))
        self.page.route("**/api/rules", lambda route: route.fulfill(status=200, json=[]))
        self.page.goto(self.url + "/", wait_until="networkidle")
        self.page.wait_for_function("document.querySelector('#res-risk-score').textContent === '20%'")

    def tearDown(self):
        self.context.close()


@unittest.skipUnless(importlib.util.find_spec("playwright") and _browser_path(),
                     "Browser checks need Playwright and an installed Chrome/Chromium/Edge")
class TestTV7BrowserIntegration(BrowserFixture, unittest.TestCase):
    def test_presets_send_flight_inputs_and_update_summary(self):
        for preset, airline, airport, hour in (("safe", "VN", "SGN", "08:00"),
                                              ("windy", "QH", "DAD", "14:45"),
                                              ("storm", "VJ", "SGN", "19:30")):
            with self.subTest(preset=preset), self.page.expect_response("**/api/predict"):
                self.page.locator(f'[data-preset="{preset}"]').click()
            payload = self.payloads[-1]
            self.assertEqual((payload["airline_code"], payload["airport"], payload["scheduled_time"]),
                             (airline, airport, hour))
            self.assertEqual(self.page.locator("#summary-origin").inner_text(), payload["origin"])
            self.assertNotIn("delay_min", payload)
            self.assertNotIn("actual_time", payload)
        self.assertEqual(self.page_errors, [])

    def test_route_swap_and_invalid_combinations(self):
        self.page.locator("#swap-route-btn").click()
        self.assertEqual(self.page.locator("#origin").input_value(), "HAN")
        self.assertEqual(self.page.locator("#destination").input_value(), "SGN")
        self.assertEqual(self.page.locator("#airport").input_value(), "HAN")
        self.page.select_option("#destination", "HAN")
        self.assertFalse(self.page.locator("#predict-form").evaluate("form => form.checkValidity()"))
        before = len(self.payloads)
        self.page.locator('button[type="submit"]').click()
        self.page.wait_for_timeout(200)
        self.assertEqual(len(self.payloads), before)
        self.assertTrue(self.page.locator("#destination").evaluate("element => !!element.validationMessage"))
        self.page.select_option("#destination", "ICN")
        self.assertFalse(self.page.locator("#route-type").evaluate("element => element.checkValidity()"))
        self.page.select_option("#route-type", "international")
        self.assertTrue(self.page.locator("#predict-form").evaluate("form => form.checkValidity()"))
        self.page.select_option("#direction", "ARRIVALS")
        self.assertFalse(self.page.locator("#airport").evaluate("element => element.checkValidity()"))
        self.page.select_option("#destination", "SGN")
        self.page.select_option("#route-type", "domestic")
        self.page.select_option("#airport", "SGN")
        self.assertTrue(self.page.locator("#predict-form").evaluate("form => form.checkValidity()"))
        self.page.locator("#swap-route-btn").click()
        self.assertEqual(self.page.locator("#airport").input_value(), "HAN")
        self.assertTrue(self.page.locator("#predict-form").evaluate("form => form.checkValidity()"))

    def test_navigation_updates_tabs_and_accessibility(self):
        for name in ("models-tab", "clusters-tab", "rules-tab", "simulator-tab"):
            self.page.locator(f'[data-tab="{name}"]').click()
            self.assertTrue(self.page.locator(f"#{name}").is_visible())
            self.assertEqual(self.page.locator(f'[data-tab="{name}"]').get_attribute("aria-selected"), "true")
            self.assertEqual(self.page.locator('.nav-btn[aria-selected="true"]').count(), 1)
        self.assertEqual(self.page_errors, [])

    def test_mobile_has_no_page_level_horizontal_overflow(self):
        self.page.set_viewport_size({"width": 390, "height": 844})
        self.page.wait_for_timeout(100)
        dims = self.page.evaluate("({width: innerWidth, scroll: document.documentElement.scrollWidth})")
        self.assertLessEqual(dims["scroll"], dims["width"])
        self.assertTrue(self.page.locator("#predict-form").is_visible())

    def test_real_backend_prediction_when_artifacts_are_ready(self):
        try:
            with urlopen(self.url + "/health", timeout=15) as response:
                if response.status != 200:
                    self.skipTest("Artifact bundle is not ready")
        except (HTTPError, URLError) as exc:
            if isinstance(exc, HTTPError):
                exc.close()
            self.skipTest("Artifact bundle is not ready")
        self.page.unroute("**/api/predict")
        for preset in ("safe", "windy", "storm"):
            with self.subTest(preset=preset), self.page.expect_response("**/api/predict", timeout=15000) as pending:
                self.page.locator(f'[data-preset="{preset}"]').click()
            response = pending.value
            self.assertEqual(response.status, 200)
            data = response.json()
            self.page.wait_for_function("level => document.querySelector('#res-risk-level').textContent.includes(level)",
                                        arg=data["risk_level"])
            self.assertIn(data["prediction"], {"on_time", "delayed", "cancelled"})
            self.assertEqual(set(data["probabilities"]), {"on_time", "delayed", "cancelled"})


if __name__ == "__main__":
    unittest.main()
