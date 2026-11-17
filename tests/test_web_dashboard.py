# -*- coding: utf-8 -*-
"""Member 8 dashboard: browser interaction and real backend integration.

Install Playwright and use a local Chrome/Chromium/Edge to run the browser checks.
The test server binds to an ephemeral local port and does not modify artifacts.
"""
import importlib.util
import unittest

from tests.test_web_frontend_browser import BrowserFixture, _browser_path


@unittest.skipUnless(importlib.util.find_spec("playwright") and _browser_path(),
                     "Browser checks need Playwright and an installed Chrome/Chromium/Edge")
class TestWebDashboardMember8(BrowserFixture, unittest.TestCase):
    def test_each_preset_sends_one_prediction_with_correct_values(self):
        initial = len(self.payloads)
        self.assertEqual(initial, 1)
        for offset, (preset, rain, wind, load) in enumerate(
            (("safe", 0, 10, 15), ("windy", 1, 32, 26), ("storm", 15, 45, 35)), start=1
        ):
            with self.page.expect_response("**/api/predict"):
                self.page.locator(f'[data-preset="{preset}"]').click()
            self.page.wait_for_function("document.querySelector('#res-risk-score').textContent === '20%'")
            self.assertEqual(len(self.payloads), initial + offset)
            payload = self.payloads[-1]
            self.assertEqual((payload["precipitation"], payload["wind_gusts_10m"], payload["airport_load"]),
                             (rain, wind, load))
        self.assertEqual(self.page_errors, [])

    def test_api_error_clears_result_and_user_can_retry(self):
        self.page.unroute("**/api/predict")
        self.page.route("**/api/predict", lambda route: route.fulfill(status=503, json={"error": "Thiếu mô hình"}))
        with self.page.expect_response("**/api/predict"):
            self.page.locator('[data-preset="storm"]').click()
        self.page.wait_for_function("document.querySelector('#res-risk-level').textContent === 'LỖI DỰ BÁO'")
        self.assertIn("Thiếu mô hình", self.page.locator("#res-prediction-text").inner_text())
        self.assertEqual(self.page.locator("#res-risk-score").inner_text(), "--")
        self.assertEqual(self.page.locator("#prob-delayed-text").inner_text(), "--")
        self.assertEqual(self.page.locator("#res-rules-list .rule-item").count(), 0)
        self.assertNotIn("risk-high", self.page.locator(".result-header").get_attribute("class"))

        self.page.unroute("**/api/predict")
        self.page.route("**/api/predict", lambda route: route.fulfill(status=200, json={
            "prediction": "on_time", "risk_score": 0.2, "risk_level": "LOW", "estimated_delay_min": 0,
            "probabilities": {"on_time": 0.8, "delayed": 0.19, "cancelled": 0.01},
            "cluster_profile": {"cluster_name": "Cụm kiểm thử", "cluster_risk_avg": "0.00%"},
            "matched_rules": [],
        }))
        with self.page.expect_response("**/api/predict"):
            self.page.locator('button[type="submit"]').click()
        self.page.wait_for_function("document.querySelector('#res-risk-score').textContent === '20%'")
        self.assertEqual(self.page.locator("#res-cluster-rate").inner_text(), "0.00%")

    def test_invalid_prediction_is_rejected_without_fake_values(self):
        self.page.unroute("**/api/predict")
        self.page.route("**/api/predict", lambda route: route.fulfill(status=200, json={
            "prediction": "delayed", "risk_score": 0.9, "risk_level": "HIGH", "estimated_delay_min": 3,
            "probabilities": {"on_time": 0.8, "delayed": 0.19, "cancelled": 0.01},
            "cluster_profile": {}, "matched_rules": [],
        }))
        with self.page.expect_response("**/api/predict"):
            self.page.locator('[data-preset="storm"]').click()
        self.page.wait_for_function("document.querySelector('#res-risk-level').textContent === 'LỖI DỰ BÁO'")
        self.assertIn("cấu trúc dữ liệu", self.page.locator("#res-prediction-text").inner_text())
        self.assertEqual(self.page.locator("#res-risk-score").inner_text(), "--")

    def test_zero_delay_and_untrusted_text_are_rendered_safely(self):
        injected = '<img src=x onerror="window.injected=1">'
        self.page.unroute("**/api/predict")
        self.page.route("**/api/predict", lambda route: route.fulfill(status=200, json={
            "prediction": "delayed", "risk_score": 0.6, "risk_level": "HIGH", "estimated_delay_min": 0,
            "probabilities": {"on_time": 0.4, "delayed": 0.6, "cancelled": 0},
            "cluster_profile": {"cluster_name": injected, "cluster_risk_avg": "0.00%"},
            "matched_rules": [{"antecedent": injected, "consequent": "Status=delayed", "confidence": 0, "lift": 1}],
        }))
        with self.page.expect_response("**/api/predict"):
            self.page.locator('[data-preset="storm"]').click()
        self.page.wait_for_function("document.querySelector('#res-risk-score').textContent === '60%'")
        self.assertIn("0 PHÚT", self.page.locator("#res-prediction-text").inner_text())
        self.assertEqual(self.page.locator("#res-cluster-rate").inner_text(), "0.00%")
        self.assertEqual(self.page.locator("#res-cluster-name").inner_text(), injected)
        self.assertIn(injected, self.page.locator("#res-rules-list").inner_text())
        self.assertIn("0.00%", self.page.locator("#res-rules-list").inner_text())
        self.assertEqual(self.page.locator("#res-rules-list img").count(), 0)
        self.assertIsNone(self.page.evaluate("window.injected"))

    def test_stale_response_and_route_swap_do_not_show_old_prediction(self):
        self.page.evaluate("""() => {
          const original = window.fetch.bind(window);
          window.pendingPredictions = [];
          window.fetch = (url, options) => url === '/api/predict'
            ? new Promise(resolve => window.pendingPredictions.push(resolve)) : original(url, options);
          window.finishPrediction = (i, score) => window.pendingPredictions[i]({ok: true, status: 200,
            json: async () => ({prediction: score === 0.2 ? 'on_time' : 'delayed', risk_score: score,
              risk_level: score === 0.2 ? 'LOW' : 'HIGH', estimated_delay_min: 0,
              probabilities: {on_time: 1-score, delayed: score, cancelled: 0},
              cluster_profile: {cluster_name: 'Test', cluster_risk_avg: '0%'}, matched_rules: []})});
        }""")
        self.page.locator('[data-preset="storm"]').click()
        self.page.locator('[data-preset="safe"]').click()
        self.page.wait_for_function("window.pendingPredictions.length === 2")
        self.page.evaluate("window.finishPrediction(1, 0.2)")
        self.page.wait_for_function("document.querySelector('#res-risk-score').textContent === '20%'")
        self.page.evaluate("window.finishPrediction(0, 0.6)")
        self.page.wait_for_timeout(100)
        self.assertEqual(self.page.locator("#res-risk-score").inner_text(), "20%")
        self.page.locator("#swap-route-btn").click()
        self.assertEqual(self.page.locator("#res-risk-score").inner_text(), "--")
        self.assertIn("Thông số đã thay đổi", self.page.locator("#res-prediction-text").inner_text())

    def test_real_dashboard_data_matches_api(self):
        for endpoint in ("stats", "clusters", "rules"):
            self.page.unroute(f"**/api/{endpoint}")
        with self.page.expect_response("**/api/stats") as pending:
            self.page.locator('[data-tab="models-tab"]').click()
        stats = pending.value.json()["supervised"]
        self.page.wait_for_function("document.querySelector('#metrics-content').dataset.loaded === 'true'")
        model_text = self.page.locator("#metrics-content").inner_text()
        self.assertIn(str(stats["split"]["test_rows"]), model_text)
        self.assertIn(stats["model_selection"]["selected"], model_text)
        selected = self.page.locator("#metrics-content tbody tr").filter(has_text=stats["model_selection"]["selected"])
        self.assertIn("Được chọn", selected.inner_text())

        with self.page.expect_response("**/api/clusters") as pending:
            self.page.locator('[data-tab="clusters-tab"]').click()
        clusters = pending.value.json()
        self.page.wait_for_function("document.querySelector('#clusters-content').dataset.loaded === 'true'")
        self.assertEqual(self.page.locator("#clusters-content tbody tr").count(), len(clusters["profiles"]))
        self.assertIn(str(clusters["anomalies"]["anomaly_count"]), self.page.locator("#clusters-content").inner_text())

        with self.page.expect_response("**/api/rules") as pending:
            self.page.locator('[data-tab="rules-tab"]').click()
        rules = pending.value.json()
        self.page.wait_for_function("document.querySelector('#rules-content').dataset.loaded === 'true'")
        self.assertEqual(self.page.locator("#rules-content tbody tr").count(), min(15, len(rules)))
        if rules:
            self.assertIn(rules[0]["Antecedent"], self.page.locator("#rules-content tbody tr").first.inner_text())
        self.assertEqual(self.page_errors, [])
        self.assertEqual(self.console_errors, [])

    def test_failed_analytics_can_be_retried(self):
        self.page.unroute("**/api/stats")
        self.page.route("**/api/stats", lambda route: route.fulfill(status=503, json={"error": "Thiếu thống kê"}))
        with self.page.expect_response("**/api/stats"):
            self.page.locator('[data-tab="models-tab"]').click()
        self.page.wait_for_selector("#metrics-content [role=alert]")
        self.assertIn("Thiếu thống kê", self.page.locator("#metrics-content").inner_text())
        self.assertEqual(self.page.locator("#metrics-content table").count(), 0)
        self.page.unroute("**/api/stats")
        self.page.route("**/api/stats", lambda route: route.fulfill(status=200, json={
            "supervised": {"classification": {"Mô hình thử": {"accuracy": 0.8}}, "regression": {}}
        }))
        with self.page.expect_response("**/api/stats"):
            self.page.locator('#metrics-content button').click()
        self.page.wait_for_function("document.querySelector('#metrics-content').dataset.loaded === 'true'")
        self.assertIn("Mô hình thử", self.page.locator("#metrics-content").inner_text())


if __name__ == "__main__":
    unittest.main()
