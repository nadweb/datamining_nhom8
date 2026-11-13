# -*- coding: utf-8 -*-
"""Measure cold and warm local HTTP prediction latency using the actual bundle.

Run from the project root: python -m tests.benchmark_web_backend
No models are trained and no artifact files are modified.
"""
import argparse
import json
import platform
import statistics
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from web.app import FlightApiHandler, ThreadingHTTPServer


def run(requests: int = 100, warmup: int = 5) -> dict:
    if requests < 1 or warmup < 0:
        raise ValueError("requests must be positive and warmup must be non-negative")
    server = ThreadingHTTPServer(("127.0.0.1", 0), FlightApiHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/api/predict"
    payload = json.dumps({"airline_code": "VJ", "airport": "SGN", "direction": "DEPARTURES",
                          "route_type": "domestic", "scheduled_time": "19:30", "flight_date": "2026-11-06",
                          "precipitation": 1.0, "wind_gusts_10m": 35, "visibility": 4000,
                          "weather_code": 95, "airport_load": 34}).encode("utf-8")

    def call() -> float:
        req = Request(url, data=payload, headers={"Content-Type": "application/json"})
        start = time.perf_counter()
        try:
            with urlopen(req, timeout=30) as response:
                json.load(response)
        except HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
            finally:
                exc.close()
        except URLError as exc:
            raise RuntimeError(f"Không thể gọi server: {exc.reason}") from exc
        return (time.perf_counter() - start) * 1000

    try:
        cold = call()
        for _ in range(warmup):
            call()
        times = [call() for _ in range(requests)]
        ordered = sorted(times)
        p95 = ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)]
        return {"platform": platform.platform(), "python": platform.python_version(),
                "cold_ms": round(cold, 2), "warmup_requests": warmup, "measured_requests": requests,
                "mean_ms": round(statistics.mean(times), 2), "median_ms": round(statistics.median(times), 2),
                "p95_ms": round(p95, 2), "max_ms": round(max(times), 2),
                "all_warm_requests_under_50ms": max(times) < 50}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.requests, args.warmup), ensure_ascii=False, indent=2))
    except (RuntimeError, ValueError) as exc:
        parser.exit(1, f"Không thể benchmark: {exc}\n")
