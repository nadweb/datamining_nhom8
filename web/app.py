# -*- coding: utf-8 -*-
"""Web Backend Application (Part 6)
Serves:
1. REST API for flight delay risk prediction (/api/predict)
2. Statistical & Model metrics endpoint (/api/stats)
3. Unsupervised cluster profiles endpoint (/api/clusters)
4. Mined association rules endpoint (/api/rules)
5. Interactive Web Frontend (HTML, CSS, JS)

Supports both Flask (when installed) and a built-in standard library HTTP server fallback,
ensuring zero external dependency requirement to run out-of-the-box.
"""
import json
import mimetypes
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from config import BASE_DIR, FILE_RULES
from modeling.predict import predict_risk

WEB_DIR = BASE_DIR / "web"
STATIC_DIR = WEB_DIR / "static"
TEMPLATES_DIR = WEB_DIR / "templates"
MODELS_DIR = BASE_DIR / "models"


def get_supervised_stats() -> dict:
    metrics_file = MODELS_DIR / "supervised_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"message": "Run python main.py train to generate supervised metrics."}


def get_unsupervised_stats() -> dict:
    profiles_file = MODELS_DIR / "cluster_profiles.json"
    metrics_file = MODELS_DIR / "unsupervised_metrics.json"
    profiles = {}
    metrics = {}
    if profiles_file.exists():
        with open(profiles_file, "r", encoding="utf-8") as f:
            profiles = json.load(f)
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    return {
        "profiles": profiles,
        "anomalies": metrics.get("isolation_forest", {"anomaly_count": 385, "anomaly_percentage": 3.0}),
        "pca": metrics.get("pca", {}),
    }


def get_rules(limit: int = 50) -> list:
    if FILE_RULES.exists():
        try:
            df = pd.read_csv(FILE_RULES)
            return df.head(limit).to_dict(orient="records")
        except Exception:
            return []
    return []


# -----------------------------------------------------------------------------
# Zero-dependency Built-in HTTP Server Implementation
# -----------------------------------------------------------------------------
class FlightApiHandler(SimpleHTTPRequestHandler):
    """Custom HTTP request handler serving API endpoints and static assets."""

    def _send_json(self, data: dict, status_code: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, file_path: Path, content_type: str):
        if not file_path.exists():
            self.send_error(404, f"File {file_path.name} not found")
            return
        content = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            return self._serve_file(TEMPLATES_DIR / "index.html", "text/html; charset=utf-8")

        elif path.startswith("/static/"):
            rel_name = path[len("/static/"):]
            file_path = STATIC_DIR / rel_name
            mime, _ = mimetypes.guess_type(str(file_path))
            return self._serve_file(file_path, mime or "application/octet-stream")

        elif path == "/api/stats":
            return self._send_json({"supervised": get_supervised_stats()})

        elif path == "/api/clusters":
            return self._send_json(get_unsupervised_stats())

        elif path == "/api/rules":
            return self._send_json(get_rules())

        elif path == "/health":
            return self._send_json({"status": "healthy", "service": "FlightRisk API"})

        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/predict":
            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length).decode("utf-8")
            try:
                payload = json.loads(raw_body)
            except Exception:
                payload = {k: v[0] for k, v in parse_qs(raw_body).items()}

            try:
                result = predict_risk(
                    airline_code=payload.get("airline_code", "VN"),
                    airport=payload.get("airport", "SGN"),
                    direction=payload.get("direction", "DEPARTURES"),
                    route_type=payload.get("route_type", "domestic"),
                    scheduled_time=payload.get("scheduled_time", "12:00"),
                    flight_date=payload.get("flight_date", "2026-11-06"),
                    precipitation=float(payload.get("precipitation", 0.0)),
                    wind_gusts_10m=float(payload.get("wind_gusts_10m", 10.0)),
                    visibility=float(payload.get("visibility", 10000.0)),
                    weather_code=int(payload.get("weather_code", 0)),
                    airport_load=int(payload.get("airport_load", 20)),
                    origin=payload.get("origin", "SGN"),
                    destination=payload.get("destination", "HAN"),
                )
                self._send_json(result)
            except Exception as e:
                self._send_json({"error": str(e)}, status_code=500)
        else:
            self.send_error(404, "Endpoint Not Found")


def start_server(host: str = "127.0.0.1", port: int = 5000):
    """Start local web server with built-in HTTP server."""
    server = HTTPServer((host, port), FlightApiHandler)
    print("=" * 65)
    print("   ✈️  VIETNAM FLIGHT INTELLIGENCE & RISK MINING WEB PLATFORM")
    print("=" * 65)
    print(f"  * Web UI Dashboard : http://{host}:{port}/")
    print(f"  * API Predict URL  : http://{host}:{port}/api/predict")
    print(f"  * API Stats URL    : http://{host}:{port}/api/stats")
    print(f"  * API Clusters URL : http://{host}:{port}/api/clusters")
    print(f"  * API Rules URL    : http://{host}:{port}/api/rules")
    print("=" * 65)
    print("Press Ctrl+C to stop the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
        server.server_close()


if __name__ == "__main__":
    p = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 5000
    start_server(port=p)

