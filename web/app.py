# -*- coding: utf-8 -*-
"""Local HTTP server and JSON API for FlightRisk VN (Part 6A).

The HTTP interface delegates all inference to modeling.predict.predict_risk.
No training, fitting, or changes to data artifacts occur in a request.
"""
import json
import logging
import mimetypes
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import BASE_DIR, FILE_RULES
from modeling.predict import ArtifactError, _load_artifacts, _read_json, _read_rules, predict_risk

WEB_DIR = BASE_DIR / "web"
STATIC_DIR = WEB_DIR / "static"
TEMPLATES_DIR = WEB_DIR / "templates"
MODELS_DIR = BASE_DIR / "models"
MAX_BODY_BYTES = 16 * 1024
REQUIRED_FIELDS = {"airline_code", "airport", "direction", "route_type", "scheduled_time", "flight_date"}
OPTIONAL_FIELDS = {"precipitation", "wind_gusts_10m", "visibility", "weather_code", "airport_load",
                   "origin", "destination"}
LOGGER = logging.getLogger(__name__)


def get_supervised_stats() -> dict:
    return _read_json(MODELS_DIR / "supervised_metrics.json")


def get_unsupervised_stats() -> dict:
    profiles = _read_json(MODELS_DIR / "cluster_profiles.json")
    metrics = _read_json(MODELS_DIR / "unsupervised_metrics.json")
    if "isolation_forest" not in metrics:
        raise ArtifactError("Thiếu thống kê Isolation Forest trong unsupervised_metrics.json.")
    return {"profiles": profiles, "anomalies": metrics["isolation_forest"], "pca": metrics.get("pca", {})}


def get_rules(limit: int = 50) -> list:
    rules = _read_rules(FILE_RULES)
    # FP-Growth output is not guaranteed to be globally sorted by lift.
    def sort_lift(rule):
        try:
            return float(rule.get("Lift") or 0)
        except (TypeError, ValueError):
            return 0.0
    return sorted(rules, key=sort_lift, reverse=True)[:limit]


class FlightApiHandler(SimpleHTTPRequestHandler):
    """Serve the existing static UI and the five contracted API endpoints."""

    def _send_json(self, data, status_code: int = 200):
        body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path: Path, content_type: str):
        if not path.is_file():
            self.send_error(404, "Not Found")
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                return self._serve_file(TEMPLATES_DIR / "index.html", "text/html; charset=utf-8")
            if path.startswith("/static/"):
                base = STATIC_DIR.resolve()
                target = (base / unquote(path[len("/static/"):])).resolve()
                if not target.is_relative_to(base):
                    self.send_error(404, "Not Found")
                    return
                mime, _ = mimetypes.guess_type(str(target))
                return self._serve_file(target, mime or "application/octet-stream")
            if path == "/api/stats":
                return self._send_json({"supervised": get_supervised_stats()})
            if path == "/api/clusters":
                return self._send_json(get_unsupervised_stats())
            if path == "/api/rules":
                return self._send_json(get_rules())
            if path == "/health":
                try:
                    _load_artifacts()
                except ArtifactError as exc:
                    return self._send_json({"status": "degraded", "service": "FlightRisk API",
                                            "ready": False, "error": str(exc)}, 503)
                return self._send_json({"status": "healthy", "service": "FlightRisk API", "ready": True})
            return self._send_json({"error": "Không tìm thấy endpoint."}, 404)
        except ArtifactError as exc:
            return self._send_json({"error": str(exc)}, 503)
        except (OSError, ValueError):
            LOGGER.exception("GET request failed: %s", path)
            return self._send_json({"error": "Không thể xử lý yêu cầu."}, 500)

    def do_POST(self):
        if urlparse(self.path).path != "/api/predict":
            return self._send_json({"error": "Không tìm thấy endpoint."}, 404)
        if self.headers.get_content_type() != "application/json":
            return self._send_json({"error": "Content-Type phải là application/json."}, 415)
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length)
            if length < 0:
                raise ValueError
        except (TypeError, ValueError):
            return self._send_json({"error": "Content-Length không hợp lệ."}, 400)
        if length > MAX_BODY_BYTES:
            return self._send_json({"error": "Dữ liệu gửi lên quá lớn."}, 413)
        try:
            def reject_constant(value):
                raise ValueError(f"Giá trị JSON không hợp lệ: {value}")
            payload = json.loads(self.rfile.read(length).decode("utf-8"), parse_constant=reject_constant)
            if not isinstance(payload, dict):
                raise ValueError("Nội dung JSON phải là một object.")
            missing = sorted(REQUIRED_FIELDS - payload.keys())
            if missing:
                raise ValueError("Thiếu trường bắt buộc: " + ", ".join(missing))
            extra = sorted(payload.keys() - REQUIRED_FIELDS - OPTIONAL_FIELDS)
            if extra:
                raise ValueError("Trường không được hỗ trợ: " + ", ".join(extra))
            return self._send_json(predict_risk(**payload))
        except (ValueError, UnicodeDecodeError) as exc:
            return self._send_json({"error": str(exc)}, 400)
        except ArtifactError as exc:
            return self._send_json({"error": str(exc)}, 503)
        except Exception:
            LOGGER.exception("Prediction request failed")
            return self._send_json({"error": "Không thể thực hiện dự báo."}, 500)

    def do_HEAD(self):
        # SimpleHTTPRequestHandler's default HEAD serves arbitrary cwd paths.
        self.send_error(405, "Method Not Allowed")


def start_server(host: str = "127.0.0.1", port: int = 5000):
    """Run the API and static UI until interrupted."""
    server = ThreadingHTTPServer((host, port), FlightApiHandler)
    print(f"FlightRisk VN: http://{host}:{port}/ (health: /health)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    p = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 5000
    start_server(port=p)
