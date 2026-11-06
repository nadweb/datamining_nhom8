# -*- coding: utf-8 -*-
"""Main Unified CLI for Vietnam Flight Intelligence & Data Mining Project.
Supports:
  Data Integrity:
    python main.py check-data      Validate dataset against data contract (24 cols, keys, labels)
    python main.py test            Run automated test suite (21 unit tests)
  Data Pipeline:
    python main.py collect         Collect flight snapshots from airport FIDS (Playwright)
    python main.py loop            Automated collection loop
    python main.py api             Collect from Aviationstack API
    python main.py weather         Update weather cache from Open-Meteo
    python main.py build-raw       Rebuild data/raw from original debug JSON
    python main.py stats           Quick statistics from data/raw
  Processing & ML:
    python main.py process         Clean, integrate, attach weather, FP-Growth rules
    python main.py features        Run Part 3 feature engineering pipeline
    python main.py encode          Run Part 4 encoding, scaling & chronological split
    python main.py train           Run Part 5A supervised training (Classification & Regression)
    python main.py unsupervised    Run Part 5B unsupervised learning (K-Means, PCA, Anomaly)
  Web Platform:
    python main.py web             Launch web UI dashboard and prediction REST API
"""
import argparse
import logging
import sys
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import LOG_DIR
from pipeline.collector import (
    api_aviationstack, collect_all, html_to_json, load_standard_records,
    rebuild_raw_csv
)
from pipeline.processor import process_and_mine
from pipeline.weather import update_from_raw
from pipeline.check_data import run_cli_check


def setup_log():
    LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "pipeline.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def print_stats():
    df = load_standard_records()
    if df.empty:
        print("Chưa có dữ liệu trong data/raw (chạy: python main.py build-raw).")
        return
    key = ["source", "airport", "direction", "flight_date", "flight_no", "scheduled_time"]
    u = df.drop_duplicates(key)
    print(f"Tổng snapshot: {len(df):,} | Chuyến duy nhất (theo nguồn): {len(u):,} | Số nguồn: {u['source'].nunique()}")
    print("\nTheo từng nguồn:")
    print(u.groupby("source").size().to_string())


def run_all_tests() -> int:
    """Run full automated test suite."""
    print("=" * 65)
    print("          RUNNING AUTOMATED TEST SUITE (pytest/unittest)")
    print("=" * 65)
    loader = unittest.TestLoader()
    suite = loader.discover("tests")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser(description="Vietnam Flight Data Mining & Prediction CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # 1. Data Contract & Quality
    sp_chk = sub.add_parser("check-data", help="Kiểm định ràng buộc dữ liệu (24 cột, khóa duy nhất, nhãn chuẩn)")
    sp_chk.add_argument("path", nargs="?", default=None, help="Đường dẫn file CSV cần kiểm tra")

    # 2. Automated Tests
    sub.add_parser("test", help="Chạy toàn bộ 21 automated unit tests kiểm tra logic và data contract")

    # 3. Collection & Raw Building
    sp_c = sub.add_parser("collect", help="Thu thập dữ liệu từ FIDS sân bay (Playwright)")
    sp_c.add_argument("--only", help="Chỉ thu thập nguồn chỉ định (vd: guide_sgn, guide_han)")

    sp_l = sub.add_parser("loop", help="Tự động lặp lại chu kỳ thu thập")
    sp_l.add_argument("--every", type=int, default=120, help="Số phút giữa 2 lượt")

    sp_a = sub.add_parser("api", help="Thu thập chuyến bay từ Aviationstack")
    sp_a.add_argument("--status", default="cancelled,landed", help="Lọc trạng thái (cancelled,landed hoặc all)")
    sp_a.add_argument("--max-requests", type=int, default=12, help="Số request tối đa")

    sp_h = sub.add_parser("html2json", help="Dựng lại JSON bảng từ HTML gốc")
    sp_h.add_argument("--overwrite", action="store_true", help="Ghi đè JSON đã có")

    sub.add_parser("build-raw", help="Dựng lại data/raw/*.csv từ JSON gốc")
    sub.add_parser("weather", help="Tải thời tiết theo giờ từ Open-Meteo")
    sub.add_parser("stats", help="Xem thống kê nhanh từ data/raw")

    # 4. Processing & Feature Engineering
    sp_p = sub.add_parser("process", help="Khử trùng cấp chuyến bay, ghép thời tiết, khai phá luật (FP-Growth)")
    sp_p.add_argument("--threshold", type=int, default=15, help="Ngưỡng trễ (phút), mặc định 15")
    sp_p.add_argument("--compare", type=int, help="Chạy thêm một ngưỡng khác để so sánh (vd 30)")
    sp_p.add_argument("--unit", choices=["leg", "event"], default="leg")
    sp_p.add_argument("--min-age-hours", type=float, default=0)

    sub.add_parser("features", help="Chạy pipeline trích xuất đặc trưng Part 3 (cyclical time, hazard, load)")
    sub.add_parser("encode", help="Chạy Part 4 mã hóa One-Hot, chuẩn hóa StandardScaler và chia tập theo thời gian")

    # 5. Machine Learning (Supervised + Unsupervised)
    sub.add_parser("train", help="Huấn luyện mô hình Supervised (Phân loại 3 lớp & Hồi quy độ trễ)")
    sub.add_parser("unsupervised", help="Chạy mô hình Unsupervised (K-Means Clustering, PCA, Isolation Forest)")

    # 6. Web Application
    sp_w = sub.add_parser("web", help="Khởi chạy Web Dashboard & Prediction REST API")
    sp_w.add_argument("--port", type=int, default=5000, help="Port máy chủ web (mặc định 5000)")
    sp_w.add_argument("--host", default="127.0.0.1", help="Host máy chủ web")

    args = parser.parse_args()
    setup_log()

    if args.cmd == "check-data":
        sys.exit(run_cli_check(args.path))
    elif args.cmd == "test":
        sys.exit(run_all_tests())
    elif args.cmd == "collect":
        collect_all(args.only)
    elif args.cmd == "loop":
        while True:
            collect_all()
            nxt = datetime.now() + timedelta(minutes=args.every)
            print(f"Lượt kế tiếp lúc: {nxt.strftime('%H:%M:%S')}")
            time.sleep(args.every * 60)
    elif args.cmd == "api":
        api_aviationstack(
            airlines=["VN", "VJ", "QH", "BL", "VU", "9G"],
            max_requests=args.max_requests,
            status_filter=args.status,
        )
    elif args.cmd == "build-raw":
        rebuild_raw_csv()
    elif args.cmd == "html2json":
        html_to_json(args.overwrite)
    elif args.cmd == "weather":
        update_from_raw()
    elif args.cmd == "process":
        process_and_mine(args.threshold, args.compare, args.min_age_hours, args.unit)
    elif args.cmd == "stats":
        print_stats()
    elif args.cmd == "features":
        from features.features import run_feature_pipeline
        run_feature_pipeline()
    elif args.cmd == "encode":
        from modeling.encoding import build_and_save_encoded_dataset
        build_and_save_encoded_dataset()
    elif args.cmd == "train":
        from modeling.train import run_supervised_training
        run_supervised_training()
    elif args.cmd == "unsupervised":
        from modeling.unsupervised import run_unsupervised_pipeline
        run_unsupervised_pipeline()
    elif args.cmd == "web":
        from web.app import start_server
        start_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
