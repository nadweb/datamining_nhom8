# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import sys
import time
from datetime import datetime, timedelta
import pandas as pd
from config import LOG_DIR, RAW_DIR
from collector import collect_all, api_aviationstack
from processor import process_and_mine
from weather import update_from_raw

def setup_log():
    LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(LOG_DIR / "pipeline.log", encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)])

def print_stats():
    files = glob.glob(str(RAW_DIR / "*.csv"))
    if not files:
        print("Chưa có file dữ liệu nào trong data/raw.")
        return
    df = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in files])
    key = ["source", "airport", "direction", "flight_date", "flight_no", "scheduled_time"]
    u = df.drop_duplicates(key)
    print(f"Tổng snapshot: {len(df):,} | Chuyến duy nhất (theo nguồn): {len(u):,} | Số nguồn: {u['source'].nunique()}")
    print("\nTheo từng nguồn:")
    print(u.groupby("source").size().to_string())

def main():
    parser = argparse.ArgumentParser(description="Flight Data Mining Pipeline CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sp_c = sub.add_parser("collect", help="Thu thập dữ liệu từ FIDS sân bay")
    sp_c.add_argument("--only", help="Chỉ thu thập nguồn chỉ định (vd: noibai, guide_)")

    sp_l = sub.add_parser("loop", help="Tự động lặp lại chu kỳ thu thập")
    sp_l.add_argument("--every", type=int, default=120, help="Số phút giữa 2 lượt")

    sp_a = sub.add_parser("api", help="Thu thập chuyến hủy/hạ cánh từ Aviationstack")
    sp_a.add_argument("--status", default="cancelled", help="Lọc trạng thái (cancelled/landed)")
    sp_a.add_argument("--max-requests", type=int, default=8, help="Số request tối đa")

    sub.add_parser("weather", help="Tải thời tiết theo giờ (Open-Meteo) cho các sân bay")

    sp_p = sub.add_parser("process", help="Khử trùng cấp chuyến bay, ghép thời tiết, khai phá luật (FP-Growth)")
    sp_p.add_argument("--threshold", type=int, default=15, help="Ngưỡng trễ (phút), mặc định 15")
    sp_p.add_argument("--compare", type=int, help="Chạy thêm một ngưỡng khác để so sánh (vd 30)")
    sp_p.add_argument("--unit", choices=["leg", "event"], default="leg",
                      help="leg: 1 chuyến = 1 mẫu (mặc định, chặt chẽ); event: cất cánh và hạ cánh tính là 2 mẫu riêng")
    sp_p.add_argument("--min-age-hours", type=float, default=0,
                      help="Chỉ giữ chuyến có giờ bay cách lần cào cuối ít nhất N giờ (vd 3), để nhãn đã chốt")
    sub.add_parser("stats", help="Xem thống kê nhanh số dòng snapshot")

    args = parser.parse_args()
    setup_log()

    if args.cmd == "collect":
        collect_all(args.only)
    elif args.cmd == "loop":
        while True:
            collect_all()
            nxt = datetime.now() + timedelta(minutes=args.every)
            print(f"Lượt kế tiếp lúc: {nxt.strftime('%H:%M:%S')}")
            time.sleep(args.every * 60)
    elif args.cmd == "api":
        api_aviationstack(airlines=["VN", "VJ", "QH", "BL", "VU", "9G"],
                          max_requests=args.max_requests, status_filter=args.status)
    elif args.cmd == "weather":
        update_from_raw()
    elif args.cmd == "process":
        process_and_mine(args.threshold, args.compare, args.min_age_hours, args.unit)
    elif args.cmd == "stats":
        print_stats()

if __name__ == "__main__":
    main()