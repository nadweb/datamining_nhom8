# -*- coding: utf-8 -*-
import csv
import json
import logging
import os
import re
import shutil
import time
import unicodedata
from datetime import datetime
import pandas as pd
import requests
from bs4 import BeautifulSoup
from config import DEBUG_DIR, RAW_DIR, USAGE_FILE, UA, FIELDS, SITES, AIRLINE_NAME_CODE

log = logging.getLogger("collector")
_PW = {}


# ----------------- DEBUG RAW JSON HELPER -----------------
def save_raw_json(data, prefix="raw", subdir="aviationstack"):
    """Lưu dữ liệu JSON gốc vào data/debug_data/<subdir>/ kèm timestamp."""
    try:
        target_dir = DEBUG_DIR / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = target_dir / f"{prefix}_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        log.info("Đã lưu debug JSON: %s/%s", subdir, path.name)
        return path
    except Exception as e:
        log.warning("Không thể lưu debug JSON (%s): %s", prefix, e)
        return None


def save_raw_html(html, tag, ts):
    """Lưu nguyên bản HTML (DOM sau khi JS chạy) vào data/debug_data/playwright/html/."""
    try:
        target_dir = DEBUG_DIR / "playwright" / "html"
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"pw_{tag}_{ts}.html"
        path.write_text(html or "", encoding="utf-8")
        log.info("Đã lưu HTML gốc: playwright/html/%s", path.name)
        return path
    except Exception as e:
        log.warning("Không thể lưu HTML gốc (%s): %s", tag, e)
        return None


def extract_guide_raw_json(dom_content, site, scraped_at, method="playwright"):
    """Trích xuất nội dung gốc từ Playwright DOM trên bộ nhớ sang cấu trúc JSON."""
    soup = BeautifulSoup(dom_content or "", "lxml")
    tables_raw = []
    for idx, t in enumerate(soup.find_all("table")):
        trs = t.find_all("tr")
        if not trs:
            continue
        headers = [c.get_text(" ", strip=True) for c in trs[0].find_all(["th", "td"])]
        raw_rows = []
        for tr in trs[1:]:
            cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            if cells:
                raw_rows.append(cells)
        if raw_rows:
            tables_raw.append({"table_index": idx, "headers": headers, "rows": raw_rows})
    return {
        "source": site["name"],
        "airport": site["airport"],
        "direction": site["direction"],
        "url": site.get("url", ""),
        "scraped_at": scraped_at,
        "method": method,
        "table_count": len(tables_raw),
        "tables": tables_raw,
    }


# ----------------- PARSING UTILITIES -----------------
def norm(s):
    s = unicodedata.normalize("NFD", str(s or ""))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").replace("đ", "d").replace("Đ", "D")
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def is_completed_status(st):
    return bool(re.search(r"arrive|depart|land|hacan|catcan", str(st or ""), re.IGNORECASE))


def blank_row(site, scraped_at):
    r = {k: "" for k in FIELDS}
    r.update(
        source=site["name"],
        airport=site["airport"],
        direction=site["direction"],
        scraped_at=scraped_at,
    )
    return r


# ----------------- GUIDE SITES (PLAYWRIGHT -> RAW JSON -> CSV) -----------------
def pw_dom_content(url):
    if "b" not in _PW:
        from playwright.sync_api import sync_playwright

        _PW["p"] = sync_playwright().start()
        for ch in ["msedge", "chrome", None]:
            try:
                _PW["b"] = _PW["p"].chromium.launch(channel=ch, headless=True)
                break
            except Exception:
                continue
        if "b" not in _PW:
            _PW["p"].stop()
            _PW.clear()
            raise RuntimeError("Không mở được trình duyệt Playwright (thử: playwright install chromium)")
    ctx = _PW["b"].new_context(user_agent=UA)
    try:
        pg = ctx.new_page()
        pg.route(
            "**/*",
            lambda route: route.abort()
            if route.request.resource_type in ("image", "media", "font")
            else route.continue_(),
        )
        pg.goto(url, wait_until="domcontentloaded", timeout=45000)
        try:
            pg.wait_for_selector("table tr td", timeout=25000)
        except Exception:
            pass
        return pg.content()
    finally:
        ctx.close()


def _guide_row_to_record(meta, cells, direction, place, scraped_at):
    """1 dòng bảng (cells: {tên cột chuẩn hóa: text}) -> 1 bản ghi chuẩn FIELDS, hoặc None nếu không đọc được."""
    flight_txt = (cells.get("flight") or "").split("Codeshare:")[0].strip()
    m = re.match(r"^(.*?)\s+([A-Z0-9*]{2,3})\s*(\d{1,4})\b", flight_txt)
    if not m:
        return None
    airline_name, code_shown, number = m.group(1).strip(), m.group(2), m.group(3)
    code = next(
        (c for k, c in AIRLINE_NAME_CODE.items() if k in airline_name.lower()),
        code_shown if len(code_shown) == 2 else "",
    )
    now = datetime.strptime(scraped_at[:19], "%Y-%m-%d %H:%M:%S")
    est_t = cells.get("revised") or ""
    status_str = cells.get("status") or ""
    act_t = est_t if (is_completed_status(status_str) and est_t) else ""

    r = blank_row(meta, scraped_at)
    r["direction"] = direction
    r.update(
        flight_date=now.strftime("%Y-%m-%d"),
        airline_code=code,
        flight_no=f"{code or code_shown}{number}",
        other_airport=place or "",
        scheduled_time=cells.get("scheduled") or "",
        estimated_time=est_t,
        actual_time=act_t,
        status=status_str,
    )
    return r


def _std_header(h):
    k = norm(h)
    return "scheduled" if "schedul" in k else "revised" if "revis" in k else k


def parse_guide_from_json(payload):
    """JSON gốc của trang Guide (bảng HTML) -> các dòng chuẩn FIELDS."""
    scraped_at = payload.get("scraped_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta = {
        "name": payload.get("source", ""),
        "airport": payload.get("airport", ""),
        "direction": payload.get("direction", ""),
    }
    rows = []
    for tbl in payload.get("tables", []):
        if not is_flight_table(tbl.get("headers", [])):
            continue
        hdr = [_std_header(h) for h in tbl["headers"]]
        direction = "arrivals" if "from" in hdr else "departures" if "to" in hdr else meta["direction"]
        for tds in tbl.get("rows", []):
            if len(tds) < len(hdr):
                continue
            cells = dict(zip(hdr, tds))
            place = cells.get("from", cells.get("to", ""))
            rec = _guide_row_to_record(meta, cells, direction, place, scraped_at)
            if rec:
                rows.append(rec)
    return rows


def is_flight_table(headers):
    hdr = [norm(h) for h in headers]
    return any("schedul" in h for h in hdr) and "flight" in hdr


def scrape_guide(site, session, scraped_at):
    tag = f"{site['name']}_{site.get('direction', 'all')}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Bước 1: lấy HTML gốc và lưu NGAY (trước khi parse) để luôn có bản gốc
    html = pw_dom_content(site["url"])
    html_path = save_raw_html(html, tag, ts)
    payload = extract_guide_raw_json(html, site, scraped_at, method="playwright")
    rows = parse_guide_from_json(payload)

    if not rows:
        resp = session.get(site["url"], timeout=30)
        html = resp.text
        html_path = save_raw_html(html, tag + "_requests", ts)
        payload = extract_guide_raw_json(html, site, scraped_at, method="requests_fallback")
        rows = parse_guide_from_json(payload)

    # Bước 2: HTML -> JSON (bảng); processor parse lại JSON này thành CSV
    payload["html_file"] = html_path.name if html_path else ""
    save_raw_json(payload, prefix=f"pw_{tag}", subdir="playwright")
    return rows


def html_to_json(overwrite=False):
    """Chuyển các file HTML đã lưu (playwright/html/*.html) thành JSON bảng trong playwright/.
    Dùng khi sửa logic trích xuất bảng và muốn dựng lại JSON từ HTML gốc, không cần cào lại."""
    html_dir = DEBUG_DIR / "playwright" / "html"
    out_dir = DEBUG_DIR / "playwright"
    if not html_dir.exists():
        log.warning("Chưa có thư mục %s", html_dir)
        return 0
    sites = {(s["name"], s["direction"]): s for s in SITES}
    n = 0
    for p in sorted(html_dir.glob("pw_*.html")):
        # pw_<guide_xxx>_<direction>[_requests]_<YYYYMMDD>_<HHMMSS>.html
        m = re.match(r"^pw_(guide_[a-z]+)_(arrivals|departures)(?:_requests)?_(\d{8})_(\d{6})$", p.stem)
        if not m or (m.group(1), m.group(2)) not in sites:
            log.warning("Bỏ qua (tên file không nhận diện được): %s", p.name)
            continue
        out = out_dir / f"{p.stem}.json"
        if out.exists() and not overwrite:
            continue
        scraped_at = datetime.strptime(m.group(3) + m.group(4), "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
        payload = extract_guide_raw_json(
            p.read_text(encoding="utf-8", errors="ignore"),
            sites[(m.group(1), m.group(2))], scraped_at, method="from_saved_html",
        )
        payload["html_file"] = p.name
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        n += 1
    log.info("Đã chuyển %d file HTML -> JSON", n)
    return n


def collect_all(only=None):
    scraped_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session = requests.Session()
    session.headers.update({"User-Agent": UA})
    total = 0
    for site in SITES:
        if only and not any(o.strip() in site["name"] for o in only.split(",")):
            continue
        tag = f"{site['name']}/{site['direction']}"
        try:
            rows = scrape_guide(site, session, scraped_at)
            n = len(rows)
            log.info("[%s] thu %d dòng (đã lưu HTML + JSON)", tag, n)
            total += n
        except Exception as e:
            log.error("[%s] lỗi: %s", tag, e)
        time.sleep(1.5)
    if "b" in _PW:
        try:
            _PW["b"].close()
            _PW["p"].stop()
        except Exception:
            pass
        _PW.clear()
    log.info("Tổng dòng thu được lượt này: %d", total)
    rebuild_raw_csv()
    return total


# ----------------- AVIATIONSTACK API -----------------
def usage_inc(name, n=1):
    USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        d = json.loads(USAGE_FILE.read_text(encoding="utf-8"))
    except Exception:
        d = {}
    m = d.setdefault(datetime.now().strftime("%Y-%m"), {})
    m[name] = m.get(name, 0) + n
    USAGE_FILE.write_text(json.dumps(d, indent=1), encoding="utf-8")


def format_aviationstack_status(fl_status, d_dir, delay_val, has_actual):
    """Sinh chuỗi status đầy đủ (Departed On-Time, Departed X minutes late, Arrived..., Cancelled)."""
    st = (fl_status or "").strip().lower()
    if st == "cancelled":
        return "Cancelled"
    if st in ("diverted", "incident"):
        return st.title()

    delay = int(delay_val) if (delay_val is not None and str(delay_val).isdigit()) else 0
    prefix = "Departed" if d_dir == "departures" else "Arrived"

    if st == "landed" or (st == "active" and d_dir == "departures") or has_actual:
        return f"{prefix} {delay} minutes late" if delay > 0 else f"{prefix} On-Time"
    if st == "active" and d_dir == "arrivals":
        return f"Delayed {delay} minutes" if delay > 0 else "On-Time"
    if st == "scheduled":
        return f"Delayed {delay} minutes" if delay > 0 else "Scheduled"
    return st.title() or "Scheduled"


def parse_aviationstack_json(raw_json, scraped_at=None):
    """Phân tích JSON gốc từ Aviationstack sang danh sách bản ghi chuẩn FIELDS."""
    scraped_at = scraped_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for it in raw_json.get("data") or []:
        fl, air = it.get("flight") or {}, it.get("airline") or {}
        if fl.get("codeshared"):
            continue
        fl_status = it.get("flight_status") or ""
        for d_dir, side, oth in [
            ("departures", "departure", "arrival"),
            ("arrivals", "arrival", "departure"),
        ]:
            a, o = it.get(side) or {}, it.get(oth) or {}
            s_t = (a.get("scheduled") or "")[11:16]
            e_t = (a.get("estimated") or "")[11:16]
            a_t = (a.get("actual") or "")[11:16]
            delay_val = a.get("delay")

            status_str = format_aviationstack_status(fl_status, d_dir, delay_val, bool(a_t))
            if not a_t and is_completed_status(status_str) and e_t:
                a_t = e_t

            row = blank_row(
                {
                    "name": "aviationstack_api",
                    "airport": a.get("iata") or "",
                    "direction": d_dir,
                },
                scraped_at,
            )
            row.update(
                flight_date=it.get("flight_date") or "",
                airline_code=air.get("iata") or "",
                flight_no=fl.get("iata") or "",
                other_airport=o.get("iata") or "",
                scheduled_time=s_t,
                estimated_time=e_t,
                actual_time=a_t,
                status=status_str,
            )
            if row["airport"] and row["flight_no"]:
                rows.append(row)
    return rows


def api_aviationstack(airlines, max_requests=8, status_filter=""):
    key = os.environ.get("AVIATIONSTACK_KEY")
    if not key:
        log.error("Chưa đặt biến môi trường AVIATIONSTACK_KEY.")
        return 0
    scraped_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows, used = [], 0
    statuses = [s.strip() for s in status_filter.split(",")] if status_filter else [""]

    for st_filter in statuses:
        for al in airlines:
            if used >= max_requests:
                break
            params = {"access_key": key, "airline_iata": al, "limit": 100}
            if st_filter and st_filter.lower() != "all":
                params["flight_status"] = st_filter
            try:
                resp = requests.get(
                    "http://api.aviationstack.com/v1/flights", params=params, timeout=30
                )
                r = resp.json()
                used += 1
                usage_inc("aviationstack")
                if r.get("error"):
                    log.error("[%s] API trả lỗi: %s", al, r["error"])
                    continue

                status_tag = st_filter if st_filter else "all"
                save_raw_json(
                    r,
                    prefix=f"aviationstack_{al}_{status_tag}",
                    subdir="aviationstack",
                )
                parsed = parse_aviationstack_json(r, scraped_at=scraped_at)
                rows.extend(parsed)
                log.info("[%s %s] %d chuyến (%d bản ghi)", al, status_tag, len(r.get("data") or []), len(parsed))
            except Exception as e:
                log.error("[%s] Lỗi: %s", al, e)
    log.info("Aviationstack: thu %d dòng (đã lưu JSON gốc)", len(rows))
    rebuild_raw_csv()
    return len(rows)

# ----------------- JSON GỐC -> CSV THÔ (mỗi nguồn giữ nguyên cột của chính JSON/HTML của nó) -----------------
RAW_META = ["source", "airport", "direction", "scraped_at", "table_index", "source_file"]
LEGACY_DIR = RAW_DIR.parent / "raw_legacy"
BUILT_MARK = RAW_DIR / ".built_from_json"


def flatten_json(obj, prefix="", out=None):
    """Làm phẳng JSON lồng nhau: {"departure": {"iata": "SGN"}} -> {"departure.iata": "SGN"}. Mọi giá trị là chuỗi."""
    out = {} if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten_json(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(obj, list):
        out[prefix] = json.dumps(obj, ensure_ascii=False)
    else:
        out[prefix] = "" if obj is None else str(obj)
    return out


def unflatten_row(row):
    """Ngược lại của flatten_json (bỏ giá trị rỗng) để dùng lại parser của JSON."""
    out = {}
    for key, val in row.items():
        if val == "":
            continue
        node, parts = out, key.split(".")
        for p in parts[:-1]:
            if not isinstance(node.get(p), dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = val
    return out


def _unique_headers(headers):
    seen, out = {}, []
    for i, h in enumerate(headers):
        h = (h or "").strip() or f"col_{i}"
        seen[h] = seen.get(h, 0) + 1
        out.append(h if seen[h] == 1 else f"{h}_{seen[h]}")
    return out


def _json_file_ts(path, data):
    if isinstance(data, dict) and data.get("scraped_at"):
        return str(data["scraped_at"])[:19]
    m = re.search(r"(\d{8}_\d{6})", path.stem)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def collect_raw_rows():
    """Đọc JSON gốc -> {tên file CSV (không có _YYYYMM): [dòng thô]}. Chỉ làm phẳng, KHÔNG đổi tên/bỏ cột."""
    groups, bad = {}, 0

    def add(name, row):
        groups.setdefault(name, []).append(row)

    pw_dir = DEBUG_DIR / "playwright"
    for p in sorted(pw_dir.glob("*.json")) if pw_dir.exists() else []:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            ts = _json_file_ts(p, data)
            if isinstance(data, list) or (isinstance(data, dict) and "tables" not in data):
                # định dạng cũ: bản ghi đã được parse sẵn (không còn bảng HTML gốc)
                recs = data if isinstance(data, list) else (data.get("parsed_records") or [])
                for r in recs:
                    row = {k: ("" if v is None else str(v)) for k, v in r.items()}
                    row.setdefault("scraped_at", ts)
                    row["scraped_at"] = row["scraped_at"] or ts
                    row["source_file"] = p.name
                    add(f"{row.get('source') or 'guide_unknown'}_legacy", row)
                continue
            for tbl in data.get("tables", []):
                if not is_flight_table(tbl.get("headers", [])):
                    continue
                hdr = _unique_headers(tbl["headers"])
                for cells in tbl.get("rows", []):
                    row = {
                        "source": data.get("source", ""), "airport": data.get("airport", ""),
                        "direction": data.get("direction", ""), "scraped_at": ts,
                        "table_index": str(tbl.get("table_index", "")), "source_file": p.name,
                    }
                    row.update({h: c for h, c in zip(hdr, cells)})
                    add(row["source"], row)
        except Exception:
            bad += 1

    av_dir = DEBUG_DIR / "aviationstack"
    for p in sorted(av_dir.glob("*.json")) if av_dir.exists() else []:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            ts = _json_file_ts(p, None)
            for it in data.get("data") or []:
                row = {"scraped_at": ts, "source_file": p.name}
                row.update(flatten_json(it))
                add("aviationstack_api", row)
        except Exception:
            bad += 1
    return groups, bad


def rebuild_raw_csv():
    """Dựng lại data/raw/<nguồn>_<YYYYMM>.csv HOÀN TOÀN từ JSON gốc: mỗi nguồn giữ cột riêng của nó.
    Lần đầu: CSV cũ được sao lưu sang data/raw_legacy/ rồi mới ghi đè."""
    groups, bad = collect_raw_rows()
    if bad:
        log.warning("Bỏ qua %d file JSON lỗi", bad)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    old = list(RAW_DIR.glob("*.csv"))
    if old and not BUILT_MARK.exists():
        LEGACY_DIR.mkdir(parents=True, exist_ok=True)
        for p in old:
            shutil.copy2(p, LEGACY_DIR / p.name)
        log.info("Đã sao lưu %d CSV cũ -> %s", len(old), LEGACY_DIR)
    for p in old:
        p.unlink()
    n = 0
    for name, rows in groups.items():
        df = pd.DataFrame(rows).fillna("").astype(str)
        df["_m"] = df["scraped_at"].str[:7].str.replace("-", "", regex=False)
        cols = [c for c in RAW_META if c in df.columns]
        cols += [c for c in df.columns if c not in cols and c != "_m"]
        for m, g in df.groupby("_m"):
            g.sort_values("scraped_at", kind="stable")[cols].to_csv(
                RAW_DIR / f"{name}_{m}.csv", index=False, encoding="utf-8-sig")
            n += len(g)
    if groups:
        BUILT_MARK.write_text("raw CSV được dựng từ debug_data/*.json\n", encoding="utf-8")
    log.info("Đã dựng lại data/raw từ JSON gốc: %d dòng, %d file nguồn", n, len(groups))
    return n


# ----------------- CSV THÔ -> SCHEMA CHUẨN FIELDS (bước tích hợp, dùng chung cho mọi nguồn) -----------------
def _std_aviationstack(df):
    recs = []
    for ts, g in df.groupby("scraped_at"):
        items = [unflatten_row(r) for r in g.to_dict("records")]
        recs.extend(parse_aviationstack_json({"data": items}, scraped_at=ts))
    return recs


def _std_guide(df):
    recs = []
    for r in df.to_dict("records"):
        meta = {"name": r.get("source", ""), "airport": r.get("airport", ""), "direction": r.get("direction", "")}
        cells = {_std_header(k): v for k, v in r.items() if k not in RAW_META}
        frm, to = (cells.get("from") or "").strip(), (cells.get("to") or "").strip()
        direction = "arrivals" if frm else "departures" if to else meta["direction"]
        rec = _guide_row_to_record(meta, cells, direction, frm or to, r["scraped_at"])
        if rec:
            recs.append(rec)
    return recs


def load_standard_records():
    """Đọc data/raw/*.csv (mỗi nguồn một cột riêng) -> DataFrame theo schema chuẩn FIELDS."""
    frames = []
    for f in sorted(RAW_DIR.glob("*.csv")):
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        if df.empty:
            continue
        if f.stem.startswith("aviationstack"):
            recs = _std_aviationstack(df)
        elif "_legacy_" in f.stem:                     # đã parse sẵn từ trước -> chỉ chọn cột chuẩn
            frames.append(df.reindex(columns=FIELDS, fill_value=""))
            continue
        else:
            recs = _std_guide(df)
        if recs:
            frames.append(pd.DataFrame(recs, columns=FIELDS))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FIELDS)
