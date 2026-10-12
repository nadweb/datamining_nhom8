# -*- coding: utf-8 -*-
import csv
import json
import logging
import os
import re
import time
import unicodedata
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from config import RAW_DIR, USAGE_FILE, UA, FIELDS, SITES, AIRLINE_NAME_CODE

log = logging.getLogger("collector")
_PW = {}

def norm(s):
    s = unicodedata.normalize("NFD", str(s or ""))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").replace("đ", "d").replace("Đ", "D")
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def split_flight(code):
    code = (code or "").strip().rstrip(". ")
    m = re.match(r"^([A-Z0-9]{2})\s*0*(\d+)", code.upper())
    return (m.group(1), code) if m else ("", code)

def blank_row(site, scraped_at):
    r = {k: "" for k in FIELDS}
    r.update(source=site["name"], airport=site["airport"], direction=site["direction"], scraped_at=scraped_at)
    return r

def write_rows(source, rows):
    if not rows:
        return 0
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{source}_{datetime.now():%Y%m}.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
    return len(rows)

# ----------------- NỘI BÀI (ASP.NET) -----------------
DATE_OPT = re.compile(r"(\d{1,2})\s*/\s*([A-Za-z]{3})\s*/\s*(\d{4})")

def parse_noibai_table(soup, site, flight_date, scraped_at):
    table = None
    for t in soup.find_all("table"):
        first = t.find("tr")
        if first and "chuyenbay" in norm(first.get_text(" ")):
            table = t
            break
    if table is None:
        return []
    trs = table.find_all("tr")
    headers = [norm(c.get_text()) for c in trs[0].find_all(["th", "td"])]
    direction = "arrivals" if len(headers) > 1 and headers[1] == "tu" else "departures" if len(headers) > 1 and headers[1] == "den" else site["direction"]
    rows = []
    for tr in trs[1:]:
        tds = tr.find_all("td")
        if len(tds) < 5:
            continue
        txt = [td.get_text(" ", strip=True) for td in tds]
        sched, _, est = [p.strip() for p in txt[0].partition("/")]
        code, flight = split_flight(txt[3])
        img = tds[2].find("img")
        if img and img.get("src"):
            m = re.search(r"/([A-Z0-9]{2})\.\w+$", img["src"])
            code = code or (m.group(1) if m else "")
        r = blank_row(site, scraped_at)
        r["direction"] = direction
        r.update(flight_date=flight_date, airline_code=code, flight_no=flight,
                 other_airport=txt[1], scheduled_time=sched, estimated_time=est,
                 terminal=txt[4], status=txt[5] if len(txt) > 5 else "")
        rows.append(r)
    return rows

def scrape_noibai(site, session, scraped_at):
    r = session.get(site["url"], timeout=40)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    date_sel = soup.find("select", id=re.compile(r"ddlDate"))
    now_str = datetime.now().strftime("%Y-%m-%d")
    return parse_noibai_table(soup, site, now_str, scraped_at)

# ----------------- GUIDE SITES (PLAYWRIGHT) -----------------
def pw_html(url):
    if "b" not in _PW:
        from playwright.sync_api import sync_playwright
        _PW["p"] = sync_playwright().start()
        for ch in ["msedge", "chrome", None]:
            try:
                _PW["b"] = _PW["p"].chromium.launch(channel=ch, headless=True)
                break
            except Exception:
                continue
    ctx = _PW["b"].new_context(user_agent=UA)
    try:
        pg = ctx.new_page()
        pg.route("**/*", lambda route: route.abort() if route.request.resource_type in ("image", "media", "font") else route.continue_())
        pg.goto(url, wait_until="domcontentloaded", timeout=45000)
        try:
            pg.wait_for_selector("table tr td", timeout=25000)
        except Exception:
            pass
        return pg.content()
    finally:
        ctx.close()

def parse_guide(html, site, scraped_at):
    soup = BeautifulSoup(html, "lxml")
    now = datetime.strptime(scraped_at, "%Y-%m-%d %H:%M:%S")
    rows = []
    for t in soup.find_all("table"):
        trs = t.find_all("tr")
        if not trs:
            continue
        hdr = [norm(c.get_text()) for c in trs[0].find_all(["th", "td"])]
        if not (any("schedul" in h for h in hdr) and "flight" in hdr):
            continue
        hdr = ["scheduled" if "schedul" in h else "revised" if "revis" in h else h for h in hdr]
        direction = "arrivals" if "from" in hdr else "departures" if "to" in hdr else site["direction"]
        ih = {h: i for i, h in enumerate(hdr)}
        place_i = ih.get("from", ih.get("to"))
        for tr in trs[1:]:
            tds = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            if len(tds) < len(hdr):
                continue
            flight_txt = tds[ih["flight"]].split("Codeshare:")[0].strip()
            m = re.match(r"^(.*?)\s+([A-Z0-9*]{2,3})\s*(\d{1,4})\b", flight_txt)
            if not m:
                continue
            airline_name, code_shown, number = m.group(1).strip(), m.group(2), m.group(3)
            code = next((c for k, c in AIRLINE_NAME_CODE.items() if k in airline_name.lower()), code_shown if len(code_shown) == 2 else "")
            r = blank_row(site, scraped_at)
            r["direction"] = direction
            r.update(flight_date=now.strftime("%Y-%m-%d"), airline_code=code, flight_no=f"{code or code_shown}{number}",
                     other_airport=tds[place_i] if place_i is not None else "",
                     scheduled_time=tds[ih["scheduled"]],
                     estimated_time=tds[ih["revised"]] if "revised" in ih else "",
                     status=tds[ih["status"]] if "status" in ih else "")
            rows.append(r)
    return rows

def scrape_guide(site, session, scraped_at):
    html = pw_html(site["url"])
    rows = parse_guide(html, site, scraped_at)
    if not rows:
        r = session.get(site["url"], timeout=30)
        rows = parse_guide(r.text, site, scraped_at)
    return rows

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
            rows = scrape_noibai(site, session, scraped_at) if site["kind"] == "noibai" else scrape_guide(site, session, scraped_at)
            n = write_rows(site["name"], rows)
            log.info("[%s] ghi %d dòng", tag, n)
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

def api_aviationstack(airlines, max_requests=8, status_filter="cancelled"):
    key = os.environ.get("AVIATIONSTACK_KEY")
    if not key:
        log.error("Chưa đặt biến môi trường AVIATIONSTACK_KEY.")
        return 0
    scraped_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows, used = [], 0
    for al in airlines:
        if used >= max_requests:
            break
        params = {"access_key": key, "airline_iata": al, "limit": 100}
        if status_filter:
            params["flight_status"] = status_filter
        try:
            r = requests.get("https://api.aviationstack.com/v1/flights", params=params, timeout=30).json()
            used += 1
            usage_inc("aviationstack")
            for it in (r.get("data") or []):
                fl, air = it.get("flight") or {}, it.get("airline") or {}
                if fl.get("codeshared"):
                    continue
                st = (it.get("flight_status") or "").title()
                for d_dir, side, oth in [("departures", "departure", "arrival"), ("arrivals", "arrival", "departure")]:
                    a, o = it.get(side) or {}, it.get(oth) or {}
                    s_t = (a.get("scheduled") or "")[11:16]
                    row = blank_row({"name": "aviationstack_api", "airport": a.get("iata") or "", "direction": d_dir}, scraped_at)
                    row.update(flight_date=it.get("flight_date") or "", airline_code=air.get("iata") or "",
                               flight_no=fl.get("iata") or "", other_airport=o.get("iata") or "",
                               scheduled_time=s_t, status=st)
                    if row["airport"] and row["flight_no"]:
                        rows.append(row)
            log.info("[%s %s] %d chuyến", al, status_filter, len(r.get("data") or []))
        except Exception as e:
            log.error("[%s] Lỗi: %s", al, e)
    n = write_rows("aviationstack_api", rows)
    log.info("Aviationstack: ghi %d dòng", n)
    return n