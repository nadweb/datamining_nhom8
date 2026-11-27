# -*- coding: utf-8 -*-
"""[STEP 1] FIDS Airport Timetable Collector (Playwright HTML -> Raw JSON -> Standardized Records)."""
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from bs4 import BeautifulSoup

from flight_delay.common.constants import AIRLINE_NAME_CODE, DEBUG_DIR, FIELDS, SITES, UA
from flight_delay.common.utils import norm_text

log = logging.getLogger(__name__)
_PW: Dict[str, object] = {}
RAW_META = ["source", "airport", "direction", "scraped_at", "table_index", "source_file"]


def is_completed_status(st: object) -> bool:
    return bool(re.search(r"arrive|depart|land|hacan|catcan", str(st or ""), re.IGNORECASE))


def is_flight_table(headers: list) -> bool:
    hdr = [norm_text(h) for h in headers]
    return any("schedul" in h for h in hdr) and "flight" in hdr


def std_header(h: object) -> str:
    k = norm_text(h)
    return "scheduled" if "schedul" in k else "revised" if "revis" in k else k


def blank_row(site: dict, scraped_at: str) -> dict:
    r = {k: "" for k in FIELDS}
    r.update(
        source=site["name"],
        airport=site["airport"],
        direction=site["direction"],
        scraped_at=scraped_at,
    )
    return r


def save_raw_json(data, prefix: str = "raw", subdir: str = "playwright") -> Optional[Path]:
    try:
        target_dir = DEBUG_DIR / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = target_dir / f"{prefix}_{ts}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
    except Exception as exc:
        log.warning("Failed saving raw JSON (%s): %s", prefix, exc)
        return None


def save_raw_html(html: str, tag: str, ts: str) -> Optional[Path]:
    try:
        target_dir = DEBUG_DIR / "playwright" / "html"
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"pw_{tag}_{ts}.html"
        path.write_text(html or "", encoding="utf-8")
        return path
    except Exception as exc:
        log.warning("Failed saving raw HTML (%s): %s", tag, exc)
        return None


def extract_guide_raw_json(dom_content: str, site: dict, scraped_at: str, method: str = "playwright") -> dict:
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


def _guide_row_to_record(meta: dict, cells: dict, direction: str, place: str, scraped_at: str) -> Optional[dict]:
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


def parse_guide_from_json(payload: dict) -> List[dict]:
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
        hdr = [std_header(h) for h in tbl["headers"]]
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


def std_guide_df(df: pd.DataFrame) -> List[dict]:
    """Convert raw FIDS CSV DataFrame into standardized 12-col records."""
    recs = []
    for r in df.to_dict("records"):
        meta = {"name": r.get("source", ""), "airport": r.get("airport", ""), "direction": r.get("direction", "")}
        cells = {std_header(k): v for k, v in r.items() if k not in RAW_META}
        frm, to = (cells.get("from") or "").strip(), (cells.get("to") or "").strip()
        direction = "arrivals" if frm else "departures" if to else meta["direction"]
        rec = _guide_row_to_record(meta, cells, direction, frm or to, r["scraped_at"])
        if rec:
            recs.append(rec)
    return recs


def _pw_dom_content(url: str) -> str:
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
            raise RuntimeError("Playwright browser launch failed")
    ctx = _PW["b"].new_context(user_agent=UA)
    try:
        pg = ctx.new_page()
        pg.route(
            "**/*",
            lambda route: route.abort() if route.request.resource_type in ("image", "media", "font") else route.continue_(),
        )
        pg.goto(url, wait_until="domcontentloaded", timeout=45000)
        try:
            pg.wait_for_selector("table tr td", timeout=25000)
        except Exception:
            pass
        return pg.content()
    finally:
        ctx.close()


def scrape_guide_site(site: dict, session, scraped_at: str) -> List[dict]:
    tag = f"{site['name']}_{site.get('direction', 'all')}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    html = _pw_dom_content(site["url"])
    html_path = save_raw_html(html, tag, ts)
    payload = extract_guide_raw_json(html, site, scraped_at, method="playwright")
    rows = parse_guide_from_json(payload)
    if not rows:
        resp = session.get(site["url"], timeout=30)
        html = resp.text
        html_path = save_raw_html(html, tag + "_requests", ts)
        payload = extract_guide_raw_json(html, site, scraped_at, method="requests_fallback")
        rows = parse_guide_from_json(payload)
    payload["html_file"] = html_path.name if html_path else ""
    save_raw_json(payload, prefix=f"pw_{tag}", subdir="playwright")
    return rows


def close_playwright() -> None:
    if "b" in _PW:
        try:
            _PW["b"].close()
            _PW["p"].stop()
        except Exception:
            pass
        _PW.clear()


def html_to_json(overwrite: bool = False) -> int:
    html_dir = DEBUG_DIR / "playwright" / "html"
    out_dir = DEBUG_DIR / "playwright"
    if not html_dir.exists():
        log.warning("Missing directory %s", html_dir)
        return 0
    sites = {(s["name"], s["direction"]): s for s in SITES}
    count = 0
    for p in sorted(html_dir.glob("pw_*.html")):
        m = re.match(r"^pw_(guide_[a-z]+)_(arrivals|departures)(?:_requests)?_(\d{8})_(\d{6})$", p.stem)
        if not m or (m.group(1), m.group(2)) not in sites:
            continue
        out = out_dir / f"{p.stem}.json"
        if out.exists() and not overwrite:
            continue
        scraped_at = datetime.strptime(m.group(3) + m.group(4), "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
        payload = extract_guide_raw_json(
            p.read_text(encoding="utf-8", errors="ignore"),
            sites[(m.group(1), m.group(2))],
            scraped_at,
            method="from_saved_html",
        )
        payload["html_file"] = p.name
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        count += 1
    return count
