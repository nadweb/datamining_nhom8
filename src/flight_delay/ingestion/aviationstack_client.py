# -*- coding: utf-8 -*-
"""[STEP 1] Aviationstack REST API Client & Flattening Parser."""
import json
import logging
import os
from datetime import datetime
from typing import List

import pandas as pd
import requests

from flight_delay.common.constants import SOURCES_CFG, USAGE_FILE
from flight_delay.ingestion.fids_collector import blank_row, is_completed_status, save_raw_json

log = logging.getLogger(__name__)


def usage_inc(name: str, n: int = 1) -> None:
    USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        d = json.loads(USAGE_FILE.read_text(encoding="utf-8"))
    except Exception:
        d = {}
    m = d.setdefault(datetime.now().strftime("%Y-%m"), {})
    m[name] = m.get(name, 0) + n
    USAGE_FILE.write_text(json.dumps(d, indent=1), encoding="utf-8")


def flatten_json(obj, prefix: str = "", out: dict = None) -> dict:
    out = {} if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten_json(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(obj, list):
        out[prefix] = json.dumps(obj, ensure_ascii=False)
    else:
        out[prefix] = "" if obj is None else str(obj)
    return out


def unflatten_row(row: dict) -> dict:
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


def format_aviationstack_status(fl_status: str, d_dir: str, delay_val, has_actual: bool) -> str:
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


def parse_aviationstack_json(raw_json: dict, scraped_at: str = None) -> List[dict]:
    scraped_at = scraped_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for it in raw_json.get("data") or []:
        fl, air = it.get("flight") or {}, it.get("airline") or {}
        if fl.get("codeshared"):
            continue
        fl_status = it.get("flight_status") or ""
        for d_dir, side, oth in (("departures", "departure", "arrival"), ("arrivals", "arrival", "departure")):
            a, o = it.get(side) or {}, it.get(oth) or {}
            s_t = (a.get("scheduled") or "")[11:16]
            e_t = (a.get("estimated") or "")[11:16]
            a_t = (a.get("actual") or "")[11:16]
            delay_val = a.get("delay")
            status_str = format_aviationstack_status(fl_status, d_dir, delay_val, bool(a_t))
            if not a_t and is_completed_status(status_str) and e_t:
                a_t = e_t
            row = blank_row({"name": "aviationstack_api", "airport": a.get("iata") or "", "direction": d_dir}, scraped_at)
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


def std_aviationstack_df(df: pd.DataFrame) -> List[dict]:
    recs = []
    for ts, g in df.groupby("scraped_at"):
        items = [unflatten_row(r) for r in g.to_dict("records")]
        recs.extend(parse_aviationstack_json({"data": items}, scraped_at=ts))
    return recs


def fetch_aviationstack(airlines: List[str], max_requests: int = 8, status_filter: str = "") -> int:
    key = os.environ.get("AVIATIONSTACK_KEY")
    if not key:
        log.error("Missing environment variable AVIATIONSTACK_KEY.")
        return 0
    endpoint = SOURCES_CFG["aviationstack"]["endpoint"]
    limit = SOURCES_CFG["aviationstack"]["default_limit"]
    scraped_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows, used = [], 0
    statuses = [s.strip() for s in status_filter.split(",")] if status_filter else [""]
    for st_filter in statuses:
        for al in airlines:
            if used >= max_requests:
                break
            params = {"access_key": key, "airline_iata": al, "limit": limit}
            if st_filter and st_filter.lower() != "all":
                params["flight_status"] = st_filter
            try:
                resp = requests.get(endpoint, params=params, timeout=30)
                r = resp.json()
                used += 1
                usage_inc("aviationstack")
                if r.get("error"):
                    log.error("[%s] API error: %s", al, r["error"])
                    continue
                status_tag = st_filter or "all"
                save_raw_json(r, prefix=f"aviationstack_{al}_{status_tag}", subdir="aviationstack")
                rows.extend(parse_aviationstack_json(r, scraped_at=scraped_at))
            except Exception as exc:
                log.error("[%s] Request failed: %s", al, exc)
    return len(rows)
