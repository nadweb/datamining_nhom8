# -*- coding: utf-8 -*-
"""[STEP 1] Unified Bronze Ingestion & Standardization into 12 FIELDS."""
import logging
import time
from typing import Optional

import pandas as pd
import requests

from flight_delay.common.constants import (
    FIELDS,
    FILE_STANDARDIZED,
    RAW_AVIATIONSTACK_DIR,
    RAW_DIR,
    RAW_FIDS_DIR,
    SITES,
    UA,
)
from flight_delay.ingestion.aviationstack_client import std_aviationstack_df
from flight_delay.ingestion.fids_collector import close_playwright, scrape_guide_site, std_guide_df
from flight_delay.ingestion.source_registry import build_bronze_manifest, sync_bronze_layout

log = logging.getLogger(__name__)


def load_standard_records() -> pd.DataFrame:
    """Load raw Bronze CSVs from data/raw/{fids,aviationstack} into canonical 12 FIELDS schema."""
    sync_bronze_layout()
    frames = []

    fids_files = sorted(RAW_FIDS_DIR.glob("*.csv"))
    av_files = sorted(RAW_AVIATIONSTACK_DIR.glob("*.csv"))
    if not fids_files and not av_files:
        # Fallback if tests populate a flat directory directly
        raw_csvs = sorted(RAW_DIR.glob("*.csv"))
        av_files = [f for f in raw_csvs if f.stem.startswith("aviationstack")]
        fids_files = [f for f in raw_csvs if not f.stem.startswith("aviationstack")]

    for f in av_files:
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        if not df.empty:
            recs = std_aviationstack_df(df)
            if recs:
                frames.append(pd.DataFrame(recs, columns=FIELDS))

    for f in fids_files:
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        if df.empty:
            continue
        if "_legacy_" in f.stem:
            frames.append(df.reindex(columns=FIELDS, fill_value=""))
            continue
        recs = std_guide_df(df)
        if recs:
            frames.append(pd.DataFrame(recs, columns=FIELDS))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FIELDS)


def collect_all(only: Optional[str] = None) -> int:
    """Collect fresh snapshots from airport FIDS portals (when requested)."""
    from datetime import datetime

    scraped_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session = requests.Session()
    session.headers.update({"User-Agent": UA})
    total = 0
    for site in SITES:
        if only and not any(o.strip() in site["name"] for o in only.split(",")):
            continue
        try:
            rows = scrape_guide_site(site, session, scraped_at)
            total += len(rows)
            log.info("[%s/%s] scraped %d rows", site["name"], site["direction"], len(rows))
        except Exception as exc:
            log.error("[%s/%s] scrape error: %s", site["name"], site["direction"], exc)
        time.sleep(1.5)
    close_playwright()
    return total


def collect_bronze_and_standardize() -> pd.DataFrame:
    """Audit Bronze data lake and emit standardized 12-field interim snapshots."""
    manifest = build_bronze_manifest()
    df = load_standard_records()
    FILE_STANDARDIZED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FILE_STANDARDIZED, index=False, encoding="utf-8-sig")
    log.info(
        "Bronze audit (%d mixed files, %d raw rows) -> %d standardized 12-col snapshots saved to %s",
        manifest["source_csv_count"],
        manifest["total_raw_mixed_rows"],
        len(df),
        FILE_STANDARDIZED,
    )
    return df
