# -*- coding: utf-8 -*-
"""[STEP 1] Bronze Data Lake Source Registry & Provenance Manifest."""
import shutil
from datetime import datetime
from typing import Dict

import pandas as pd

from flight_delay.common.constants import (
    DATA_DIR,
    DATA_MANIFESTS_DIR,
    DEBUG_DIR,
    FIELDS,
    FILE_WEATHER,
    FILE_WEATHER_LEGACY,
    RAW_AVIATIONSTACK_DIR,
    RAW_DIR,
    RAW_FIDS_DIR,
    RAW_WEATHER_DIR,
)
from flight_delay.common.hashing import file_checksum_record
from flight_delay.common.utils import write_json


def sync_bronze_layout() -> Dict[str, int]:
    """Ensure raw scraped CSVs and weather cache are mirrored safely into data/raw/{fids,aviationstack,weather}.

    Never modifies data/debug_data/ or deletes existing raw files.
    """
    RAW_FIDS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_AVIATIONSTACK_DIR.mkdir(parents=True, exist_ok=True)
    RAW_WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    copied = {"fids": 0, "aviationstack": 0, "weather": 0}

    for src in sorted(RAW_DIR.glob("*.csv")):
        dest_dir = RAW_AVIATIONSTACK_DIR if src.name.startswith("aviationstack") else RAW_FIDS_DIR
        dest = dest_dir / src.name
        if not dest.exists() or dest.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dest)
            key = "aviationstack" if dest_dir == RAW_AVIATIONSTACK_DIR else "fids"
            copied[key] += 1

    if FILE_WEATHER_LEGACY.exists() and (
        not FILE_WEATHER.exists() or FILE_WEATHER.stat().st_size != FILE_WEATHER_LEGACY.stat().st_size
    ):
        shutil.copy2(FILE_WEATHER_LEGACY, FILE_WEATHER)
        copied["weather"] += 1

    return copied


def build_bronze_manifest() -> dict:
    """Audit Bronze files (CSV counts, raw columns, debug archives, checksums) into data/manifests/bronze_manifest.json."""
    sync_bronze_layout()
    sources = []
    all_raw_cols = set()
    total_raw_rows = 0

    for category, directory in (
        ("fids", RAW_FIDS_DIR),
        ("aviationstack", RAW_AVIATIONSTACK_DIR),
        ("weather", RAW_WEATHER_DIR),
    ):
        for p in sorted(directory.glob("*.csv")):
            df_head = pd.read_csv(p, nrows=0)
            row_count = max(sum(1 for _ in p.open("r", encoding="utf-8", errors="ignore")) - 1, 0)
            if category in ("fids", "aviationstack"):
                total_raw_rows += row_count
                all_raw_cols.update(df_head.columns)
            rec = file_checksum_record(p, DATA_DIR.parent)
            rec.update({"category": category, "rows": row_count, "columns": list(df_head.columns)})
            sources.append(rec)

    debug_counts = {}
    if DEBUG_DIR.exists():
        for sub in sorted(DEBUG_DIR.iterdir()):
            if sub.is_dir():
                debug_counts[sub.name] = sum(1 for x in sub.rglob("*") if x.is_file())

    manifest = {
        "layer": "bronze_raw",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_csv_count": sum(1 for s in sources if s["category"] in ("fids", "aviationstack")),
        "total_raw_mixed_rows": total_raw_rows,
        "total_unique_raw_columns": len(all_raw_cols),
        "standardized_target_fields": FIELDS,
        "debug_archive_counts": debug_counts,
        "debug_archive_total_files": sum(debug_counts.values()),
        "files": sources,
    }
    DATA_MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(DATA_MANIFESTS_DIR / "bronze_manifest.json", manifest)
    return manifest
