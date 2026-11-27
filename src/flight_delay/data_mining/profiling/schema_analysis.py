# -*- coding: utf-8 -*-
"""[STEP 3] Multi-Layer Schema Lineage & Mixed-Column Inventory."""
from typing import Dict

import pandas as pd

from flight_delay.common.constants import RAW_AVIATIONSTACK_DIR, RAW_FIDS_DIR


def analyze_raw_mixed_schema(std_rows: int) -> Dict:
    """Inspect heterogeneous Bronze CSV columns (e.g. 58 raw fields across FIDS & Aviationstack)."""
    files = sorted(list(RAW_FIDS_DIR.glob("*.csv")) + list(RAW_AVIATIONSTACK_DIR.glob("*.csv")))
    raw_cols = set()
    total_rows = 0
    per_file = []
    av_miss = {}

    for p in files:
        head = pd.read_csv(p, nrows=0)
        rc = max(sum(1 for _ in p.open("r", encoding="utf-8", errors="ignore")) - 1, 0)
        total_rows += rc
        raw_cols.update(head.columns)
        per_file.append({"file": p.name, "rows": rc, "cols": len(head.columns)})
        if p.name.startswith("aviationstack"):
            av_df = pd.read_csv(p, dtype=str, keep_default_na=False)
            for col in ("arrival.baggage", "departure.gate", "aircraft.registration", "departure.terminal", "arrival.terminal"):
                if col in av_df.columns:
                    av_miss[col] = round(float(av_df[col].str.strip().eq("").mean() * 100), 2)

    return {
        "raw_file_count": len(files),
        "raw_mixed_rows": total_rows,
        "raw_mixed_columns": len(raw_cols),
        "standardized_snapshot_rows": std_rows,
        "dropped_sparse_aviationstack_fields_pct": av_miss,
        "files_summary": per_file,
    }
