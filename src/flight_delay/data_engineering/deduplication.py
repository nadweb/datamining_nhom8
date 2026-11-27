# -*- coding: utf-8 -*-
"""[STEP 2] 3-Tier SHA-256 Deduplication & Pre-Flight Snapshot Lineage."""
from typing import Dict, Tuple

import pandas as pd

from flight_delay.common.constants import AIRPORT_COORD, FIELDS
from flight_delay.common.hashing import row_hash

K1 = ["airport", "direction", "flight_date", "flight_no", "scheduled_time"]


def deduplicate_snapshots(
    df: pd.DataFrame,
    prefer_side: str = "DEPARTURES",
    unit: str = "leg",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Deduplicate multi-source snapshots into unique flight legs while tracking snapshot horizons."""
    stats: Dict = {"raw_snapshots": int(len(df))}
    df = df.copy()
    df["raw_hash"] = row_hash(df, FIELDS)
    stats["exact_duplicates"] = int(df["raw_hash"].duplicated().sum())
    df = df.drop_duplicates("raw_hash", keep="last").copy()
    stats["after_exact_dedup"] = int(len(df))

    df["has_done_act"] = (df["actual_time"].fillna("").astype(str).str.strip() != "").astype(int)
    df["has_info"] = (~df["status"].fillna("").astype(str).str.upper().isin(["", "SCHEDULED"])).astype(int)
    df["is_pref_side"] = (df["direction"] == prefer_side).astype(int)
    df["at_monitored"] = df["airport"].isin(AIRPORT_COORD).astype(int)
    df["event_hash"] = row_hash(df, K1)
    df["leg_hash"] = row_hash(df, ["leg_key"])

    # Compute snapshot availability flags relative to scheduled_dt (for T-1h, T-3h, T-6h audits)
    scrap_dt = pd.to_datetime(df["scraped_at"], errors="coerce")
    lead_hours = (df["sched_dt"] - scrap_dt).dt.total_seconds() / 3600.0
    df["_has_t1h"] = (lead_hours >= 1.0).astype(int)
    df["_has_t3h"] = (lead_hours >= 3.0).astype(int)
    df["_has_t6h"] = (lead_hours >= 6.0).astype(int)

    df = df.sort_values(K1 + ["has_info", "has_done_act", "scraped_at"])
    per_src = df.drop_duplicates(["event_hash", "source"], keep="last")
    src_cnt = df.groupby("event_hash")["source"].nunique().rename("source_recorded_count")
    side = df.drop_duplicates("event_hash", keep="last").merge(src_cnt, on="event_hash", how="left")
    stats["after_side_dedup"] = int(len(side))

    # Aggregate horizon coverage per leg before final row collapse
    leg_cov = df.groupby("leg_hash").agg(
        has_t1h_snapshot=("_has_t1h", "max"),
        has_t3h_snapshot=("_has_t3h", "max"),
        has_t6h_snapshot=("_has_t6h", "max"),
        earliest_scraped_at=("scraped_at", "min"),
        latest_scraped_at=("scraped_at", "max"),
    )

    side = side.sort_values(["leg_key", "has_info", "has_done_act", "at_monitored", "is_pref_side", "scraped_at"])
    legs_cnt = side.groupby("leg_hash")["direction"].nunique().rename("n_sides_seen")
    if unit == "event":
        leg = side.merge(legs_cnt, on="leg_hash", how="left")
    else:
        leg = side.drop_duplicates("leg_hash", keep="last").merge(legs_cnt, on="leg_hash", how="left")
    leg = leg.merge(leg_cov, on="leg_hash", how="left")

    stats["unit"] = unit
    stats["after_leg_dedup"] = int(len(leg))
    stats["legs_seen_both_sides"] = int((leg["n_sides_seen"] > 1).sum())
    stats["legs_with_t1h_snapshot"] = int(leg["has_t1h_snapshot"].sum())
    stats["legs_with_t3h_snapshot"] = int(leg["has_t3h_snapshot"].sum())
    stats["legs_with_t6h_snapshot"] = int(leg["has_t6h_snapshot"].sum())
    return side, leg, per_src, stats
