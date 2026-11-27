# -*- coding: utf-8 -*-
"""[STEP 4] FP-Growth Risk Pattern Mining with Redundancy Pruning & Holdout Temporal Validation."""
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth

from flight_delay.common.constants import EXPERIMENTS_CFG, FILE_RULES


def _build_transaction_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Gold flight rows into clean operational+weather + status outcome boolean items."""
    items = pd.DataFrame(index=df.index)
    for col in df["airline_code"].fillna("UNK").unique():
        items[f"Airline={col}"] = df["airline_code"] == col
    for col in ["SGN", "HAN", "DAD", "CXR", "PQC"]:
        items[f"Airport={col}"] = df["airport"] == col
    if "time_slot" in df.columns:
        for slot in df["time_slot"].dropna().unique():
            items[f"Slot={slot}"] = df["time_slot"] == slot
    items["Peak=Yes"] = df.get("is_peak_hour", 0).astype(int) == 1
    items["HeavyTraffic=Yes"] = df.get("is_heavy_traffic", 0).astype(int) == 1
    items["Weekend=Yes"] = df.get("is_weekend_flag", 0).astype(int) == 1
    items["Rain=Yes"] = df.get("is_rain", 0).astype(int) == 1
    items["HighWind=Yes"] = df.get("is_high_wind", 0).astype(int) == 1
    items["Domestic=Yes"] = df.get("is_domestic_flag", 0).astype(int) == 1
    items["Status=delayed"] = df["status_clean"] == "delayed"
    items["Status=cancelled"] = df["status_clean"] == "cancelled"
    return items.astype(bool)


def _prune_redundant_rules(df_rules: pd.DataFrame) -> pd.DataFrame:
    """Remove redundant superset rules whose lift & confidence fail to beat a simpler subset rule."""
    if df_rules.empty:
        return df_rules
    kept = []
    # Sort shortest antecedents first, then highest lift
    ordered = df_rules.sort_values(["ante_len", "lift", "confidence"], ascending=[True, False, False])
    selected = []
    for _, row in ordered.iterrows():
        ant_set = row["_ant_set"]
        cons = row["consequent"]
        dominated = False
        for prev in selected:
            if prev["consequent"] == cons and prev["_ant_set"].issubset(ant_set) and prev["_ant_set"] != ant_set:
                if row["lift"] <= prev["lift"] * 1.02 and row["confidence"] <= prev["confidence"] * 1.02:
                    dominated = True
                    break
        if not dominated:
            selected.append(row)
            kept.append(row)
    return pd.DataFrame(kept) if kept else df_rules.iloc[0:0]


def _evaluate_rule_on_tx(tx_df: pd.DataFrame, ant_items: frozenset, cons_item: str) -> tuple[float, float, float, int]:
    n = len(tx_df)
    if n == 0:
        return 0.0, 0.0, 1.0, 0
    ant_mask = tx_df[list(ant_items)].all(axis=1)
    cons_mask = tx_df[cons_item]
    sup_cons = float(cons_mask.mean())
    both = int((ant_mask & cons_mask).sum())
    sup_both = both / n
    conf = (both / int(ant_mask.sum())) if int(ant_mask.sum()) > 0 else 0.0
    lift = (conf / sup_cons) if sup_cons > 0 else 1.0
    return round(sup_both * 100, 2), round(conf * 100, 2), round(lift, 2), both


def mine_association_rules(gold_df: pd.DataFrame) -> dict:
    """Mine non-redundant FP-Growth operational risk rules on Train and validate stability on Holdout Test."""
    cfg = EXPERIMENTS_CFG["unsupervised"]["fpgrowth"]
    max_len = int(cfg["max_len"])
    targets = cfg["targets"]

    if "split_role" in gold_df.columns:
        tr_df = gold_df[gold_df["split_role"] != "test"].copy()
        te_df = gold_df[gold_df["split_role"] == "test"].copy()
    else:
        cut = int(len(gold_df) * 0.8)
        tr_df = gold_df.iloc[:cut].copy()
        te_df = gold_df.iloc[cut:].copy()

    tx_tr = _build_transaction_df(tr_df)
    tx_te = _build_transaction_df(te_df)

    all_kept = []
    stats_per_target = {}
    for cons_item, tc in targets.items():
        min_sup = float(tc["min_sup"])
        min_conf = float(tc["min_conf"])
        min_lift = float(tc["min_lift"])

        freq = fpgrowth(tx_tr, min_support=min_sup, use_colnames=True, max_len=max_len)
        if freq.empty:
            stats_per_target[cons_item] = {"raw_rules": 0, "pruned_rules": 0, "holdout_retained": 0}
            continue
        raw = association_rules(freq, metric="confidence", min_threshold=min_conf)
        cand = raw[
            (raw["consequents"] == frozenset([cons_item]))
            & (raw["lift"] >= min_lift)
            & (~raw["antecedents"].apply(lambda s: any(x.startswith("Status=") for x in s)))
        ].copy()
        if cand.empty:
            stats_per_target[cons_item] = {"raw_rules": 0, "pruned_rules": 0, "holdout_retained": 0}
            continue

        cand["_ant_set"] = cand["antecedents"]
        cand["ante_len"] = cand["antecedents"].apply(len)
        cand["consequent"] = cons_item
        pruned = _prune_redundant_rules(cand).sort_values(["lift", "confidence"], ascending=False).head(30)

        retained_cnt = 0
        for _, r in pruned.iterrows():
            ant_items = r["_ant_set"]
            tr_sup = round(float(r["support"] * 100), 2)
            tr_conf = round(float(r["confidence"] * 100), 2)
            tr_lift = round(float(r["lift"]), 2)
            te_sup, te_conf, te_lift, te_cnt = _evaluate_rule_on_tx(tx_te, ant_items, cons_item)
            ok_holdout = bool(te_lift >= 1.05 and te_cnt >= 2)
            if ok_holdout:
                retained_cnt += 1
            all_kept.append({
                "antecedents": " + ".join(sorted(ant_items)),
                "consequent": cons_item,
                "support_pct": tr_sup,
                "confidence_pct": tr_conf,
                "lift": tr_lift,
                "train_count": int(round(float(r["support"] * len(tr_df)))),
                "test_support_pct": te_sup,
                "test_confidence_pct": te_conf,
                "test_lift": te_lift,
                "test_count": te_cnt,
                "holdout_validated": ok_holdout,
                "ante_len": int(r["ante_len"]),
            })
        stats_per_target[cons_item] = {
            "raw_rules": int(len(cand)),
            "pruned_rules": int(len(pruned)),
            "holdout_retained": int(retained_cnt),
        }

    rules_df = pd.DataFrame(all_kept)
    if not rules_df.empty:
        rules_df = rules_df.sort_values(["consequent", "lift", "confidence_pct"], ascending=[True, False, False])
        rules_df.to_csv(FILE_RULES, index=False, encoding="utf-8-sig")

    return {
        "total_non_redundant_rules": int(len(rules_df)),
        "train_transactions": int(len(tx_tr)),
        "holdout_transactions": int(len(tx_te)),
        "per_target_summary": stats_per_target,
        "top_rules": rules_df.head(20).to_dict("records") if not rules_df.empty else [],
    }
