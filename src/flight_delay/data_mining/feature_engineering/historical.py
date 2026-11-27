# -*- coding: utf-8 -*-
"""[STEP 3] Strictly Causal Historical Delay Aggregations (< prediction_time)."""
from bisect import bisect_left
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def compute_outcome_known_at(df: pd.DataFrame) -> pd.Series:
    """Estimate earliest wall-clock timestamp when each flight's outcome became observable."""
    hm = df["scheduled_time"].astype(str).str.extract(r"(\d{1,2}):(\d{2})").astype(float)
    sched_dt = pd.to_datetime(df["flight_date"], errors="coerce") + pd.to_timedelta(
        hm[0].fillna(12.0) * 60 + hm[1].fillna(0.0), unit="m"
    )
    delay = pd.to_numeric(df.get("delay_min", np.nan), errors="coerce").fillna(30.0).clip(lower=0.0, upper=720.0)
    is_canc = df.get("status_clean", pd.Series("", index=df.index)).astype(str).str.lower().eq("cancelled")
    # Cancelled is observed at least at scheduled departure; completed at scheduled + max(delay, 0)
    offset_min = np.where(is_canc, 30.0, delay)
    return sched_dt + pd.to_timedelta(offset_min, unit="m")


@dataclass
class CausalHistoryStore:
    """Point-in-time smoothed delay rate store guaranteeing zero future leakage."""

    prior_smooth: float
    fallback_prior: float
    global_ts: List[int]
    global_bad_cum: List[int]
    airline_index: Dict[str, Tuple[List[int], List[int]]]
    airport_index: Dict[str, Tuple[List[int], List[int]]]
    airport_median_load: Dict[str, float]
    global_median_load: float
    max_observed_ts: Optional[str] = None

    @classmethod
    def from_observations(
        cls,
        obs_df: pd.DataFrame,
        prior_smooth: float = 20.0,
        fallback_prior: float = 0.28,
    ) -> "CausalHistoryStore":
        if obs_df is None or obs_df.empty:
            return cls(
                prior_smooth=prior_smooth,
                fallback_prior=fallback_prior,
                global_ts=[],
                global_bad_cum=[],
                airline_index={},
                airport_index={},
                airport_median_load={},
                global_median_load=20.0,
                max_observed_ts=None,
            )

        work = obs_df.copy()
        work["_known_at"] = pd.to_datetime(
            work["outcome_known_at"] if "outcome_known_at" in work.columns else compute_outcome_known_at(work),
            errors="coerce",
        )
        work = work[work["_known_at"].notna()].sort_values("_known_at", kind="stable").copy()
        work["_ts_ns"] = work["_known_at"].dt.as_unit("ns").astype("int64")
        work["_is_bad"] = work["status_clean"].astype(str).str.lower().isin(["delayed", "cancelled"]).astype(int)

        global_ts = work["_ts_ns"].tolist()
        global_bad_cum = work["_is_bad"].cumsum().tolist()

        airline_index: Dict[str, Tuple[List[int], List[int]]] = {}
        for al, g in work.groupby("airline_code"):
            airline_index[str(al).upper()] = (g["_ts_ns"].tolist(), g["_is_bad"].cumsum().tolist())

        airport_index: Dict[str, Tuple[List[int], List[int]]] = {}
        for ap, g in work.groupby("airport"):
            airport_index[str(ap).upper()] = (g["_ts_ns"].tolist(), g["_is_bad"].cumsum().tolist())

        loads = pd.to_numeric(work.get("airport_load", pd.Series(dtype=float)), errors="coerce")
        g_med = float(loads.median()) if loads.notna().any() else 20.0
        ap_meds = (
            work.assign(_ld=loads)
            .groupby("airport")["_ld"]
            .median()
            .dropna()
            .astype(float)
            .to_dict()
        )
        ap_meds = {str(k).upper(): v for k, v in ap_meds.items()}

        return cls(
            prior_smooth=prior_smooth,
            fallback_prior=fallback_prior,
            global_ts=global_ts,
            global_bad_cum=global_bad_cum,
            airline_index=airline_index,
            airport_index=airport_index,
            airport_median_load=ap_meds,
            global_median_load=g_med,
            max_observed_ts=str(work["_known_at"].max()) if not work.empty else None,
        )

    def _causal_rate(self, series_map: Dict[str, Tuple[List[int], List[int]]], key: str, ts_ns: int, prior: float) -> float:
        entry = series_map.get(str(key).upper())
        if not entry:
            return round(prior, 4)
        ts_list, cum_bad = entry
        idx = bisect_left(ts_list, ts_ns)  # strictly < prediction_time
        n = idx
        bad = cum_bad[idx - 1] if idx > 0 else 0
        return round((bad + self.prior_smooth * prior) / (n + self.prior_smooth), 4)

    def global_prior_before(self, ts_ns: int) -> float:
        if not self.global_ts:
            return self.fallback_prior
        idx = bisect_left(self.global_ts, ts_ns)
        if idx == 0:
            return self.fallback_prior
        return float(self.global_bad_cum[idx - 1] / idx)

    def rates_for_row(self, airline_code: str, airport: str, prediction_time) -> Tuple[float, float]:
        ts = pd.Timestamp(prediction_time)
        if pd.isna(ts):
            return self.fallback_prior, self.fallback_prior
        ts_ns = int(ts.as_unit("ns").value)
        g_prior = self.global_prior_before(ts_ns)
        al_rate = self._causal_rate(self.airline_index, airline_code, ts_ns, g_prior)
        ap_rate = self._causal_rate(self.airport_index, airport, ts_ns, g_prior)
        return al_rate, ap_rate

    def enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        al_rates = []
        ap_rates = []
        for al, ap, ptime in zip(df["airline_code"], df["airport"], df["prediction_time"]):
            r_al, r_ap = self.rates_for_row(al, ap, ptime)
            al_rates.append(r_al)
            ap_rates.append(r_ap)
        df["historical_airline_delay_rate"] = al_rates
        df["historical_airport_delay_rate"] = ap_rates
        return df

    def to_dict(self) -> dict:
        return {
            "prior_smooth": self.prior_smooth,
            "fallback_prior": self.fallback_prior,
            "global_ts": self.global_ts,
            "global_bad_cum": self.global_bad_cum,
            "airline_index": {k: [list(v[0]), list(v[1])] for k, v in self.airline_index.items()},
            "airport_index": {k: [list(v[0]), list(v[1])] for k, v in self.airport_index.items()},
            "airport_median_load": self.airport_median_load,
            "global_median_load": self.global_median_load,
            "max_observed_ts": self.max_observed_ts,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CausalHistoryStore":
        return cls(
            prior_smooth=float(payload.get("prior_smooth", 20.0)),
            fallback_prior=float(payload.get("fallback_prior", 0.28)),
            global_ts=list(payload.get("global_ts", [])),
            global_bad_cum=list(payload.get("global_bad_cum", [])),
            airline_index={k: (list(v[0]), list(v[1])) for k, v in (payload.get("airline_index") or {}).items()},
            airport_index={k: (list(v[0]), list(v[1])) for k, v in (payload.get("airport_index") or {}).items()},
            airport_median_load={k: float(v) for k, v in (payload.get("airport_median_load") or {}).items()},
            global_median_load=float(payload.get("global_median_load", 20.0)),
            max_observed_ts=payload.get("max_observed_ts"),
        )
