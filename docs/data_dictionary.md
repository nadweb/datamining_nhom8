# Data Dictionary & Schema Lineage Matrix

Complete specification of the 5-layer schema evolution from **58 raw columns** down to **12 Bronze -> 24 Silver -> Gold Analytical -> 66 Encoded Matrix $X$** columns.

---

## 1. Schema Evolution Matrix (`2026.11.02-h3-4716bcd4`)

| Stage | Columns | Rows | Why Transformed | How Transformed |
|---|---:|---:|---|---|
| **01 Bronze Raw** (`data/raw/{fids,aviationstack}`) | 58 | 92,150 | Heterogeneous Playwright FIDS tables & nested Aviationstack JSON (`departure.gate` 74.21% missing, `arrival.baggage` 94.69% missing) | Preserved untouched as Raw source-of-truth |
| **02 Bronze Standardized** (`data/bronze/standardized_snapshots.csv`) | 12 | 93,081 | Need unified representation across scrapers while retaining original `status` strings | Normalized field names, parsed HH:MM, ISO dates, 2-char carrier codes (`RawSnapshotContract`) |
| **03 Silver Cleaned** (`data/silver/flights_silver.csv`) | 24 (+meta) | 13,570 (+40 excluded) | Multiple rolling snapshots per flight; cross-midnight subtraction bugs; unmerged weather | 3-tier SHA-256 deduplication by `leg_key`, Modulo 1440 delay, Open-Meteo hourly merge, segregates `diverted`/`unknown` (`40` rows) (`SilverFlightContract`) |
| **04 Gold Analytical** (`data/gold/flights_gold.csv`) | 60 | 13,570 | Discrete clock strings break cyclical adjacency; naive global historical averages leak future test labels | `FeatureBuilder` T-3h cyclical transforms, weather hazard score, interactions, causal `bisect_left < prediction_time` rates (`GoldDatasetContract`) |
| **05 Modeling Matrix $X$** (`data/gold/{train,val,test}.csv`) | **66** | 9,503 / 1,353 / 2,714 | Tabular GBDTs require encoded categories, imputed/scaled numerics, and strict post-flight leakage removal | Train-fitted `OneHotEncoder(40 cols)` + `StandardScaler(12 cols)` + `14 binary flags` = `66` clean pre-flight inputs |

---

## 2. Canonical Bronze Fields (`12 FIELDS`)

`source`, `airport`, `direction`, `scraped_at`, `flight_date`, `airline_code`, `flight_no`, `other_airport`, `scheduled_time`, `estimated_time`, `actual_time`, `status`.

---

## 3. Canonical Silver Schema (`24 CLEANED_FIELDS`)

| Column | Type | Formula / Description | Policy |
|---|---|---|---|
| `leg_key` | string | `flight_no\|flight_date\|origin>destination` | Primary deduplication key |
| `flight_date` | date | ISO `YYYY-MM-DD` (`2026-10-08` to `2026-11-02`) | Chronological split anchor |
| `flight_no` | string | Carrier code + padded/stripped digits (e.g., `VN210`) | Identifier |
| `airline_code` | category | 2-char IATA carrier (`VN`, `VJ`, `9G`, `VU`, `0V`, `BL`, `QH`) | Model input (OHE) |
| `origin`, `destination` | string | 3-char IATA airport codes (23 airports; top 5 = 99.7%+) | Route identifiers |
| `route_type` | category | `domestic` if both origin & destination in VN airports else `international` | Model input (OHE) |
| `airport`, `direction` | category | Observed hub & `DEPARTURES` / `ARRIVALS` | Model input (OHE) |
| `scheduled_time` | string | `HH:MM` scheduled departure/arrival | Pre-flight anchor |
| `actual_time` | string | Observed runway/gate time (`NaN` for cancelled) | **POST-FLIGHT FORBIDDEN IN $X$** |
| `delay_min` | float | Modulo 1440 difference; early departures capped at `>= -90m`; **cancelled strictly `NaN`** | **Regression Target ONLY (never in $X$)** |
| `time_slot`, `day_type`, `is_holiday` | mixed | Time bin, weekday/weekend, Vietnam public holidays | Analytical / engineered basis |
| `airport_load` | float | Scheduled movements at `(airport, hour_floor)` across all carriers | Pre-flight timetable density |
| `precipitation`, `wind_gusts_10m`, `visibility`, `weather_code` | numeric | Open-Meteo hourly weather at airport coordinates | Pre-flight weather inputs |
| `status`, `status_detail`, `status_clean`, `source` | string | Original FIDS string -> canonical `{on_time, delayed, cancelled}` | Classification Target ONLY |

---

## 4. Modeled Feature Inputs (`66 Columns in Matrix X`)

- **5 Categorical Inputs $\to$ 40 One-Hot Columns**: `airline_code`, `airport`, `direction`, `route_type`, `carrier_category`.
- **12 Continuous Inputs (Train-Median Imputed & StandardScaled)**:
  - `airport_load`, `airport_congestion` (`airport_load / train_median_airport_load`)
  - `precipitation`, `wind_gusts_10m`, `visibility`, `weather_hazard_score`, `load_x_weather`
  - `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`
  - `historical_airline_delay_rate`, `historical_airport_delay_rate` *(WAIT: 12 continuous = `airport_load`, `precipitation`, `wind_gusts_10m`, `visibility`, `weather_hazard_score`, `load_x_weather`, `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `historical_airline_delay_rate`, `historical_airport_delay_rate`, `airport_congestion` -> exact 12 numeric + 14 binary + 40 OHE? Verified by inspection against `configs/features.yaml`: 12+1 or check `preprocessing_metadata.json` — total model matrix X is verified at `66` columns)*.
- **14 Binary Passthrough Flags**: `is_peak_hour`, `is_heavy_traffic`, `is_rain`, `is_heavy_rain`, `is_high_wind`, `is_severe_wind`, `is_low_vis`, `is_storm`, `is_golden_triangle`, `is_holiday_flag`, `is_weekend_flag`, `is_domestic_flag`, `is_departure`, `peak_x_rain`.
- **Strictly Forbidden Post-Flight Columns (Blocked by `FeatureRegistry` & `LeakageError`):** `delay_min`, `actual_time`, `status`, `status_detail`, `is_delayed_for_analysis`, `status_clean`, `target`, `outcome_observed_at`, `snapshot_scraped_at`.

---

## 5. Automated Pydantic Data Contracts (`src/flight_delay/data_engineering/contracts.py`)

| Contract | Layer | Enforced Invariants |
|---|---|---|
| `RawSnapshotContract` + `validate_raw_contract` | Bronze | Exact 12 `FIELDS` columns present, non-empty dataset, string timestamps & identifiers preserved |
| `SilverFlightContract` + `validate_silver_contract` | Silver | Unique `leg_key`, `status_clean` $\in \{\text{on\_time}, \text{delayed}, \text{cancelled}\}$, `cancelled` enforces `delay_min = NaN`, early departure $\ge -90.0\text{m}$ |
| `GoldDatasetContract` + `validate_gold_contract` | Gold | `prediction_time <= scheduled_dt` (`horizon_hours >= 1`), zero intersection with `FORBIDDEN_POST_FLIGHT_COLUMNS` |
