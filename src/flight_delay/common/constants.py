# -*- coding: utf-8 -*-
"""Centralized configuration loader backed by configs/*.yaml."""
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT_DIR = Path(__file__).resolve().parents[3]
CONFIGS_DIR = ROOT_DIR / "configs"


@lru_cache(maxsize=16)
def load_yaml_config(filename: str) -> Dict[str, Any]:
    """Load and cache a YAML configuration file from configs/."""
    path = CONFIGS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing configuration file: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def reload_configs() -> None:
    load_yaml_config.cache_clear()


PROJECT_CFG = load_yaml_config("project.yaml")
SOURCES_CFG = load_yaml_config("sources.yaml")
AIRPORTS_CFG = load_yaml_config("airports.yaml")
AIRLINES_CFG = load_yaml_config("airlines.yaml")
SCHEMA_CFG = load_yaml_config("schema.yaml")
FEATURES_CFG = load_yaml_config("features.yaml")
EXPERIMENTS_CFG = load_yaml_config("experiments.yaml")
MONITORING_CFG = load_yaml_config("monitoring.yaml")

# Canonical filesystem layout
PATHS = PROJECT_CFG["paths"]
RAW_DIR = ROOT_DIR / PATHS["raw_dir"]
RAW_FIDS_DIR = ROOT_DIR / PATHS["raw_fids_dir"]
RAW_AVIATIONSTACK_DIR = ROOT_DIR / PATHS["raw_aviationstack_dir"]
RAW_WEATHER_DIR = ROOT_DIR / PATHS["raw_weather_dir"]
DEBUG_DIR = ROOT_DIR / PATHS["debug_dir"]
BRONZE_DIR = ROOT_DIR / PATHS.get("bronze_dir", "data/bronze")
SILVER_DIR = ROOT_DIR / PATHS.get("silver_dir", "data/silver")
GOLD_DIR = ROOT_DIR / PATHS.get("gold_dir", "data/gold")
INTERIM_DIR = SILVER_DIR
PROCESSED_DIR = GOLD_DIR
DATA_MANIFESTS_DIR = ROOT_DIR / PATHS["data_manifests_dir"]

EXPERIMENTS_DIR = ROOT_DIR / PATHS["experiments_dir"]
ARTIFACTS_DIR = ROOT_DIR / PATHS["artifacts_dir"]
MODELS_DIR = ROOT_DIR / PATHS["models_dir"]
ENCODERS_DIR = ROOT_DIR / PATHS["encoders_dir"]
SCALERS_DIR = ROOT_DIR / PATHS["scalers_dir"]
FIGURES_DIR = ROOT_DIR / PATHS["figures_dir"]
UNSUPERVISED_ARTIFACTS_DIR = ROOT_DIR / PATHS.get("unsupervised_dir", "artifacts/unsupervised")
ARTIFACT_MANIFESTS_DIR = ROOT_DIR / PATHS["artifact_manifests_dir"]
REPORTS_DIR = ROOT_DIR / PATHS["reports_dir"]
LOG_DIR = ROOT_DIR / PATHS["logs_dir"]
WEB_DIR = ROOT_DIR / PATHS["web_dir"]

# Key files (compatibility & Raw / Bronze / Silver / Gold boundaries)
DATA_DIR = ROOT_DIR / "data"
USAGE_FILE = ROOT_DIR / SOURCES_CFG["aviationstack"]["usage_file"]
FILE_WEATHER = RAW_WEATHER_DIR / "weather_hourly.csv"
FILE_WEATHER_LEGACY = DATA_DIR / "weather_hourly.csv"
FILE_STANDARDIZED = BRONZE_DIR / "standardized_snapshots.csv"
FILE_SILVER = SILVER_DIR / "flights_silver.csv"
FILE_EXCLUDED = SILVER_DIR / "excluded_outcomes.csv"
FILE_CLEANED = FILE_SILVER
FILE_GOLD = GOLD_DIR / "flights_gold.csv"
FILE_FEATURES = GOLD_DIR / "flights_features.csv"
FILE_RULES = UNSUPERVISED_ARTIFACTS_DIR / "mined_association_rules.csv"
FILE_CAUSAL_HISTORY_META = GOLD_DIR / "causal_history_metadata.json"

# Airport & airline domain constants
UA = SOURCES_CFG["user_agent"]
SITES = SOURCES_CFG["fids_sites"]
AIRPORT_COORD = {k: (v["lat"], v["lon"]) for k, v in AIRPORTS_CFG["monitored_airports"].items()}
CITY_IATA = AIRPORTS_CFG["city_to_iata"]
VN_AIRPORTS = set(AIRPORT_COORD) | set(CITY_IATA.values())
GOLDEN_TRIANGLE = {tuple(pair) for pair in AIRPORTS_CFG["golden_triangle_routes"]}

VN_AIRLINES = set(AIRLINES_CFG["vn_airlines"])
AIRLINE_NAME_CODE = AIRLINES_CFG["airline_name_to_code"]
CARRIER_TYPE = AIRLINES_CFG["carrier_categories"]
OFFICIAL_STATS = {
    k: {
        **({"delay": v["delay_pct"]} if "delay_pct" in v else {}),
        **({"cancel": v["cancel_pct"]} if "cancel_pct" in v else {}),
        **({"otp": v["otp_pct"]} if "otp_pct" in v else {}),
    }
    for k, v in AIRLINES_CFG["caav_official_benchmark"].items()
}

# Schemas & prediction horizon
FIELDS = list(SCHEMA_CFG["bronze"]["standardized_fields"])
WX_COLS = list(SCHEMA_CFG["bronze"]["weather_columns"])
CLEANED_FIELDS = list(SCHEMA_CFG["silver"]["cleaned_24_columns"])
EARLY_LIMIT = float(SCHEMA_CFG["silver"]["early_departure_limit_minutes"])
DELAY_THRESHOLD_MIN = int(SCHEMA_CFG["silver"]["delay_threshold_minutes"])
VALID_OUTCOME_LABELS = list(SCHEMA_CFG["silver"]["valid_outcome_labels"])

LABEL_MAP_3CLASS = {k: int(v) for k, v in SCHEMA_CFG["gold"]["primary_classification_labels"].items()}
LABEL_MAP_BINARY = {k: int(v) for k, v in SCHEMA_CFG["gold"]["binary_ablation_labels"].items()}
FORBIDDEN_POST_FLIGHT_COLUMNS = list(SCHEMA_CFG["gold"]["forbidden_post_flight_columns"])
PRIMARY_HORIZON_HOURS = int(PROJECT_CFG["prediction_horizon"]["primary_hours"])
ABLATION_HORIZONS = [int(h) for h in PROJECT_CFG["prediction_horizon"]["ablation_hours"]]
RANDOM_SEED = int(PROJECT_CFG["project"]["random_seed"])
RANDOM_STATE = RANDOM_SEED
