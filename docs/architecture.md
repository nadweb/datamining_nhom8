# FlightRisk VN — System Architecture (`src/flight_delay/`)

**FlightRisk VN** is a modular, leakage-guarded, production-oriented **7-Step CRISP-DM** machine learning & data mining platform built using a clean `src/` Python package layout and served via **FastAPI + Uvicorn**.

---

## 1. End-to-End 7-Step Architecture

```
01. DATA INGESTION (src/flight_delay/ingestion/)
    FIDS Playwright · Aviationstack REST · Open-Meteo API
    --> Immutable Raw: data/raw/{fids,aviationstack,weather} + data/debug_data/ (2,923 raw captures)
    --> Canonical Bronze (12 FIELDS) -> data/bronze/standardized_snapshots.csv (93,081 rows)

02. DATA ENGINEERING (src/flight_delay/data_engineering/)
    Validation · Cleaning · Modulo 1440 · 3-Tier SHA-256 Deduplication · Weather Integration
    --> Valid Silver: data/silver/flights_silver.csv (13,570 unique legs, 24 CLEANED_FIELDS)
    --> Segregated Ambiguous: data/silver/excluded_outcomes.csv (40 rows: diverted/unknown)
    --> Quality Audit: reports/data_quality/silver_quality_report.json

03. DATA MINING & CAUSAL FEATURE STORE (src/flight_delay/data_mining/)
    Profiling · EDA · Statistical Tests (Mann-Whitney/Kruskal/Cramér's V)
    Causal T-3h History Store (bisect_left < prediction_time) · FeatureBuilder (training-serving parity)
    Chronological Split (70% Train / 10% Val / 20% Holdout Test)
    --> Gold: data/gold/flights_gold.csv + train/val/test.csv (66 encoded model features)
    --> Lineages: data/manifests/dataset_manifest.json & data/gold/causal_history_store.json

04. EXPERIMENTATION & MODELING (src/flight_delay/modeling/)
    Baselines: MajorityClassifier, HeuristicBaseline, MedianBaseline
    Classification (Primary 3-Class ICAO + Binary & T-1h/T-3h/T-6h Ablations): HGB, XGBoost, CatBoost
    Regression (Completed flights only, L1 loss): HGB_MAE, XGBoost_MAE, CatBoost_MAE
    Unsupervised: K-Means (k=4, Seed ARI stability), PCA (3D), Isolation Forest, FP-Growth (Holdout validated)
    Explainability & Calibration: SHAP TreeExplainer, Permutation Importance, Brier / LogLoss Reliability

05. EVALUATION & VALIDATION (src/flight_delay/evaluation/)
    Macro-F1, BalAcc, Per-Class Recall (Delayed & Cancelled separately), PR-AUC, ROC-AUC OvR
    Expanding Temporal Fold CV · Paired Bootstrap CI · McNemar Test · Permutation Test (200x)
    Subgroup Audit (Airport, Airline, Peak, Weather, Route) + Qualitative Error Analysis
    --> reports/experiments/evaluation_report.json

06. MODEL GOVERNANCE (src/flight_delay/governance/)
    Local Experiment Tracker (experiments/) · Model Registry Lifecycle (Candidate->Validated->Approved->Production)
    Versioned Model Bundles (artifacts/models/<name>_<ver>/) with SHA-256 Checksums, Model Cards & Data Lineage
    --> artifacts/manifests/{model_registry.json, data_lineage.json}

07. DEPLOYMENT & MONITORING (src/flight_delay/inference/, monitoring/, api/)
    Fail-Closed predict_risk(flight_data, prediction_time) (checks checksums, approved==True, prediction_time<=sched_dt)
    4-Tier Monitoring: Data Quality, Data Drift (PSI + 2-sample KS + Chi2), Concept Drift, Latency SLO
    FastAPI + Uvicorn Web Dashboard & 20-Slide Presentation
```

---

## 2. Directory Layout

```text
ScrapeData/
├── configs/                  # YAML configurations (project, sources, airports, airlines, schema, features, experiments, monitoring)
├── data/
│   ├── raw/{fids,aviationstack,weather}/   # Immutable Raw sources
│   ├── debug_data/                         # 2,923 archived HTML/JSON captures (untouched)
│   ├── bronze/                             # Canonical 12-field standardized snapshots
│   ├── silver/                             # Valid 24-field Silver legs + excluded_outcomes.csv
│   ├── gold/                               # Gold matrices, splits, causal store, rules & audit sample
│   └── manifests/                          # bronze_manifest.json, dataset_manifest.json
├── src/flight_delay/
│   ├── common/               # Config loader, domain exceptions, hashing, structured logging
│   ├── ingestion/            # Step 1 Collectors & Source Registry
│   ├── data_engineering/     # Step 2 Cleaning, Contracts, Integration & Quality
│   ├── data_mining/          # Step 3 Profiling, EDA, Causal Features, Encoding, Datasets
│   ├── modeling/             # Step 4 Baselines, Classification, Regression, Unsupervised, SHAP, Calibration
│   ├── evaluation/           # Step 5 Metrics, Statistical Tests, Subgroups, Temporal CV, Error Analysis
│   ├── governance/           # Step 6 Experiment Tracker, Registry, Checksums, Model Cards, Lineage
│   ├── inference/            # Step 7 Fail-closed Predict API, Model Loader, Audit Sample Builder
│   ├── monitoring/           # Step 7 Data Quality, PSI + KS Drift, Concept Drift, Alerts
│   ├── visualization/        # Matplotlib/Seaborn figure generation -> artifacts/figures/
│   └── api/                  # FastAPI + Pydantic schemas + REST & UI routes
├── scripts/                  # Standalone stage runners (collect, prepare, build_features, train, evaluate, register_model, run_web)
├── notebooks/                # 00_business_understanding.ipynb .. 07_deployment_monitoring.ipynb
├── artifacts/                # Approved model bundles, checksums, encoders, scalers, manifests, figures
├── reports/                  # Data quality, EDA, experiment benchmarks, monitoring drift reports
├── experiments/              # Structured per-run JSON experiment logs
├── tests/                    # Unit, Data Quality, Leakage Guard, Temporal Integrity, Parity, Evaluation & API/E2E suites
├── web/                      # Frontend HTML/CSS/JS Dashboard & Interactive CRISP-DM Presentation
├── main.py                   # Unified CLI entrypoint (`python main.py <command>`)
├── pyproject.toml            # Packaging, pytest, and ruff lint configuration
├── Dockerfile & docker-compose.yml & Makefile
```
