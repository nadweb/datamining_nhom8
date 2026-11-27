# FastAPI Inference & Analytics REST API (`src/flight_delay/api/`)

Serve locally or via Docker on `http://127.0.0.1:5000` (`python main.py serve`). Interactive OpenAPI documentation is automatically served at `/docs`.

---

## 1. Endpoints Overview

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web Dashboard (Simulator + Audit + EDA Matrix + Models + Clusters + Rules + Drift) |
| `GET` | `/presentation` | Interactive 20-Slide Academic Presentation |
| `GET` | `/health` | Service health & approved artifact integrity check (`approved_classifier`, `approved_regressor`, `dataset_version`) |
| `POST` | `/api/predict` | Fail-closed pre-flight 3-class ICAO + MAE delay prediction, operational factors, matched cluster & FP-Growth rules |
| `GET` | `/api/fids-flights` | Latest deduplicated FIDS/Gold flight records for quick-load simulator presets |
| `GET` | `/api/audit-flights` | 80 Holdout Test flights audited end-to-end against `predict_risk()` |
| `GET` | `/api/eda` | Full EDA, raw provenance, schema matrix, Pearson/Spearman heatmaps & Cramér's V |
| `GET` | `/api/stats` | 3-Class Primary, Binary Ablation, Horizon Ablation, Regression MAE, Statistical Tests & SHAP |
| `GET` | `/api/backtest` | Subgroup holdout evaluation (`by_airport`, `by_airline`, peak, weather), Permutation, McNemar & Error Analysis |
| `GET` | `/api/clusters` | K-Means (`k=2..8` Sweep + Seed ARI), PCA 3D variance/loadings & Isolation Forest anomaly slices |
| `GET` | `/api/rules` | Non-redundant FP-Growth risk rules with Train & Holdout Test metrics |
| `GET` | `/api/drift` | 4-Tier Monitoring (Data Quality, Feature PSI + KS-test, Concept Drift, Alert flags) |
| `GET` | `/api/manifest` | `LOCK_pipeline_manifest.json` + Model Registry + Dataset Manifest SHA-256 lineage |
| `GET` | `/api/governance` | Registry lifecycle states & artifact lineage snapshot |
| `GET` | `/api/experiments` | Structured experiment logs from `experiments/` |

---

## 2. Example `POST /api/predict` Payload & Guardrails

```json
{
  "airline_code": "VJ",
  "airport": "SGN",
  "direction": "DEPARTURES",
  "route_type": "domestic",
  "scheduled_time": "19:30",
  "flight_date": "2026-11-06",
  "origin": "SGN",
  "destination": "HAN",
  "precipitation": 15.0,
  "wind_gusts_10m": 45.0,
  "visibility": 2000.0,
  "weather_code": 95,
  "airport_load": 35,
  "prediction_time": "2026-11-06 16:30:00"
}
```

**Fail-Closed Validation (`422` / `503`):**
- Supplying `prediction_time` later than `scheduled_dt` raises `422 Unprocessable Entity` (`InvalidPredictionTimeError`).
- Supplying post-flight columns (`delay_min`, `actual_time`, `status_clean`) raises `422` (`LeakageError`).
- Missing/tampered SHA-256 checksums or unapproved bundle metadata raises `503 Service Unavailable`.
