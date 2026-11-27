# -*- coding: utf-8 -*-
"""[STEP 7] FastAPI Application Entrypoint Serving API + Interactive Dashboard + Presentation."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from flight_delay.api.routes import drift, eda, model, predict
from flight_delay.api.schemas.predict import HealthResponse
from flight_delay.common.constants import PROJECT_CFG, WEB_DIR
from flight_delay.inference.model_loader import load_production_artifacts


def create_app() -> FastAPI:
    app = FastAPI(
        title=PROJECT_CFG["project"]["name"],
        version=PROJECT_CFG["project"]["version"],
        description="Vietnamese Airline Flight Data Mining & Pre-Flight Risk Prediction Platform (7-Step CRISP-DM)",
    )

    static_dir = WEB_DIR / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    app.include_router(predict.router)
    app.include_router(eda.router)
    app.include_router(model.router)
    app.include_router(drift.router)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def serve_dashboard():
        idx = Path(WEB_DIR) / "templates" / "index.html"
        return HTMLResponse(idx.read_text(encoding="utf-8"))

    @app.get("/presentation", response_class=HTMLResponse, include_in_schema=False)
    def serve_presentation():
        pres = Path(WEB_DIR) / "templates" / "presentation.html"
        return HTMLResponse(pres.read_text(encoding="utf-8"))

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    def health():
        try:
            bundle = load_production_artifacts()
            return HealthResponse(
                status="ok",
                service=PROJECT_CFG["project"]["short_name"],
                version=PROJECT_CFG["project"]["version"],
                approved_classifier=f"{bundle.cls_meta.get('model_name')}:{bundle.cls_meta.get('version')}",
                approved_regressor=f"{bundle.reg_meta.get('model_name')}:{bundle.reg_meta.get('version')}",
                dataset_version=bundle.cls_meta.get("dataset_version"),
                artifact_integrity=True,
            )
        except Exception as exc:
            return HealthResponse(
                status="degraded",
                service=PROJECT_CFG["project"]["short_name"],
                version=PROJECT_CFG["project"]["version"],
                artifact_integrity=False,
                error=str(exc),
            )

    return app


app = create_app()
