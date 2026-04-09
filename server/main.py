"""FastAPI application — Bremsstrahlung Calculator.

Entry point: uvicorn server.main:app --reload --port 8001
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

import config

log = logging.getLogger(__name__)

app = FastAPI(
    title="Bremsstrahlung Calculator",
    version="0.1.0",
    description="Thick-target bremsstrahlung spectra based on NASA TN D-4755",
)

app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(config.TEMPLATES_DIR))


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------
@app.get("/")
async def index(request: Request) -> templates.TemplateResponse:  # type: ignore[name-defined]
    """Serve the main SPA shell."""
    return templates.TemplateResponse("index.html", {"request": request})


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict[str, object]:
    """Return service health status."""
    data_exists = config.NASA_DATA_PATH.exists()
    return {
        "status": "ok",
        "data_loaded": data_exists,
        "materials_count": len(config.ALL_MATERIALS),
    }
