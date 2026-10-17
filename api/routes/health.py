from __future__ import annotations

from typing import Dict

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from db import session as db_session
from services import snapshot_store
from services.data_cache import get_catalog_size

router = APIRouter(tags=["Health"])


@router.get("/livez")
def liveness() -> Dict[str, str]:
    return {"status": "alive"}


@router.get("/readyz")
async def readiness() -> JSONResponse:
    checks = {"catalog": get_catalog_size() > 0}
    if db_session.engine_configured():
        checks["database"] = await db_session.ping()
        snapshot = snapshot_store.current()
        checks["engine_catalog"] = snapshot is not None and snapshot.size > 0
    ready = all(checks.values())
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"status": "ready" if ready else "not_ready", "checks": checks},
    )


# Kept for Render's existing health check until deploys move to Kubernetes.
@router.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "healthy"}


@router.get("/", tags=["Home"])
def home() -> Dict[str, str]:
    return {
        "message": "Welcome to the Travely Recommendation API",
        "documentation": "/docs",
        "version": "2.0.0",
    }
