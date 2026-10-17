import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from core.config import get_settings
from core.logging_config import configure_logging
from core.sentry import init_sentry

settings = get_settings()
configure_logging(use_json=settings.is_production)
init_sentry()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from api.routes.destinations import router as destinations_router
from api.routes.health import router as health_router
from api.routes.recommendations import limiter, router as recommendations_router
from api.v1.admin import router as v1_admin_router
from api.v1.destinations import router as v1_destinations_router
from api.v1.me import router as v1_me_router
from api.v1.meta import router as v1_meta_router
from api.v1.recommendations import router as v1_recommendations_router
from core.config import init_firebase_admin
from core.exceptions import global_exception_handler, http_exception_handler
from core.middleware import RequestLoggingMiddleware
from db.session import dispose_engine, init_engine
from services import snapshot_store
from services.data_cache import load_catalog

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    init_firebase_admin()
    watcher = None
    if settings.database_url:
        init_engine(settings.database_url)
        statuses = settings.catalog_statuses_list
        try:
            await snapshot_store.refresh(statuses)
        except Exception:
            # Readiness stays false until the watcher manages a load.
            logger.exception("initial engine snapshot load failed")
        watcher = asyncio.create_task(
            snapshot_store.watch(statuses, settings.catalog_refresh_seconds)
        )
    load_catalog()
    logger.info("Travely API started env=%s docs_enabled=%s", settings.env, settings.docs_enabled)
    yield
    if watcher is not None:
        watcher.cancel()
    await dispose_engine()
    logger.info("Travely API shutting down")


app = FastAPI(
    title="Travely API",
    description="API for personalized travel recommendations in Nigeria",
    version="2.0.0",
    docs_url="/docs" if settings.docs_enabled else None,
    redoc_url="/redoc" if settings.docs_enabled else None,
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, global_exception_handler)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID", "Idempotency-Key"],
)

app.add_middleware(SlowAPIMiddleware)

app.include_router(health_router)
app.include_router(recommendations_router)
app.include_router(destinations_router)
app.include_router(v1_meta_router)
app.include_router(v1_recommendations_router)
app.include_router(v1_admin_router)
app.include_router(v1_destinations_router)
app.include_router(v1_me_router)
