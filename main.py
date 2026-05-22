import logging

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
from core.exceptions import global_exception_handler, http_exception_handler
from core.middleware import RequestLoggingMiddleware
from services.data_cache import load_catalog

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Travely API",
    description="API for personalized travel recommendations in Nigeria",
    version="2.0.0",
    docs_url="/docs" if settings.docs_enabled else None,
    redoc_url="/redoc" if settings.docs_enabled else None,
)


@app.on_event("startup")
async def startup_load_catalog() -> None:
    load_catalog()
    logger.info("Travely API started env=%s docs_enabled=%s", settings.env, settings.docs_enabled)


app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, global_exception_handler)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)

app.add_middleware(SlowAPIMiddleware)

app.include_router(health_router)
app.include_router(recommendations_router)
app.include_router(destinations_router)
