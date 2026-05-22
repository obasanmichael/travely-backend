from __future__ import annotations

import logging
import os

from core.config import get_settings

logger = logging.getLogger(__name__)


def init_sentry() -> None:
    settings = get_settings()
    dsn = os.getenv("SENTRY_DSN", "").strip()
    if not dsn:
        logger.info("Sentry disabled — SENTRY_DSN not set")
        return

    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.starlette import StarletteIntegration

    release = os.getenv("SENTRY_RELEASE") or os.getenv("GIT_SHA") or "travely-api@dev"
    traces_sample_rate = 0.2 if settings.is_production else 1.0

    sentry_sdk.init(
        dsn=dsn,
        environment=settings.env,
        release=release,
        integrations=[
            StarletteIntegration(),
            FastApiIntegration(),
        ],
        traces_sample_rate=traces_sample_rate,
        send_default_pii=False,
        enable_logs=True,
    )
    logger.info("Sentry initialized for release=%s env=%s", release, settings.env)
