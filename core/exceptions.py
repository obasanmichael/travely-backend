from __future__ import annotations

import logging
import traceback
from typing import Dict

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from core.config import get_settings

logger = logging.getLogger(__name__)


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    logger.error("Unhandled error on %s %s", request.method, request.url.path)
    logger.error(traceback.format_exc())

    settings = get_settings()
    content: Dict[str, str] = {"detail": "Internal server error"}
    if not settings.is_production:
        content["debug"] = str(exc)

    return JSONResponse(status_code=500, content=content)
