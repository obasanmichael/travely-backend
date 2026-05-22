from __future__ import annotations

import hashlib
import logging
import time
import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start = time.perf_counter()

        response = await call_next(request)

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        uid_hash = None
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token_fragment = auth_header[7:][:16]
            uid_hash = hashlib.sha256(token_fragment.encode()).hexdigest()[:12]

        logger.info(
            "request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
                "uid_hash": uid_hash,
            },
        )
        response.headers["X-Request-ID"] = request_id
        return response
