from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from core.config import get_settings

logger = logging.getLogger(__name__)
_bearer = HTTPBearer(auto_error=False)


async def verify_firebase_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> dict[str, Any]:
    settings = get_settings()

    if settings.auth_disabled and not settings.is_production:
        return {"uid": "dev-user", "email": "dev@local.test"}

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        from firebase_admin import auth

        # check_revoked: a signed-out or disabled account can't keep using a still-valid token.
        decoded = auth.verify_id_token(credentials.credentials, check_revoked=settings.auth_check_revoked)
        return decoded
    except Exception as exc:
        logger.warning("Token verification failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


async def require_admin(token: dict[str, Any] = Depends(verify_firebase_token)) -> dict[str, Any]:
    """Only accounts with the Firebase custom claim admin=true (set by scripts/set_admin.py)."""
    settings = get_settings()
    if settings.auth_disabled and not settings.is_production:
        return {**token, "admin": True}
    if token.get("admin") is not True:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return token
