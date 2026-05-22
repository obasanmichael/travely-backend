from __future__ import annotations

from typing import Dict

from fastapi import APIRouter

router = APIRouter(tags=["Health"])


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
