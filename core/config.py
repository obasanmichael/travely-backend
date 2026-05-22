import json
import logging
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: str = "development"
    allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    firebase_project_id: str = ""
    google_application_credentials_json: str = ""
    rate_limit_per_minute: int = 30
    docs_enabled: bool = True
    auth_disabled: bool = False

    @property
    def allowed_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]

    @property
    def is_production(self) -> bool:
        return self.env.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def init_firebase_admin() -> None:
    """Initialize Firebase Admin SDK from env credentials."""
    import firebase_admin
    from firebase_admin import credentials

    if firebase_admin._apps:
        return

    settings = get_settings()

    if settings.google_application_credentials_json:
        cred_dict = json.loads(settings.google_application_credentials_json)
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(
            cred, {"projectId": settings.firebase_project_id or cred_dict.get("project_id")}
        )
        logger.info("Firebase Admin initialized from service account JSON")
        return

    if settings.firebase_project_id:
        try:
            firebase_admin.initialize_app(options={"projectId": settings.firebase_project_id})
            logger.info("Firebase Admin initialized with project ID only")
            return
        except Exception as exc:
            logger.warning("Firebase Admin init failed: %s", exc)

    if settings.auth_disabled and not settings.is_production:
        logger.warning("Firebase Admin not configured — auth disabled for local development")
        return

    logger.warning(
        "Firebase Admin not configured. Set GOOGLE_APPLICATION_CREDENTIALS_JSON "
        "or AUTH_DISABLED=true for local development."
    )
