from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# asyncpg rejects libpq-only query params that Neon includes in its URLs.
_LIBPQ_ONLY_PARAMS = {"sslmode", "channel_binding"}


def to_asyncpg_url(url: str) -> str:
    """Normalise a Postgres URL (e.g. one copied from Neon) for SQLAlchemy + asyncpg."""
    if not url:
        raise ValueError("DATABASE_URL is empty")

    parts = urlsplit(url)
    scheme = parts.scheme
    if scheme in {"postgres", "postgresql", "postgresql+psycopg", "postgresql+psycopg2"}:
        scheme = "postgresql+asyncpg"
    elif scheme != "postgresql+asyncpg":
        raise ValueError(f"Unsupported database URL scheme: {parts.scheme}")

    params = dict(parse_qsl(parts.query))
    sslmode = params.get("sslmode")
    query = {k: v for k, v in params.items() if k not in _LIBPQ_ONLY_PARAMS}
    if sslmode and sslmode != "disable" and "ssl" not in query:
        query["ssl"] = "require"

    return urlunsplit((scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))
