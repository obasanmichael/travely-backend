import pytest

from db.url import to_asyncpg_url


def test_neon_url_is_normalised_for_asyncpg():
    url = "postgresql://u:p@ep-x.eu-west-2.aws.neon.tech/travely?sslmode=require&channel_binding=require"
    assert to_asyncpg_url(url) == "postgresql+asyncpg://u:p@ep-x.eu-west-2.aws.neon.tech/travely?ssl=require"


def test_local_url_without_ssl_is_left_plain():
    assert to_asyncpg_url("postgres://u:p@db:5432/travely") == "postgresql+asyncpg://u:p@db:5432/travely"


def test_unsupported_scheme_is_rejected():
    with pytest.raises(ValueError):
        to_asyncpg_url("mysql://u:p@db/x")


def test_empty_url_is_rejected():
    with pytest.raises(ValueError):
        to_asyncpg_url("")
