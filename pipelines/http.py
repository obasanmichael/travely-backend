from __future__ import annotations

import httpx

# Wikimedia and others require an identifying User-Agent with a contact URL.
USER_AGENT = "TravelyDataPipeline/0.1 (+https://github.com/obasanmichael/travely-backend)"


def client(timeout: float = 60.0) -> httpx.Client:
    transport = httpx.HTTPTransport(retries=3)
    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
        follow_redirects=True,
        transport=transport,
    )
