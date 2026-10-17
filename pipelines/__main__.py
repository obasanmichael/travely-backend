"""Run a data pipeline job: python -m pipelines <job>"""
from __future__ import annotations

import argparse
import asyncio
import logging

from pipelines.db import session_scope

logger = logging.getLogger("pipelines")


async def _seed_geography() -> None:
    from pipelines.geography import fetch_adm1, seed_geography

    collection = fetch_adm1()
    async with session_scope() as session:
        count = await seed_geography(session, collection)
    logger.info("seeded %d states", count)


async def _seed_taxonomy() -> None:
    from pipelines.taxonomy import seed_taxonomy

    async with session_scope() as session:
        count = await seed_taxonomy(session)
    logger.info("seeded %d tags", count)


async def _ingest_wikidata() -> None:
    from pipelines.wikidata import fetch_candidates, upsert_candidates

    candidates = fetch_candidates()
    async with session_scope() as session:
        report = await upsert_candidates(session, candidates.values())
    logger.info("wikidata ingest: %s", report)


async def _attach_images() -> None:
    from pipelines.commons import attach_primary_images

    async with session_scope() as session:
        report = await attach_primary_images(session)
    logger.info("commons images: %s", report)


async def _fill_climate() -> None:
    from pipelines.climate import fill_climate

    async with session_scope() as session:
        report = await fill_climate(session)
    logger.info("climate: %s", report)


JOBS = {
    "fill-climate": _fill_climate,
    "attach-images": _attach_images,
    "ingest-wikidata": _ingest_wikidata,
    "seed-geography": _seed_geography,
    "seed-taxonomy": _seed_taxonomy,
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="pipelines")
    parser.add_argument("job", choices=sorted(JOBS))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # httpx logs every request URL at INFO, which drowns the job's own summary.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(JOBS[args.job]())


if __name__ == "__main__":
    main()
