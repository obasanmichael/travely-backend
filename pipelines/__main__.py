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


JOBS = {
    "seed-geography": _seed_geography,
    "seed-taxonomy": _seed_taxonomy,
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="pipelines")
    parser.add_argument("job", choices=sorted(JOBS))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    asyncio.run(JOBS[args.job]())


if __name__ == "__main__":
    main()
