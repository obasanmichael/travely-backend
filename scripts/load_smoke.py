#!/usr/bin/env python3
"""Lightweight load smoke test — run against a live API base URL."""
from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

DEFAULT_PAYLOAD = {
    "budget": 20000,
    "destination_type": "Nature/Adventure",
    "activity_type": "Hiking",
}


def run_once(client: httpx.Client, url: str) -> float:
    start = time.perf_counter()
    response = client.post(url, json=DEFAULT_PAYLOAD, timeout=30.0)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.raise_for_status()
    return elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Travely recommendations load smoke test")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    url = f"{args.base_url.rstrip('/')}/recommendations"
    latencies: list[float] = []

    with httpx.Client() as client:
        health = client.get(f"{args.base_url.rstrip('/')}/health", timeout=10.0)
        health.raise_for_status()

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(run_once, client, url) for _ in range(args.requests)]
            for future in as_completed(futures):
                latencies.append(future.result())

    p95_index = max(0, int(len(latencies) * 0.95) - 1)
    sorted_latencies = sorted(latencies)
    print(f"requests={len(latencies)} concurrency={args.concurrency}")
    print(f"mean_ms={statistics.mean(latencies):.1f}")
    print(f"p95_ms={sorted_latencies[p95_index]:.1f}")
    print(f"max_ms={max(latencies):.1f}")


if __name__ == "__main__":
    main()
