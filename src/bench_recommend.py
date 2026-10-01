"""Measure recommendation latency and throughput.

Steady-state timings start after the index and encoder are loaded.
Title search is a nearest-neighbor lookup. Text search also encodes the query.

    python src/bench_recommend.py
    python src/bench_recommend.py --title-requests 200 --text-requests 40
"""

from __future__ import annotations

import argparse
import random
import sys
import time

import numpy as np

from recommend import Recommender

TEXT_QUERIES = [
    "an insomniac and a soap salesman start an underground fight club",
    "a family of toys comes to life when nobody is watching",
    "a lost clownfish searches the ocean for his son",
    "a heist crew tries to steal from a casino vault",
    "two detectives hunt a serial killer who stages murders",
    "a lonely astronaut stranded far from earth",
    "a young wizard starts school and fights a dark lord",
    "friends take a road trip across the country",
]


def _configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


def _percentile(samples_ms: np.ndarray, pct: float) -> float:
    return float(np.percentile(samples_ms, pct))


def _report(name: str, samples_ms: list[float]) -> None:
    samples = np.asarray(samples_ms, dtype=np.float64)
    elapsed_s = float(samples.sum()) / 1000.0
    throughput = len(samples) / elapsed_s if elapsed_s else 0.0
    print(f"{name}")
    print(f"  requests:   {len(samples)}")
    print(f"  throughput: {throughput:,.1f} queries/sec")
    print(
        "  latency ms: "
        f"mean {samples.mean():.2f}  "
        f"p50 {_percentile(samples, 50):.2f}  "
        f"p95 {_percentile(samples, 95):.2f}  "
        f"p99 {_percentile(samples, 99):.2f}  "
        f"min {samples.min():.2f}  "
        f"max {samples.max():.2f}"
    )


def _time_calls(function, requests: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(requests):
        started = time.perf_counter()
        function()
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples


def main() -> None:
    _configure_stdout()
    parser = argparse.ArgumentParser(description="Benchmark recommender latency and throughput.")
    parser.add_argument("--title-requests", type=int, default=200)
    parser.add_argument("--text-requests", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    started = time.perf_counter()
    engine = Recommender()
    index_ms = (time.perf_counter() - started) * 1000.0
    print(f"Index load: {index_ms:.1f} ms ({len(engine.records):,} movies)")

    titles = [record["title"] for record in engine.records if record.get("title")]
    rng = random.Random(args.seed)
    title_queries = [rng.choice(titles) for _ in range(args.title_requests + args.warmup)]
    title_cursor = 0

    def title_query() -> None:
        nonlocal title_cursor
        engine.similar_to_title(title_queries[title_cursor], args.k)
        title_cursor += 1

    title_samples = _time_calls(title_query, args.title_requests, args.warmup)
    print()
    _report("Title similarity", title_samples)

    started = time.perf_counter()
    engine.encode(TEXT_QUERIES[0])
    model_ms = (time.perf_counter() - started) * 1000.0
    print()
    print(f"Encoder load: {model_ms:.1f} ms")

    text_cursor = 0

    def text_query() -> None:
        nonlocal text_cursor
        engine.match_text(TEXT_QUERIES[text_cursor % len(TEXT_QUERIES)], args.k)
        text_cursor += 1

    text_samples = _time_calls(text_query, args.text_requests, min(args.warmup, len(TEXT_QUERIES)))
    print()
    _report("Text query (encode + search)", text_samples)


if __name__ == "__main__":
    main()
