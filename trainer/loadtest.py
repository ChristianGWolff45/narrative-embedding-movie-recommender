"""Open-loop load test of the running web app's /api/recommend endpoint.

Requests are sent on a fixed schedule at each target rate regardless of how
fast earlier ones finish, and latency is measured from the scheduled send
time, so queueing delay under overload is counted instead of hidden.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import threading
import time
import urllib.error
import urllib.request

from .stats import latency_summary

DIMENSIONS = ("plot", "themes", "characters", "tone", "narrative_structure")


def _post(url: str, payload: dict, timeout: float) -> int:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
            return response.status
    except urllib.error.HTTPError as error:
        return error.code
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return 0


def _payload(mode: str, tmdb_ids: list[int], rng: random.Random) -> dict:
    payload = {"seeds": rng.sample(tmdb_ids, k=rng.choice((1, 1, 2, 3))), "mode": mode, "count": 12}
    if mode == "multi":
        payload["weights"] = {name: round(rng.uniform(0, 3), 1) for name in DIMENSIONS}
        if sum(payload["weights"].values()) == 0:
            payload["weights"]["plot"] = 1.0
    return payload


def run_rate(url: str, mode: str, rate: float, duration: float, tmdb_ids: list[int], workers: int,
             timeout: float, seed: int) -> dict:
    rng = random.Random(seed)
    total = max(1, int(rate * duration))
    payloads = [_payload(mode, tmdb_ids, rng) for _ in range(total)]
    results: list[tuple[float, int]] = []
    lock = threading.Lock()

    def task(scheduled: float, payload: dict) -> None:
        status = _post(url, payload, timeout)
        finished = time.perf_counter()
        with lock:
            results.append((finished - scheduled, status))

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, payload in enumerate(payloads):
            scheduled = started + index / rate
            delay = scheduled - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            pool.submit(task, scheduled, payload)
    elapsed = time.perf_counter() - started
    ok = [latency for latency, status in results if status == 200]
    errors = len(results) - len(ok)
    return {
        "mode": mode,
        "offered_rps": rate,
        "requests": total,
        "achieved_rps": len(ok) / elapsed if elapsed else None,
        "errors": errors,
        "error_rate": errors / total,
        "latency": latency_summary(ok),
    }


def load_tmdb_ids(args: argparse.Namespace) -> list[int]:
    metadata = Path(args.artifact).with_suffix(".metadata.json")
    if metadata.exists():
        return [int(item["tmdb_id"]) for item in json.loads(metadata.read_text(encoding="utf-8"))]
    with urllib.request.urlopen(f"{args.url.rstrip('/')}/api/status", timeout=args.timeout) as response:
        featured = json.loads(response.read()).get("featured", [])
    ids = [int(movie["tmdbId"]) for movie in featured if movie.get("tmdbId")]
    if len(ids) < 3:
        raise SystemExit("Need the artifact metadata file or a running server with featured films.")
    return ids


def loadtest(args: argparse.Namespace) -> dict:
    tmdb_ids = load_tmdb_ids(args)
    endpoint = f"{args.url.rstrip('/')}/api/recommend"
    probe = _post(endpoint, _payload("single", tmdb_ids, random.Random(0)), args.timeout)
    if probe != 200:
        raise SystemExit(f"{endpoint} answered {probe or 'nothing'}; start the server with bin/web first.")
    rows = []
    for mode in args.modes:
        for rate in args.rates:
            row = run_rate(endpoint, mode, rate, args.duration, tmdb_ids, args.workers, args.timeout, args.seed)
            rows.append(row)
            latency = row["latency"]
            p95 = f"{latency['p95_ms']:.1f} ms" if latency.get("count") else "n/a"
            print(f"  {mode:<6} {rate:>6.0f} req/s offered -> {row['achieved_rps'] or 0:7.1f} ok/s, "
                  f"p95 {p95}, errors {row['errors']}", flush=True)
    sustainable = {}
    for mode in args.modes:
        passing = [
            row["offered_rps"] for row in rows
            if row["mode"] == mode and row["error_rate"] == 0 and row["latency"].get("count")
            and row["latency"]["p95_ms"] <= args.latency_budget_ms
        ]
        sustainable[mode] = max(passing, default=None)
    return {
        "url": endpoint,
        "duration_seconds_per_rate": args.duration,
        "latency_budget_ms": args.latency_budget_ms,
        "note": "Open loop; latency includes queueing from the scheduled send time and HTTP/JSON overhead.",
        "results": rows,
        "max_rate_within_budget": sustainable,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Load test the running recommender at increasing request rates.")
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--artifact", default="artifacts/movie_embeddings.npz")
    parser.add_argument("--modes", nargs="+", choices=("single", "multi"), default=["single", "multi"])
    parser.add_argument("--rates", type=float, nargs="+", default=[5, 10, 25, 50, 100, 200, 400])
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds per rate.")
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=585)
    parser.add_argument("--latency-budget-ms", type=float, default=100.0)
    parser.add_argument("--report", default="reports/loadtest.json")
    args = parser.parse_args()
    if any(rate <= 0 for rate in args.rates) or args.duration <= 0:
        parser.error("--rates and --duration must be positive")
    result = loadtest(args)
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Max rate within p95 <= {args.latency_budget_ms} ms: {result['max_rate_within_budget']}")
    print(f"Saved report to {report}")


if __name__ == "__main__":
    main()
