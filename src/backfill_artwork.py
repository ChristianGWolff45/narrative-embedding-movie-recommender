"""Fill poster and backdrop paths for cached TMDB catalog rows."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
import requests

from config import ROOT
from fetch_tmdb import RateLimiter, load_catalog, write_catalog


load_dotenv(ROOT / ".env")
load_dotenv(Path.home() / ".config" / "csce585-labeler" / ".env", override=False)


def backfill(workers: int = 8) -> tuple[int, int]:
    bearer = os.getenv("TMDB_READ_TOKEN") or os.getenv("TMDB_BEARER_TOKEN")
    api_key = os.getenv("TMDB_API_KEY")
    if not bearer and not api_key:
        raise SystemExit("TMDB credential is missing from the private config file.")

    catalog = load_catalog()
    pending = [
        record for record in catalog.values()
        if (not record.get("poster_path") or not record.get("backdrop_path"))
        and not record.get("artwork_checked")
    ]
    if not pending:
        complete = sum(bool(row.get("poster_path") and row.get("backdrop_path")) for row in catalog.values())
        print(
            f"Artwork lookup complete: {complete:,}/{len(catalog):,} movies have both paths; "
            "no unattempted TMDB lookups remain.",
            flush=True,
        )
        return 0, 0

    limiter = RateLimiter(per_second=6)
    thread_state = threading.local()

    def fetch(record: dict) -> tuple[int, dict | None]:
        session = getattr(thread_state, "session", None)
        if session is None:
            session = requests.Session()
            thread_state.session = session
        tmdb_id = int(record["tmdbId"])
        headers = {"accept": "application/json"}
        params = {"language": "en-US"}
        if bearer:
            headers["Authorization"] = f"Bearer {bearer}"
        else:
            params["api_key"] = api_key
        for attempt in range(5):
            limiter.wait()
            try:
                response = session.get(
                    f"https://api.themoviedb.org/3/movie/{tmdb_id}",
                    headers=headers,
                    params=params,
                    timeout=30,
                )
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if response.status_code == 429:
                limiter.slow_down()
                time.sleep(int(response.headers.get("Retry-After", 2 ** attempt)))
                continue
            if response.status_code == 404:
                return tmdb_id, None
            if response.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            payload = response.json()
            return tmdb_id, {
                "poster_path": payload.get("poster_path") or "",
                "backdrop_path": payload.get("backdrop_path") or "",
            }
        raise RuntimeError(f"TMDB artwork request failed for id {tmdb_id}")

    failures = 0
    complete = 0
    print(f"Backfilling artwork for {len(pending):,} movies.", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, record) for record in pending]
        for future in as_completed(futures):
            try:
                tmdb_id, artwork = future.result()
                if artwork is None:
                    failures += 1
                    catalog[tmdb_id]["artwork_checked"] = True
                else:
                    catalog[tmdb_id].update(artwork, artwork_checked=True)
            except Exception as exc:
                failures += 1
                print(f"Artwork lookup failed ({type(exc).__name__}); continuing.", flush=True)
            complete += 1
            if complete % 100 == 0 or complete == len(pending):
                write_catalog(catalog)
                print(f"Processed {complete:,}/{len(pending):,} artwork records.", flush=True)
    write_catalog(catalog)
    return complete - failures, failures


if __name__ == "__main__":
    succeeded, failed = backfill()
    print(f"Artwork paths saved: {succeeded:,}; unavailable or failed: {failed:,}", flush=True)
