"""Download TMDB metadata and write one catalog row per movie."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from dotenv import load_dotenv

from config import CATALOG_PATH, LINKS_PATH, PROCESSED_DIR, RATING_COUNTS_PATH, RATINGS_PATH, ROOT
from movie_document import fields_from_tmdb

load_dotenv(ROOT / ".env")


class RateLimiter:
    def __init__(self, per_second: float) -> None:
        self.per_second = per_second
        self._lock = threading.Lock()
        self._next_time = 0.0

    def wait(self) -> None:
        with self._lock:
            interval = 1.0 / self.per_second
            now = time.monotonic()
            delay = max(0.0, self._next_time - now)
            self._next_time = max(now, self._next_time) + interval
        if delay:
            time.sleep(delay)

    def slow_down(self) -> None:
        with self._lock:
            self.per_second = max(1.0, self.per_second / 2)


def _configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


def rating_counts() -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if RATING_COUNTS_PATH.exists():
        return pd.read_csv(RATING_COUNTS_PATH)
    print("Counting MovieLens ratings...", flush=True)
    ratings = pd.read_csv(RATINGS_PATH, usecols=["movieId"])
    counts = ratings.groupby("movieId").size().rename("n_ratings").reset_index()
    counts.to_csv(RATING_COUNTS_PATH, index=False)
    return counts


def select_movies(min_ratings: int, limit: int | None) -> pd.DataFrame:
    links = pd.read_csv(LINKS_PATH, usecols=["movieId", "tmdbId"])
    counts = rating_counts()
    movies = links.merge(counts, on="movieId", how="inner")
    movies = movies[movies["tmdbId"].notna() & (movies["n_ratings"] >= min_ratings)]
    movies["tmdbId"] = movies["tmdbId"].astype(int)
    movies = movies.sort_values(["n_ratings", "movieId"], ascending=[False, True])
    movies = movies.drop_duplicates("tmdbId", keep="first")
    if limit is not None:
        movies = movies.head(limit)
    return movies.reset_index(drop=True)


def load_catalog() -> dict[int, dict]:
    catalog: dict[int, dict] = {}
    if not CATALOG_PATH.exists():
        return catalog
    with CATALOG_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            tmdb_id = record.get("tmdbId")
            if isinstance(tmdb_id, int):
                catalog[tmdb_id] = record
    return catalog


def write_catalog(records: dict[int, dict]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records.values(), key=lambda record: (-record.get("n_ratings", 0), record["movieId"]))
    temporary = CATALOG_PATH.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in ordered:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(CATALOG_PATH)


def _fetch_payload(session: requests.Session, token: str, tmdb_id: int, limiter: RateLimiter) -> dict | None:
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
    headers = {"Authorization": f"Bearer {token}", "accept": "application/json"}
    params = {"language": "en-US", "append_to_response": "credits,keywords"}
    for attempt in range(5):
        limiter.wait()
        try:
            response = session.get(url, headers=headers, params=params, timeout=30)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if response.status_code == 429:
            limiter.slow_down()
            time.sleep(int(response.headers.get("Retry-After", 2 ** attempt)))
            continue
        if response.status_code == 404:
            return None
        if response.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"TMDB request failed for id {tmdb_id}")


def _record_from_payload(payload: dict, movie_id: int, tmdb_id: int, n_ratings: int) -> dict | None:
    if payload.get("adult") is True:
        return None
    fields = fields_from_tmdb(payload)
    if not fields["title"]:
        return None
    return {
        "movieId": movie_id,
        "tmdbId": tmdb_id,
        "n_ratings": n_ratings,
        **fields,
    }


def fetch_catalog(min_ratings: int = 500, limit: int | None = None, workers: int = 8) -> int:
    _configure_stdout()
    token = os.getenv("TMDB_READ_TOKEN")
    if not token:
        raise SystemExit("TMDB_READ_TOKEN is missing from .env")

    selected = select_movies(min_ratings, limit)
    catalog = load_catalog()
    pending = selected[~selected["tmdbId"].isin(catalog)]
    print(f"Catalog target: {len(selected):,} movies ({len(pending):,} still to fetch)", flush=True)
    if pending.empty:
        write_catalog({row.tmdbId: catalog[row.tmdbId] for row in selected.itertuples(index=False) if row.tmdbId in catalog})
        print(f"Wrote {CATALOG_PATH}", flush=True)
        return len(selected)

    limiter = RateLimiter(per_second=8)
    thread_state = threading.local()
    write_lock = threading.Lock()
    failures = 0

    def worker(row) -> dict | None:
        session = getattr(thread_state, "session", None)
        if session is None:
            session = requests.Session()
            thread_state.session = session
        payload = _fetch_payload(session, token, int(row.tmdbId), limiter)
        if payload is None:
            return None
        return _record_from_payload(payload, int(row.movieId), int(row.tmdbId), int(row.n_ratings))

    done = 0
    with CATALOG_PATH.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(worker, row) for row in pending.itertuples(index=False)]
            for future in as_completed(futures):
                done += 1
                try:
                    record = future.result()
                except Exception as error:
                    failures += 1
                    print(f"Fetch error: {error}", flush=True)
                    continue
                if record is None:
                    failures += 1
                    continue
                catalog[record["tmdbId"]] = record
                with write_lock:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                if done % 50 == 0 or done == len(futures):
                    print(f"Fetched {done:,}/{len(futures):,}", flush=True)

    kept = {
        int(row.tmdbId): catalog[int(row.tmdbId)]
        for row in selected.itertuples(index=False)
        if int(row.tmdbId) in catalog
    }
    write_catalog(kept)
    print(f"Wrote {len(kept):,} movies to {CATALOG_PATH} ({failures:,} skipped)", flush=True)
    return len(kept)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch TMDB metadata for MovieLens titles.")
    parser.add_argument("--min-ratings", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    fetch_catalog(args.min_ratings, args.limit, args.workers)


if __name__ == "__main__":
    main()
