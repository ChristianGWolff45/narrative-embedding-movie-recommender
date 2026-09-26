"""Fetch TMDB metadata and build the single movie embedding index."""

from __future__ import annotations

import argparse

from embed_movies import embed_catalog
from fetch_tmdb import fetch_catalog


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build one embedding per movie from TMDB title, overview, genres, "
            "keywords, director, and top cast."
        )
    )
    parser.add_argument("--min-ratings", type=int, default=500, help="Minimum MovieLens ratings to include")
    parser.add_argument("--limit", type=int, default=None, help="Maximum movies, most-rated first")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-fetch", action="store_true")
    args = parser.parse_args()
    if not args.skip_fetch:
        fetch_catalog(args.min_ratings, args.limit, args.workers)
    embed_catalog()


if __name__ == "__main__":
    main()
