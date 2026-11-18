from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_settings
from .llm import LocalLabeler
from .tmdb import TMDBClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and label selected TMDB movies")
    parser.add_argument("movie_id", type=int, nargs="+")
    args = parser.parse_args()
    output = Path("data/labels.jsonl")
    existing: set[int] = set()
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            try:
                existing.add(int(json.loads(line)["movie"]["tmdb_id"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass
    settings = load_settings()
    labeler = LocalLabeler(settings.model_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TMDBClient(
        api_key=settings.tmdb_api_key,
        bearer_token=settings.tmdb_bearer_token,
        language=settings.tmdb_language,
    ) as tmdb, output.open("a", encoding="utf-8") as stream:
        for movie_id in args.movie_id:
            if movie_id in existing:
                print(f"already present: {movie_id}")
                continue
            movie = tmdb.movie_bundle(movie_id)
            label = labeler.label(movie)
            stream.write(
                json.dumps(
                    {"movie": movie, "label": label.model_dump(), "run_id": "comparison"},
                    ensure_ascii=False,
                )
                + "\n"
            )
            stream.flush()
            existing.add(movie_id)
            print(f"added: {movie['title']} ({movie_id})")
