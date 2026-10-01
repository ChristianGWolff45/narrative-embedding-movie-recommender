"""Recommend movies from the single TMDB embedding index."""

from __future__ import annotations

import argparse
import json
import re
import sys

import numpy as np

from config import EMBEDDINGS_PATH, MODEL_INFO_PATH, MODEL_NAME
from embed_movies import load_records

_YEAR_SUFFIX = re.compile(r"\s+\(\d{4}\)$")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


def normalize_title(title: str) -> str:
    title = _YEAR_SUFFIX.sub("", title.strip())
    title = title.casefold()
    title = _NON_ALNUM.sub(" ", title)
    return " ".join(title.split())


def _year(record: dict) -> str:
    release_date = record.get("release_date") or ""
    return release_date[:4] if len(release_date) >= 4 else ""


def _label(record: dict) -> str:
    year = _year(record)
    title = record.get("title") or "Untitled"
    return f"{title} ({year})" if year else title


def load_index() -> tuple[list[dict], np.ndarray]:
    if not EMBEDDINGS_PATH.exists():
        raise SystemExit(f"Missing {EMBEDDINGS_PATH}. Run src/build_single_embedding.py first.")
    records = load_records()
    embeddings = np.load(EMBEDDINGS_PATH)
    if len(records) != len(embeddings):
        raise SystemExit("Catalog and embeddings are out of sync. Re-run src/embed_movies.py.")
    if MODEL_INFO_PATH.exists():
        info = json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))
        if info.get("model_name") != MODEL_NAME:
            raise SystemExit(f"Index was built with {info.get('model_name')}, expected {MODEL_NAME}.")
    return records, embeddings


def _title_index(records: list[dict]) -> dict[str, list[int]]:
    matches: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        key = normalize_title(record.get("title") or "")
        if key:
            matches.setdefault(key, []).append(index)
    return matches


def _best_title_hit(hits: list[int], records: list[dict]) -> int:
    return max(hits, key=lambda index: records[index].get("n_ratings") or 0)


def _top_indices(scores: np.ndarray, k: int, exclude: set[int]) -> list[int]:
    if exclude:
        scores = scores.copy()
        for index in exclude:
            scores[index] = -np.inf
    k = min(k, len(scores) - len(exclude))
    if k <= 0:
        return []
    chosen = np.argpartition(scores, -k)[-k:]
    return sorted(chosen.tolist(), key=lambda index: scores[index], reverse=True)


class Recommender:
    """Loaded catalog and encoder. One instance serves many queries."""

    def __init__(self) -> None:
        self.records, self.embeddings = load_index()
        self.titles = _title_index(self.records)
        self._model = None

    def resolve_title(self, title: str) -> int | None:
        hits = self.titles.get(normalize_title(title), [])
        if not hits:
            return None
        return _best_title_hit(hits, self.records)

    def encode(self, text: str) -> np.ndarray:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(MODEL_NAME)
        vector = np.asarray(next(iter(self._model.embed([text]))), dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        return vector / norm

    def neighbors(self, query_vector: np.ndarray, k: int, exclude: set[int]) -> list[tuple[float, int]]:
        scores = self.embeddings @ query_vector
        return [(float(scores[index]), index) for index in _top_indices(scores, k, exclude)]

    def similar_to_title(self, title: str, k: int = 10) -> list[tuple[float, int]] | None:
        index = self.resolve_title(title)
        if index is None:
            return None
        return self.neighbors(self.embeddings[index], k, {index})

    def from_likes(self, titles: list[str], k: int = 10) -> tuple[list[int], list[tuple[float, int]]]:
        indices = []
        for title in titles:
            index = self.resolve_title(title)
            if index is None:
                raise SystemExit(f"No catalog movie matched {title!r}.")
            indices.append(index)
        query_vector = self.embeddings[indices].mean(axis=0)
        query_vector = query_vector / np.linalg.norm(query_vector)
        return indices, self.neighbors(query_vector, k, set(indices))

    def match_text(self, text: str, k: int = 10) -> list[tuple[float, int]]:
        return self.neighbors(self.encode(text), k, set())


def _print_hits(engine: Recommender, hits: list[tuple[float, int]]) -> None:
    print()
    for rank, (score, index) in enumerate(hits, start=1):
        record = engine.records[index]
        genres = record.get("genres") or "Unknown genre"
        print(f"{rank:>2}. {score:.3f}  {_label(record)} | {genres}")


def recommend(query: str | None, likes: list[str], k: int, as_text: bool) -> None:
    _configure_stdout()
    engine = Recommender()

    if likes:
        indices, hits = engine.from_likes(likes, k)
        print("Based on " + ", ".join(_label(engine.records[index]) for index in indices))
        _print_hits(engine, hits)
        return

    if not query:
        raise SystemExit("Pass a movie title, a description, or at least one --like title.")

    title_hits = None if as_text else engine.similar_to_title(query, k)
    if title_hits is not None:
        index = engine.resolve_title(query)
        record = engine.records[index]
        print(f"Movies similar to {_label(record)}")
        if record.get("director"):
            print(f"Director: {record['director']}")
        _print_hits(engine, title_hits)
        return

    print(f"Movies matching: {query}")
    _print_hits(engine, engine.match_text(query, k))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recommend movies with the single TMDB embedding.",
        epilog=(
            'examples:\n'
            '  python src/recommend.py "Fight Club"\n'
            '  python src/recommend.py --text "an insomniac starts an underground fight club"\n'
            '  python src/recommend.py --like "Toy Story" --like "Finding Nemo"'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", nargs="?", help="Movie title, or a free-text description with --text")
    parser.add_argument("--text", action="store_true", help="Embed the query as description text")
    parser.add_argument("--like", action="append", default=[], help="Anchor movie title. Repeat to combine.")
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()
    if not args.query and not args.like:
        parser.error("provide a query or --like")
    recommend(args.query, args.like, args.k, args.text)


if __name__ == "__main__":
    main()
