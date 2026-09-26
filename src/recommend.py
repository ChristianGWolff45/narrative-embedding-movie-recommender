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


def _encode_text(text: str) -> np.ndarray:
    from embed_movies import embed_texts

    return embed_texts([text])[0]


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


def recommend(query: str | None, likes: list[str], k: int, as_text: bool) -> None:
    _configure_stdout()
    records, embeddings = load_index()
    titles = _title_index(records)

    if likes:
        indices = []
        for title in likes:
            hits = titles.get(normalize_title(title), [])
            if not hits:
                raise SystemExit(f"No catalog movie matched {title!r}.")
            indices.append(_best_title_hit(hits, records))
        query_vector = embeddings[indices].mean(axis=0)
        query_vector = query_vector / np.linalg.norm(query_vector)
        exclude = set(indices)
        print("Based on " + ", ".join(_label(records[index]) for index in indices))
    elif query:
        hits = [] if as_text else titles.get(normalize_title(query), [])
        if hits:
            index = _best_title_hit(hits, records)
            query_vector = embeddings[index]
            exclude = {index}
            print(f"Movies similar to {_label(records[index])}")
            if records[index].get("director"):
                print(f"Director: {records[index]['director']}")
        else:
            print(f"Movies matching: {query}")
            query_vector = _encode_text(query)
            exclude = set()
    else:
        raise SystemExit("Pass a movie title, a description, or at least one --like title.")

    scores = embeddings @ query_vector
    print()
    for rank, index in enumerate(_top_indices(scores, k, exclude), start=1):
        record = records[index]
        genres = record.get("genres") or "Unknown genre"
        print(f"{rank:>2}. {scores[index]:.3f}  {_label(record)} | {genres}")


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
