"""Reuse existing TMDB narrative labels for the MovieLens catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from pydantic import ValidationError

from .catalog import _catalog_rows, _movie_from_row
from .schema import MovieLabel


def seed(input_path: Path, source_path: Path, output_path: Path) -> tuple[int, int]:
    catalog = _catalog_rows(input_path)
    existing: dict[int, dict] = {}
    for current_path in (source_path, output_path):
        if not current_path.exists():
            continue
        with current_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    tmdb_id = int(row.get("movie", {}).get("tmdb_id"))
                    if row.get("label"):
                        # Keep the already-adapted catalog row on reruns so this
                        # step never discards newly generated labels.
                        if current_path == output_path:
                            existing[tmdb_id] = row
                        elif tmdb_id not in existing:
                            existing[tmdb_id] = row
                except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
                    continue

    output_path.parent.mkdir(parents=True, exist_ok=True)
    seeded = 0
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in catalog:
            tmdb_id = int(row["tmdbId"])
            source = existing.get(tmdb_id)
            if source is None:
                continue
            try:
                label = MovieLabel.model_validate(source["label"]).model_dump()
            except ValidationError:
                continue
            result = {
                "movie": _movie_from_row(row),
                "label": label,
                "movieId": int(row["movieId"]),
                "n_ratings": int(row.get("n_ratings") or 0),
                "run_id": str(source.get("run_id") or "reused-tmdb-label"),
            }
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            seeded += 1
    temporary.replace(output_path)
    return seeded, len(catalog) - seeded


def main() -> None:
    parser = argparse.ArgumentParser(description="Match saved TMDB labels to the MovieLens catalog.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/movie_catalog.jsonl"))
    parser.add_argument("--source", type=Path, default=Path("data/labels.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/movie_lens_labels.jsonl"))
    args = parser.parse_args()
    seeded, remaining = seed(args.input, args.source, args.output)
    print(f"Reused {seeded:,} saved labels; {remaining:,} catalog movies still need labels.", flush=True)


if __name__ == "__main__":
    main()
