from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .config import DEFAULT_MODEL_ID
from .llm import LocalLabeler, OllamaLabeler
from .progress import new_status, update_metrics, write_status


def _split_names(value: str | list[str] | None) -> list[str]:
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _existing_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    ids: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line).get("movie", {}).get("tmdb_id")
            if value is not None:
                ids.add(int(value))
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
    return ids


def _catalog_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                if row.get("tmdbId") is not None and row.get("movieId") is not None:
                    rows.append(row)
            except (AttributeError, json.JSONDecodeError):
                continue
    if not rows:
        raise SystemExit(f"No valid MovieLens/TMDB catalog records found in {path}")
    return rows


def _movie_from_row(row: dict) -> dict:
    cast = row.get("cast")
    if not isinstance(cast, list):
        cast = [{"name": name, "character": ""} for name in _split_names(row.get("top_cast"))]
    return {
        "tmdb_id": int(row["tmdbId"]),
        "title": row.get("title") or "",
        "overview": row.get("overview") or "",
        "release_date": row.get("release_date") or "",
        "genres": _split_names(row.get("genres")),
        "keywords": _split_names(row.get("keywords")),
        "director": row.get("director") or "",
        "document": row.get("document") or "",
        "cast": cast,
    }


def label_catalog(
    input_path: Path,
    output_path: Path,
    model_id: str,
    backend: str = "ollama",
    limit: int = 0,
    run_id: str | None = None,
) -> Path:
    rows = _catalog_rows(input_path)
    existing = _existing_ids(output_path)
    pending = [row for row in rows if int(row["tmdbId"]) not in existing]
    if limit > 0:
        pending = pending[:limit]
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path("runs") / run_id
    status_path = run_dir / "status.json"
    errors_path = run_dir / "errors.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    status = new_status(run_id, len(pending), model_id, str(output_path))
    status["status"] = "running"
    status["phase"] = "labeling_movie_lens_catalog"
    write_status(status_path, status)
    if not pending:
        status["status"] = "completed"
        status["phase"] = "done"
        update_metrics(status)
        write_status(status_path, status)
        print(f"All {len(rows):,} catalog movies are already labeled.")
        return status_path

    print(f"Labeling {len(pending):,} catalog movies with {model_id}", flush=True)
    labeler = OllamaLabeler(model_id) if backend == "ollama" else LocalLabeler(model_id)
    with output_path.open("a", encoding="utf-8") as output:
        for index, row in enumerate(pending, start=1):
            movie_id = int(row["tmdbId"])
            status["last_movie_id"] = movie_id
            try:
                movie = _movie_from_row(row)
                label = labeler.label(movie)
                result = {
                    "movie": movie,
                    "label": label.model_dump(),
                    "movieId": int(row["movieId"]),
                    "n_ratings": int(row.get("n_ratings") or 0),
                    "run_id": run_id,
                }
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
                status["completed"] += 1
            except Exception as exc:  # retain completed labels and continue after a bad record
                status["failed"] += 1
                status["last_error"] = f"movie {movie_id}: {type(exc).__name__}: {exc}"
                with errors_path.open("a", encoding="utf-8") as errors:
                    errors.write(json.dumps({"movie_id": movie_id, "error": status["last_error"]}) + "\n")
            update_metrics(status)
            write_status(status_path, status)
            if index % 25 == 0 or index == len(pending):
                processed = status["completed"] + status["failed"]
                print(f"Processed {processed:,}/{len(pending):,}", flush=True)

    status["status"] = "completed" if status["failed"] == 0 else "completed_with_errors"
    status["phase"] = "done"
    update_metrics(status)
    write_status(status_path, status)
    return status_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Label the MovieLens-selected TMDB catalog locally.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/movie_catalog.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/labels.jsonl"))
    parser.add_argument("--backend", choices=("ollama", "transformers"), default="ollama")
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=0, help="maximum new movies to label; 0 means all")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be zero or positive")
    model_id = args.model or ("qwen2.5:3b" if args.backend == "ollama" else DEFAULT_MODEL_ID)
    print(
        "Label status: "
        f"{label_catalog(args.input, args.output, model_id, args.backend, args.limit, args.run_id)}"
    )


if __name__ == "__main__":
    main()
