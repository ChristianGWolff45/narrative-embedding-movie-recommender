from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from .config import Settings
from .llm import LocalLabeler
from .progress import new_status, update_metrics, write_status
from .tmdb import TMDBClient


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _existing_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    result: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line).get("movie", {}).get("tmdb_id")
            if value is not None:
                result.add(int(value))
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
    return result


def run_labeling(
    settings: Settings,
    limit: int = 20,
    max_pages: int = 1,
    run_id: str | None = None,
    start_page: int = 1,
    end_page: int | None = None,
    output_file: str | None = None,
) -> Path:
    root = Path.cwd()
    data_dir = root / "data"
    run_id = run_id or _run_id()
    run_dir = root / "runs" / run_id
    end_page = end_page if end_page is not None else max_pages
    output_path = Path(output_file) if output_file else data_dir / "labels.jsonl"
    if not output_path.is_absolute():
        output_path = root / output_path
    status_path = run_dir / "status.json"
    errors_path = run_dir / "errors.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with TMDBClient(
        api_key=settings.tmdb_api_key,
        bearer_token=settings.tmdb_bearer_token,
        language=settings.tmdb_language,
    ) as tmdb:
        first_page = tmdb.discover_page(start_page)
        available = int(first_page.get("total_results", 0))
        page_size = len(first_page.get("results", [])) or 20
        range_end = min(available, end_page * page_size)
        range_start = min(available, (start_page - 1) * page_size)
        range_total = max(range_end - range_start, 0)
        total = min(limit, range_total) if limit > 0 else range_total
        status = new_status(run_id, total, settings.model_id, str(output_path))
        write_status(status_path, status)
        labeler = LocalLabeler(settings.model_id)
        existing = _existing_ids(output_path)
        processed_candidates = 0
        page = start_page

        try:
            status["status"] = "running"
            status["phase"] = "fetching_and_labeling"
            write_status(status_path, status)
            with output_path.open("a", encoding="utf-8") as output:
                while page <= end_page and processed_candidates < total:
                    payload = first_page if page == start_page else tmdb.discover_page(page)
                    for summary in payload.get("results", []):
                        if processed_candidates >= total:
                            break
                        processed_candidates += 1
                        movie_id = int(summary["id"])
                        status["last_movie_id"] = movie_id
                        try:
                            if movie_id in existing:
                                status["completed"] += 1
                                status["phase"] = "resuming_existing_output"
                                update_metrics(status)
                                write_status(status_path, status)
                                continue
                            movie = tmdb.movie_bundle(movie_id)
                            label = labeler.label(movie)
                            output.write(
                                json.dumps(
                                    {"movie": movie, "label": label.model_dump(), "run_id": run_id},
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            output.flush()
                            existing.add(movie_id)
                            status["completed"] += 1
                            status["phase"] = "fetching_and_labeling"
                        except Exception as exc:  # keep a long job moving past bad records
                            status["failed"] += 1
                            status["last_error"] = f"movie {movie_id}: {type(exc).__name__}: {exc}"
                            with errors_path.open("a", encoding="utf-8") as errors:
                                errors.write(json.dumps({"movie_id": movie_id, "error": status["last_error"]}) + "\n")
                        update_metrics(status)
                        write_status(status_path, status)
                    page += 1
            status["status"] = "completed"
            status["phase"] = "done"
        except KeyboardInterrupt:
            status["status"] = "stopped"
            status["phase"] = "interrupted"
        except Exception as exc:
            status["status"] = "failed"
            status["phase"] = "fatal_error"
            status["last_error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            update_metrics(status)
            write_status(status_path, status)
    return status_path
