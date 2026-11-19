from __future__ import annotations

import argparse
import json
import math
import mimetypes
from pathlib import Path
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
STATIC = Path(__file__).resolve().parent / "static"
ARTIFACT = ROOT / "artifacts" / "movie_embeddings.npz"
ARTIFACT_METADATA = ARTIFACT.with_suffix(".metadata.json")
sys.path.insert(0, str(SRC))

from config import EMBEDDINGS_PATH, MODEL_INFO_PATH  # noqa: E402
from embed_movies import load_records  # noqa: E402
from recommend import Recommender, normalize_title  # noqa: E402

DIMENSIONS = ("plot", "themes", "characters", "tone", "narrative_structure")
YEAR = re.compile(r"^\d{4}")


def _genres(record: dict) -> list[str]:
    value = record.get("genres") or []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(part).strip() for part in value if str(part).strip()]


def _movie_view(record: dict, score: float | None = None) -> dict:
    release_date = str(record.get("release_date") or "")
    poster = record.get("poster_path")
    if not isinstance(poster, str) or not poster.startswith("/"):
        poster = None
    backdrop = record.get("backdrop_path")
    if not isinstance(backdrop, str) or not backdrop.startswith("/"):
        backdrop = None
    return {
        "tmdbId": int(record.get("tmdbId") or record.get("tmdb_id") or 0),
        "movieId": int(record.get("movieId") or record.get("movie_id") or 0),
        "title": str(record.get("title") or "Untitled"),
        "year": release_date[:4] if YEAR.match(release_date) else "",
        "genres": _genres(record),
        "overview": str(record.get("overview") or ""),
        "posterPath": poster,
        "backdropPath": backdrop,
        "director": str(record.get("director") or ""),
        "ratingCount": int(record.get("n_ratings") or 0),
        "score": round(float(score), 4) if score is not None else None,
    }


MIN_YEAR, MAX_YEAR = 1870, 2100


def _year_of(record: dict) -> int:
    release_date = str(record.get("release_date") or "")
    return int(release_date[:4]) if YEAR.match(release_date) else 0


def parse_filters(value, known_genres: set[str]) -> dict:
    """Validate the optional filters object from a recommend request."""
    if value in (None, {}):
        return {}
    if not isinstance(value, dict):
        raise ValueError("Filters must be an object.")
    result: dict = {}
    for key in ("genres", "excludeGenres"):
        items = value.get(key) or []
        if isinstance(items, str):
            items = [items]
        if not isinstance(items, list) or len(items) > 30:
            raise ValueError("Genre filters must be a short list.")
        names = [str(item) for item in items if str(item)]
        unknown = [name for name in names if name not in known_genres]
        if unknown:
            raise ValueError(f"Unknown genre: {unknown[0]}.")
        if names:
            result[key] = names
    for key in ("yearMin", "yearMax"):
        raw = value.get(key)
        if raw in (None, ""):
            continue
        try:
            year = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("Years must be whole numbers.") from exc
        if not MIN_YEAR <= year <= MAX_YEAR:
            raise ValueError(f"Years must be between {MIN_YEAR} and {MAX_YEAR}.")
        result[key] = year
    if result.get("yearMin", MIN_YEAR) > result.get("yearMax", MAX_YEAR):
        raise ValueError("The start year must not be after the end year.")
    if set(result.get("genres", [])) & set(result.get("excludeGenres", [])):
        raise ValueError("A genre cannot be both required and excluded.")
    return result


def filter_mask(genre_sets: list[set[str]], years: np.ndarray, filters: dict) -> np.ndarray | None:
    """Boolean mask of films allowed by the filters, or None when nothing is filtered."""
    if not filters:
        return None
    mask = np.ones(len(genre_sets), dtype=bool)
    include = set(filters.get("genres", []))
    exclude = set(filters.get("excludeGenres", []))
    if include or exclude:
        mask &= np.fromiter(
            ((not include or bool(genres & include)) and not (genres & exclude) for genres in genre_sets),
            dtype=bool,
            count=len(genre_sets),
        )
    if "yearMin" in filters:
        mask &= years >= filters["yearMin"]
    if "yearMax" in filters:
        mask &= (years <= filters["yearMax"]) & (years > 0)
    return mask


class RecommenderApp:
    def __init__(self) -> None:
        if not EMBEDDINGS_PATH.exists():
            raise RuntimeError(f"Missing baseline vectors at {EMBEDDINGS_PATH}")
        self.engine = Recommender()
        self.records = self.engine.records
        self.embeddings = self.engine.embeddings
        self.by_tmdb = {
            int(record["tmdbId"]): index
            for index, record in enumerate(self.records)
            if record.get("tmdbId") is not None
        }
        self.genre_sets = [set(_genres(record)) for record in self.records]
        self.genre_list = sorted(set().union(*self.genre_sets)) if self.genre_sets else []
        self.years = np.asarray([_year_of(record) for record in self.records], dtype=np.int32)
        self.multi_data: dict[str, np.ndarray] | None = None
        self.multi_metadata: list[dict] = []
        self.multi_to_catalog: list[int] = []
        self._load_multi_artifact()

    def _load_multi_artifact(self) -> None:
        if self.multi_data is not None:
            return
        if not ARTIFACT.exists() or not ARTIFACT_METADATA.exists():
            return
        try:
            arrays = np.load(ARTIFACT)
            metadata = json.loads(ARTIFACT_METADATA.read_text(encoding="utf-8"))
            if not all(name in arrays.files for name in ("single", *DIMENSIONS)):
                return
            if len(metadata) != len(self.records):
                return
            rows = []
            for item in metadata:
                row = self.by_tmdb.get(int(item.get("tmdb_id") or 0))
                if row is None:
                    return
                rows.append(row)
            self.multi_data = {name: arrays[name] for name in ("single", *DIMENSIONS)}
            self.multi_metadata = metadata
            self.multi_to_catalog = rows
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return

    def _label_status(self) -> dict | None:
        status_files = sorted(
            (ROOT / "runs").glob("*/status.json"), key=lambda path: path.stat().st_mtime
        )
        if not status_files:
            return None
        try:
            return json.loads(status_files[-1].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _label_count(self) -> int:
        target_path = ROOT / "data" / "processed" / "movie_lens_labels.jsonl"
        has_target = target_path.exists()
        path = target_path if has_target else ROOT / "data" / "labels.jsonl"
        if not path.exists():
            return 0
        ids: set[int] = set()
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line).get("movie", {}).get("tmdb_id")
                    if value is not None:
                        movie_id = int(value)
                        if has_target or movie_id in self.by_tmdb:
                            ids.add(movie_id)
                except (AttributeError, ValueError, TypeError, json.JSONDecodeError):
                    continue
        return len(ids)

    def status(self) -> dict:
        self._load_multi_artifact()
        model_info = {}
        if MODEL_INFO_PATH.exists():
            try:
                model_info = json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        count = self._label_count()
        latest = self._label_status()
        evaluation = None
        evaluation_path = ROOT / "reports" / "evaluation.json"
        if evaluation_path.exists():
            try:
                evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        featured = sorted(
            self.records,
            key=lambda row: int(row.get("n_ratings") or 0),
            reverse=True,
        )[:8]
        return {
            "movieCount": len(self.records),
            "labelCount": count,
            "labelPercent": round(count * 100 / max(1, len(self.records))),
            "multiReady": self.multi_data is not None,
            "baselineModel": model_info.get("model_name", "all-MiniLM-L6-v2"),
            "labelRun": {
                "status": latest.get("status"),
                "completed": int(latest.get("completed") or 0),
                "failed": int(latest.get("failed") or 0),
                "total": int(latest.get("total") or 0),
                "remaining": int(latest.get("remaining") or 0),
                "itemsPerMinute": latest.get("items_per_minute"),
                "etaSeconds": latest.get("eta_seconds"),
                "updatedAt": latest.get("updated_at"),
            } if latest else None,
            "evaluation": evaluation,
            "genres": self.genre_list,
            "yearRange": [int(known.min()), int(known.max())] if (known := self.years[self.years > 0]).size else None,
            "featured": [_movie_view(row) for row in featured],
        }

    def search(self, query: str, limit: int = 8) -> list[dict]:
        normalized = normalize_title(query)
        if not normalized:
            return []
        tokens = normalized.split()
        hits: list[tuple[int, int, dict]] = []
        for row in self.records:
            title = normalize_title(str(row.get("title") or ""))
            if not title:
                continue
            if title == normalized:
                relevance = 0
            elif title.startswith(normalized):
                relevance = 1
            elif normalized in title:
                relevance = 2
            elif all(token in title for token in tokens):
                relevance = 3
            else:
                continue
            hits.append((relevance, -int(row.get("n_ratings") or 0), row))
        hits.sort(key=lambda item: (item[0], item[1]))
        return [_movie_view(row) for _, _, row in hits[:limit]]

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    @staticmethod
    def _top(scores: np.ndarray, count: int, excluded: set[int], allowed: np.ndarray | None = None) -> list[int]:
        values = scores.copy()
        if allowed is not None:
            values[~allowed] = -np.inf
        for index in excluded:
            if 0 <= index < len(values):
                values[index] = -np.inf
        amount = min(count, int(np.isfinite(values).sum()))
        if amount < 1:
            return []
        choices = np.argpartition(values, -amount)[-amount:]
        return sorted((int(index) for index in choices), key=lambda i: values[i], reverse=True)

    def recommend(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("A JSON object is required.")
        seed_ids = payload.get("seeds") or []
        if not isinstance(seed_ids, list) or len(seed_ids) > 3:
            raise ValueError("Choose one to three reference films.")
        try:
            seed_ids = [int(value) for value in seed_ids]
        except (TypeError, ValueError) as exc:
            raise ValueError("Reference film IDs must be integers.") from exc
        if len(set(seed_ids)) != len(seed_ids):
            raise ValueError("Remove duplicate reference films.")
        seed_indices = []
        for tmdb_id in seed_ids:
            if tmdb_id not in self.by_tmdb:
                raise ValueError("A selected film is not in this collection.")
            seed_indices.append(self.by_tmdb[tmdb_id])
        mode = str(payload.get("mode") or "single")
        if mode not in {"single", "multi"}:
            raise ValueError("Unknown recommendation profile.")
        if mode == "multi" and self.multi_data is None:
            raise RuntimeError("The narrative profile is still being built.")
        if not seed_indices:
            raise ValueError("Choose a reference film first.")
        try:
            count = max(1, min(24, int(payload.get("count") or 12)))
        except (ValueError, TypeError) as exc:
            raise ValueError("Result count must be a number.") from exc
        selected_weights = payload.get("weights") or {}
        weights = {}
        for name in DIMENSIONS:
            value = selected_weights.get(name, 1.0) if isinstance(selected_weights, dict) else 1.0
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("Profile weights must be numbers.") from exc
            if not math.isfinite(value) or value < 0 or value > 3:
                raise ValueError("Profile weights must be between 0 and 3.")
            weights[name] = value

        filters = parse_filters(payload.get("filters"), set(self.genre_list))
        allowed = filter_mask(self.genre_sets, self.years, filters)
        matching = int(allowed.sum()) if allowed is not None else len(self.records)

        if mode == "single":
            query = self._normalize(self.embeddings[seed_indices].mean(axis=0))
            scores = self.embeddings @ query
            top = self._top(scores, count, set(seed_indices), allowed)
            catalog_indices = top
            ranked_scores = scores[top]
        else:
            assert self.multi_data is not None
            metadata_by_tmdb = {
                int(item["tmdb_id"]): index for index, item in enumerate(self.multi_metadata)
            }
            try:
                multi_seeds = [metadata_by_tmdb[value] for value in seed_ids]
            except KeyError as exc:
                raise ValueError("A selected film is missing from the narrative index.") from exc
            total_weight = sum(weights.values())
            if total_weight <= 0:
                raise ValueError("Give at least one story dimension a nonzero weight.")
            scores = np.zeros(len(self.multi_metadata), dtype=np.float32)
            for name, weight in weights.items():
                if weight:
                    query = self._normalize(self.multi_data[name][multi_seeds].mean(axis=0))
                    scores += (weight / total_weight) * (self.multi_data[name] @ query)
            multi_allowed = allowed[self.multi_to_catalog] if allowed is not None else None
            top = self._top(scores, count, set(multi_seeds), multi_allowed)
            catalog_indices = [self.multi_to_catalog[index] for index in top]
            ranked_scores = scores[top]

        return {
            "mode": mode,
            "results": [
                _movie_view(self.records[index], float(score))
                for index, score in zip(catalog_indices, ranked_scores)
            ],
            "seeds": [_movie_view(self.records[index]) for index in seed_indices],
            "weights": weights if mode == "multi" else None,
            "filters": filters,
            "matchingCount": matching,
            "method": "Five narrative vectors blended by cosine similarity" if mode == "multi" else "Single movie-document vector",
        }


APP: RecommenderApp | None = None


class Handler(BaseHTTPRequestHandler):
    server_version = "ReelNotes/1.0"

    def _send(self, body: bytes, content_type: str, status: int = 200, cache: str = "no-store") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' https://image.tmdb.org data:; "
            "style-src 'self'; script-src 'self'; connect-src 'self'; "
            "font-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def _json(self, value: dict, status: int = 200) -> None:
        self._send(json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        if parsed.path == "/api/status":
            assert APP is not None
            self._json(APP.status())
            return
        if parsed.path == "/api/search":
            assert APP is not None
            query = parse_qs(parsed.query).get("q", [""])[0][:120]
            self._json({"results": APP.search(query)})
            return
        if parsed.path in {"/", "/index.html"}:
            self._static(STATIC / "index.html")
            return
        if parsed.path.startswith("/static/"):
            relative = unquote(parsed.path.removeprefix("/static/"))
            target = (STATIC / relative).resolve()
            if STATIC.resolve() not in target.parents or not target.is_file():
                self._json({"error": "Not found."}, 404)
                return
            self._static(target)
            return
        self._json({"error": "Not found."}, 404)

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/api/recommend":
            self._json({"error": "Not found."}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length < 1 or length > 64_000:
                raise ValueError("Request size is invalid.")
            payload = json.loads(self.rfile.read(length))
            assert APP is not None
            self._json(APP.recommend(payload))
        except RuntimeError as exc:
            self._json({"error": str(exc)}, 409)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, 400)

    def _static(self, path: Path) -> None:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send(path.read_bytes(), content_type, cache="no-cache")

    def log_message(self, format: str, *args) -> None:
        sys.stderr.write(f"{self.address_string()} - {format % args}\n")


def main() -> None:
    global APP
    parser = argparse.ArgumentParser(description="Run the local movie recommender website.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    APP = RecommenderApp()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Reel Notes is ready at http://{args.host}:{args.port}", flush=True)
    print(f"Collection: {len(APP.records):,} films; narrative profile {'ready' if APP.multi_data is not None else 'waiting for labels'}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Reel Notes.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
