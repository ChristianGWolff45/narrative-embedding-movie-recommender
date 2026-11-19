from __future__ import annotations

import argparse
from pathlib import Path
import json
import time

import numpy as np

from .benchmark import add_arguments as add_benchmark_arguments, run as run_benchmark


DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    seen: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            movie_id = int(record["movie"]["tmdb_id"])
            if movie_id not in seen:
                records.append(record)
                seen.add(movie_id)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    if not records:
        raise RuntimeError(f"No valid labeled movies found in {path}")
    return records


def cast_text(movie: dict) -> str:
    entries = []
    for person in movie.get("cast", []):
        name = person.get("name", "").strip()
        character = person.get("character", "").strip()
        if name:
            entries.append(f"{name} as {character}" if character else name)
    return "; ".join(entries)


def texts_for(record: dict) -> tuple[str, dict[str, str]]:
    movie = record["movie"]
    label = record.get("label", {})
    title = movie.get("title", "")
    overview = movie.get("overview", "")
    genres = ", ".join(movie.get("genres", []))
    cast = cast_text(movie)
    single = movie.get("document") or "\n".join(
        part for part in [f"Title: {title}", f"Plot: {overview}", f"Genres: {genres}", f"Cast: {cast}"] if part
    )
    multi = {
        "plot": overview or title,
        "themes": ", ".join(label.get("themes", [])) or genres or title,
        "characters": ", ".join(label.get("characters", [])) or cast or title,
        "tone": ", ".join(label.get("tone", [])) or title,
        "narrative_structure": ", ".join(label.get("narrative_structure", [])) or overview or title,
    }
    return single, multi


class MiniLMEncoder:
    def __init__(self, model_id: str) -> None:
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_id)

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        vectors = np.asarray(
            [np.asarray(vector, dtype=np.float32) for vector in self.model.embed(texts, batch_size=batch_size)],
            dtype=np.float32,
        )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


def build(args: argparse.Namespace) -> None:
    records = load_records(Path(args.input))
    single_texts: list[str] = []
    multi_texts = {name: [] for name in ("plot", "themes", "characters", "tone", "narrative_structure")}
    metadata: list[dict] = []
    for record in records:
        single, multi = texts_for(record)
        single_texts.append(single)
        for name in multi_texts:
            multi_texts[name].append(multi[name])
        metadata.append(
            {
                "tmdb_id": record["movie"]["tmdb_id"],
                "movie_id": int(record.get("movieId") or 0),
                "n_ratings": int(record.get("n_ratings") or 0),
                "title": record["movie"].get("title", ""),
                "genres": record["movie"].get("genres", []),
            }
        )

    print(f"encoding {len(records)} movies with {args.model_id}", flush=True)
    encoder = MiniLMEncoder(args.model_id)
    started = time.perf_counter()
    baseline_path = Path(args.baseline_embeddings)
    catalog_path = Path(args.catalog)
    if baseline_path.exists():
        if not catalog_path.exists():
            raise SystemExit(f"Missing {catalog_path}; the baseline embeddings need their catalog metadata.")
        catalog = [
            json.loads(line)
            for line in catalog_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        baseline = np.load(baseline_path)
        if len(catalog) != len(baseline):
            raise SystemExit("GitHub baseline catalog and embedding rows are out of sync.")
        positions = {int(item["movieId"]): index for index, item in enumerate(catalog)}
        try:
            order = [positions[int(record["movieId"])] for record in records]
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit("Labels contain a MovieLens ID missing from the GitHub baseline catalog.") from exc
        arrays: dict[str, np.ndarray] = {"single": np.asarray(baseline[order], dtype=np.float32)}
        print(f"reusing {len(order):,} aligned single embeddings from {baseline_path}", flush=True)
        timings = {"single": None}
    else:
        single_started = time.perf_counter()
        arrays = {"single": encoder.encode(single_texts, args.batch_size)}
        timings = {"single": time.perf_counter() - single_started}
        print("baseline embedding file not found; encoded the single-text fallback", flush=True)
    for name, texts in multi_texts.items():
        print(f"encoding {name}", flush=True)
        dimension_started = time.perf_counter()
        arrays[name] = encoder.encode(texts, args.batch_size)
        timings[name] = time.perf_counter() - dimension_started
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    narrative_seconds = sum(timings[name] for name in multi_texts)
    output.with_suffix(".build.json").write_text(
        json.dumps(
            {
                "movies": len(records),
                "model_id": args.model_id,
                "batch_size": args.batch_size,
                "encode_seconds": timings,
                "narrative_encode_seconds": narrative_seconds,
                "narrative_ms_per_film": narrative_seconds / len(records) * 1000,
                "note": "single is null when the baseline vectors were reused instead of re-encoded",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output} in {time.perf_counter() - started:.1f}s", flush=True)


def recommend(args: argparse.Namespace) -> None:
    artifact = Path(args.artifact)
    data = np.load(artifact)
    metadata = json.loads(artifact.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    lookup = {item["title"].casefold(): index for index, item in enumerate(metadata)}
    if args.title.casefold() not in lookup:
        raise SystemExit(f"Title not found in artifact: {args.title}")
    query_index = lookup[args.title.casefold()]
    weights = {
        "plot": args.plot,
        "themes": args.themes,
        "characters": args.characters,
        "tone": args.tone,
        "narrative_structure": args.structure,
    }
    total = sum(weights.values()) or 1.0
    scores = sum((weights[name] / total) * (data[name] @ data[name][query_index]) for name in weights)
    scores[query_index] = -1.0
    top = np.argsort(scores)[-args.k :][::-1]
    for rank, index in enumerate(top, 1):
        print(f"{rank:2d}. {metadata[index]['title']}\t{scores[index]:.4f}\t{metadata[index]['tmdb_id']}")


def compare(args: argparse.Namespace) -> None:
    artifact = Path(args.artifact)
    data = np.load(artifact)
    metadata = json.loads(artifact.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    lookup = {item["title"].casefold(): index for index, item in enumerate(metadata)}
    first = lookup.get(args.first.casefold())
    second = lookup.get(args.second.casefold())
    if first is None or second is None:
        raise SystemExit(f"Both titles must be in the artifact: {args.first!r}, {args.second!r}")
    names = ("plot", "themes", "characters", "tone", "narrative_structure")
    scores = {name: float(data[name][first] @ data[name][second]) for name in names}
    scores["single"] = float(data["single"][first] @ data["single"][second])
    scores["multi_equal_weight"] = sum(scores[name] for name in names) / len(names)
    print(json.dumps({"first": args.first, "second": args.second, "similarity": scores}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct the CSCE 585 single/multi embedding systems")
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--input", default="data/labels.jsonl")
    build_parser.add_argument("--output", default="artifacts/movie_embeddings.npz")
    build_parser.add_argument("--baseline-embeddings", default="data/processed/embeddings.npy")
    build_parser.add_argument("--catalog", default="data/processed/movie_catalog.jsonl")
    build_parser.add_argument("--model-id", default=DEFAULT_ENCODER)
    build_parser.add_argument("--batch-size", type=int, default=128)
    build_parser.set_defaults(function=build)
    recommend_parser = sub.add_parser("recommend")
    recommend_parser.add_argument("title")
    recommend_parser.add_argument("--artifact", default="artifacts/movie_embeddings.npz")
    recommend_parser.add_argument("-k", type=int, default=10)
    recommend_parser.add_argument("--plot", type=float, default=1.0)
    recommend_parser.add_argument("--themes", type=float, default=1.0)
    recommend_parser.add_argument("--characters", type=float, default=1.0)
    recommend_parser.add_argument("--tone", type=float, default=1.0)
    recommend_parser.add_argument("--structure", type=float, default=1.0)
    recommend_parser.set_defaults(function=recommend)
    benchmark_parser = sub.add_parser("benchmark")
    add_benchmark_arguments(benchmark_parser)
    benchmark_parser.set_defaults(function=run_benchmark)
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("first")
    compare_parser.add_argument("second")
    compare_parser.add_argument("--artifact", default="artifacts/movie_embeddings.npz")
    compare_parser.set_defaults(function=compare)
    args = parser.parse_args()
    args.function(args)
