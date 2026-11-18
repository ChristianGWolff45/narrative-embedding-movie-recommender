from __future__ import annotations

import argparse
from pathlib import Path
import json
import time

import numpy as np


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
    return "; ".join(
        f"{person.get('name', '')} as {person.get('character', '')}".strip()
        for person in movie.get("cast", [])
    )


def texts_for(record: dict) -> tuple[str, dict[str, str]]:
    movie = record["movie"]
    label = record.get("label", {})
    title = movie.get("title", "")
    overview = movie.get("overview", "")
    genres = ", ".join(movie.get("genres", []))
    cast = cast_text(movie)
    single = "\n".join(
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
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(self.device)
        self.model.eval()

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        import torch.nn.functional as F

        output: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            with self.torch.inference_mode():
                hidden = self.model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
                pooled = F.normalize(pooled, p=2, dim=1)
            output.append(pooled.float().cpu().numpy())
        return np.concatenate(output, axis=0)


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
                "title": record["movie"].get("title", ""),
                "genres": record["movie"].get("genres", []),
            }
        )

    print(f"encoding {len(records)} movies with {args.model_id}", flush=True)
    encoder = MiniLMEncoder(args.model_id)
    started = time.perf_counter()
    arrays: dict[str, np.ndarray] = {"single": encoder.encode(single_texts, args.batch_size)}
    for name, texts in multi_texts.items():
        print(f"encoding {name}", flush=True)
        arrays[name] = encoder.encode(texts, args.batch_size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
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


def benchmark(args: argparse.Namespace) -> None:
    data = np.load(args.artifact)
    rng = np.random.default_rng(585)
    count = data["single"].shape[0]
    queries = rng.integers(0, count, size=args.queries)
    started = time.perf_counter()
    single_scores = data["single"][queries] @ data["single"].T
    np.argpartition(single_scores, -10, axis=1)[:, -10:]
    single_seconds = time.perf_counter() - started
    started = time.perf_counter()
    multi_scores = sum(
        data[name][queries] @ data[name].T
        for name in ("plot", "themes", "characters", "tone", "narrative_structure")
    ) / 5.0
    np.argpartition(multi_scores, -10, axis=1)[:, -10:]
    multi_seconds = time.perf_counter() - started
    result = {
        "movies": int(count),
        "queries": args.queries,
        "single_mean_ms": single_seconds / args.queries * 1000,
        "multi_mean_ms": multi_seconds / args.queries * 1000,
        "single_qps": args.queries / single_seconds,
        "multi_qps": args.queries / multi_seconds,
    }
    print(json.dumps(result, indent=2))
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


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
    benchmark_parser.add_argument("--artifact", default="artifacts/movie_embeddings.npz")
    benchmark_parser.add_argument("--queries", type=int, default=200)
    benchmark_parser.add_argument("--report", default="reports/reconstruction_benchmark.json")
    benchmark_parser.set_defaults(function=benchmark)
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("first")
    compare_parser.add_argument("second")
    compare_parser.add_argument("--artifact", default="artifacts/movie_embeddings.npz")
    compare_parser.set_defaults(function=compare)
    args = parser.parse_args()
    args.function(args)
