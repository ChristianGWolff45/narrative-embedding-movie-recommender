"""Systems cost of the narrative recommender versus the single-vector baseline.

Measures what the proposal's evaluation plan names: per-request latency and
throughput under real-time weight changes, retrieval depth (k), catalog-size
scaling, memory, storage, and the one-time cost of labeling and encoding.
Every query is timed on its own, the way the web server answers it, rather
than as one batched matrix product.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import platform
import resource
import time
import zipfile

import numpy as np

from .stats import latency_summary

DIMENSIONS = ("plot", "themes", "characters", "tone", "narrative_structure")
ROOT = Path(__file__).resolve().parents[1]
TMDB_CATALOG_SIZE = 87_585  # Films reachable through TMDB at Milestone 1.


def _top(scores: np.ndarray, query: int, k: int) -> np.ndarray:
    scores[query] = -np.inf
    candidates = np.argpartition(scores, -k)[-k:]
    return candidates[np.argsort(-scores[candidates])]


def single_query(single: np.ndarray, query: int, k: int) -> np.ndarray:
    return _top(single @ single[query], query, k)


def multi_query(dimensions: list[np.ndarray], weights: np.ndarray, query: int, k: int) -> np.ndarray:
    # Mirrors webapp.server.RecommenderApp.recommend: zero-weight dimensions are skipped.
    total = float(weights.sum())
    scores = np.zeros(dimensions[0].shape[0], dtype=np.float32)
    for matrix, weight in zip(dimensions, weights):
        if weight:
            scores += (weight / total) * (matrix @ matrix[query])
    return _top(scores, query, k)


def random_slider_weights(rng: np.random.Generator) -> np.ndarray:
    """Weights a viewer could set: 0-3 in 0.1 steps, about one in five sliders at zero."""
    weights = np.round(rng.uniform(0, 3, size=len(DIMENSIONS)), 1)
    weights[rng.random(len(DIMENSIONS)) < 0.2] = 0.0
    if weights.sum() == 0:
        weights[rng.integers(len(DIMENSIONS))] = 1.0
    return weights


def _time(function, queries: np.ndarray, warmup: int = 20) -> list[float]:
    for query in queries[:warmup]:
        function(int(query))
    timings = []
    for query in queries:
        started = time.perf_counter()
        function(int(query))
        timings.append(time.perf_counter() - started)
    return timings


def measure(single: np.ndarray, dimensions: list[np.ndarray], queries: np.ndarray, k: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    random_weights = {int(q): random_slider_weights(rng) for q in queries}
    equal = np.ones(len(DIMENSIONS), dtype=np.float32)
    return {
        "single": latency_summary(_time(lambda q: single_query(single, q, k), queries)),
        "multi_equal_weights": latency_summary(_time(lambda q: multi_query(dimensions, equal, q, k), queries)),
        "multi_random_weights": latency_summary(
            _time(lambda q: multi_query(dimensions, random_weights[q], q, k), queries)
        ),
    }


def _synthetic(rows: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    matrix = rng.standard_normal((rows, dim), dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix


def scaling(
    single: np.ndarray,
    dimensions: list[np.ndarray],
    sizes: list[int],
    query_count: int,
    seed: int,
) -> list[dict]:
    """Latency versus catalog size. Sizes above the real catalog use random unit vectors."""
    rng = np.random.default_rng(seed)
    rows, dim = single.shape
    results = []
    for size in sizes:
        if size <= rows:
            pick = np.sort(rng.choice(rows, size=size, replace=False))
            sized_single = single[pick]
            sized_dims = [matrix[pick] for matrix in dimensions]
            source = "subsample"
        else:
            sized_single = _synthetic(size, dim, rng)
            sized_dims = [_synthetic(size, dim, rng) for _ in dimensions]
            source = "synthetic"
        queries = rng.integers(0, size, size=query_count)
        timings = measure(sized_single, sized_dims, queries, 10, seed)
        results.append(
            {
                "movies": size,
                "source": source,
                "vector_mb": vector_bytes(size, dim) / 2**20,
                "single_p95_ms": timings["single"]["p95_ms"],
                "multi_random_p95_ms": timings["multi_random_weights"]["p95_ms"],
                "multi_random_mean_ms": timings["multi_random_weights"]["mean_ms"],
            }
        )
        del sized_single, sized_dims
    return results


def vector_bytes(movies: int, dim: int, arrays: int = 1 + len(DIMENSIONS), itemsize: int = 4) -> int:
    return movies * dim * itemsize * arrays


def storage(artifact: Path, baseline: Path | None) -> dict:
    members = {}
    with zipfile.ZipFile(artifact) as archive:
        for info in archive.infolist():
            members[info.filename.removesuffix(".npy")] = {
                "compressed_bytes": info.compress_size,
                "uncompressed_bytes": info.file_size,
            }
    narrative = sum(members[name]["compressed_bytes"] for name in DIMENSIONS if name in members)
    single_in_artifact = members.get("single", {}).get("compressed_bytes", 0)
    baseline_bytes = baseline.stat().st_size if baseline and baseline.exists() else None
    single_reference = single_in_artifact or baseline_bytes
    return {
        "artifact_bytes": artifact.stat().st_size,
        "artifact_members": members,
        "single_bytes": single_reference,
        "narrative_bytes": narrative,
        "baseline_npy_bytes": baseline_bytes,
        "narrative_to_single_ratio": narrative / single_reference if single_reference else None,
        "note": "The .npz holds the single vectors too; narrative_bytes counts only the five narrative arrays.",
    }


def offline_cost(artifact: Path) -> dict:
    """One-time costs: LLM labeling speed from labeler run logs and encoder build timings."""
    runs = []
    for path in sorted((ROOT / "runs").glob("*/status.json")):
        try:
            status = json.loads(path.read_text(encoding="utf-8"))
            started = datetime.fromisoformat(status["started_at"])
            updated = datetime.fromisoformat(status["updated_at"])
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            continue
        processed = int(status.get("completed") or 0) + int(status.get("failed") or 0)
        seconds = (updated - started).total_seconds()
        rate = status.get("items_per_minute")
        runs.append(
            {
                "run_id": status.get("run_id", path.parent.name),
                "model_id": status.get("model_id"),
                "status": status.get("status"),
                "processed": processed,
                "failed": int(status.get("failed") or 0),
                "wall_seconds": seconds,
                "items_per_minute": rate,
                "seconds_per_film": 60.0 / rate if rate else None,
            }
        )
    useful = [run for run in runs if run["seconds_per_film"] and run["processed"] >= 50]
    build_path = artifact.with_suffix(".build.json")
    build = None
    if build_path.exists():
        try:
            build = json.loads(build_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            build = None
    return {
        "labeling_runs": runs,
        "labeling_median_seconds_per_film": (
            float(np.median([run["seconds_per_film"] for run in useful])) if useful else None
        ),
        "labeling_note": "Runs resuming saved labels report inflated speed; prefer runs that labeled new films.",
        "encoding": build,
    }


def benchmark(args: argparse.Namespace) -> dict:
    artifact = Path(args.artifact)
    if not artifact.exists():
        raise SystemExit(f"Missing {artifact}. Build the narrative embedding artifact first.")
    started = time.perf_counter()
    data = np.load(artifact)
    single = np.ascontiguousarray(data["single"], dtype=np.float32)
    dimensions = [np.ascontiguousarray(data[name], dtype=np.float32) for name in DIMENSIONS]
    load_seconds = time.perf_counter() - started
    movies, dim = single.shape
    rng = np.random.default_rng(args.seed)
    queries = rng.integers(0, movies, size=args.queries)

    headline = measure(single, dimensions, queries, 10, args.seed)
    by_k = {}
    for k in args.k_values:
        if k < movies:
            timings = measure(single, dimensions, queries, k, args.seed)
            by_k[str(k)] = {name: value["p95_ms"] for name, value in timings.items()}

    sizes = sorted({size for size in args.scale_sizes if size > 0} | {movies})
    scale = scaling(single, dimensions, sizes, min(args.queries, 100), args.seed) if not args.skip_scaling else []

    memory = {
        "single_vectors_mb": single.nbytes / 2**20,
        "narrative_vectors_mb": sum(matrix.nbytes for matrix in dimensions) / 2**20,
        "total_vectors_mb": (single.nbytes + sum(matrix.nbytes for matrix in dimensions)) / 2**20,
        "dtype": str(single.dtype),
        "dimension": dim,
        "peak_process_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "projected_total_mb_at_tmdb_scale": vector_bytes(TMDB_CATALOG_SIZE, dim) / 2**20,
    }

    multi_p95 = headline["multi_random_weights"]["p95_ms"]
    budget = {
        "latency_p95_ms_limit": args.latency_budget_ms,
        "memory_mb_limit": args.memory_budget_mb,
        "measured_multi_random_p95_ms": multi_p95,
        "measured_total_vectors_mb": memory["total_vectors_mb"],
        "latency_pass": multi_p95 <= args.latency_budget_ms,
        "memory_pass": memory["total_vectors_mb"] <= args.memory_budget_mb,
        "largest_catalog_within_memory": int(args.memory_budget_mb * 2**20 // vector_bytes(1, dim)),
        "largest_measured_catalog_within_latency": max(
            (row["movies"] for row in scale if row["multi_random_p95_ms"] <= args.latency_budget_ms), default=None
        ),
    }
    budget["pass"] = budget["latency_pass"] and budget["memory_pass"]

    single_mean = headline["single"]["mean_ms"]
    return {
        "artifact": str(artifact),
        "movies": movies,
        "queries": args.queries,
        "machine": {"platform": platform.platform(), "processor": platform.processor() or platform.machine(),
                    "numpy": np.__version__},
        "artifact_load_seconds": load_seconds,
        "latency_k10": headline,
        "multi_to_single_latency_ratio": (
            headline["multi_random_weights"]["mean_ms"] / single_mean if single_mean else None
        ),
        "p95_ms_by_k": by_k,
        "scaling": scale,
        "memory": memory,
        "storage": storage(artifact, Path(args.baseline_embeddings) if args.baseline_embeddings else None),
        "offline_cost": offline_cost(artifact),
        "budget": budget,
        # Kept so older scripts reading the previous benchmark report still work.
        "single_mean_ms": single_mean,
        "multi_mean_ms": headline["multi_equal_weights"]["mean_ms"],
        "single_qps": headline["single"]["serial_qps"],
        "multi_qps": headline["multi_equal_weights"]["serial_qps"],
    }


def _print_summary(result: dict) -> None:
    latency = result["latency_k10"]
    print(f"\n{result['movies']:,} films, {result['queries']} single-query requests, k=10")
    for name, row in latency.items():
        print(f"  {name:<22} mean {row['mean_ms']:.3f} ms  p95 {row['p95_ms']:.3f} ms  {row['serial_qps']:,.0f} q/s")
    memory, budget = result["memory"], result["budget"]
    print(f"  vectors in RAM: {memory['total_vectors_mb']:.1f} MB "
          f"(narrative {memory['narrative_vectors_mb']:.1f} MB, single {memory['single_vectors_mb']:.1f} MB)")
    ratio = result["storage"]["narrative_to_single_ratio"]
    if ratio:
        print(f"  on-disk narrative/single storage ratio: {ratio:.2f}x")
    for row in result["scaling"]:
        print(f"  scale {row['movies']:>7,} ({row['source']:<9}) multi p95 {row['multi_random_p95_ms']:.2f} ms, "
              f"{row['vector_mb']:.0f} MB")
    verdict = "PASS" if budget["pass"] else "FAIL"
    print(f"  budget (p95 <= {budget['latency_p95_ms_limit']} ms, vectors <= {budget['memory_mb_limit']} MB): {verdict}")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--artifact", default="artifacts/movie_embeddings.npz")
    parser.add_argument("--baseline-embeddings", default="data/processed/embeddings.npy")
    parser.add_argument("--queries", type=int, default=500)
    parser.add_argument("--seed", type=int, default=585)
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 20, 50])
    parser.add_argument("--scale-sizes", type=int, nargs="+", default=[1000, 2500, 25_000, 50_000, TMDB_CATALOG_SIZE])
    parser.add_argument("--skip-scaling", action="store_true")
    parser.add_argument("--latency-budget-ms", type=float, default=100.0,
                        help="p95 limit for one narrative request with random weights (default 100 ms).")
    parser.add_argument("--memory-budget-mb", type=float, default=256.0,
                        help="Limit for all vectors held in RAM (default 256 MB).")
    parser.add_argument("--report", default="reports/reconstruction_benchmark.json")


def run(args: argparse.Namespace) -> None:
    if args.queries < 1:
        raise SystemExit("--queries must be positive")
    result = benchmark(args)
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _print_summary(result)
    print(f"Saved report to {report}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Latency, memory, storage, and budget benchmark.")
    add_arguments(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
