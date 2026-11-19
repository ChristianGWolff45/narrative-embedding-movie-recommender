from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd

from .stats import describe, holm_adjust, paired_comparison


ROOT = Path(__file__).resolve().parents[1]
RATINGS_PATH = ROOT / "data" / "raw" / "ml-32m" / "ratings.csv"
DIMENSIONS = ("plot", "themes", "characters", "tone", "narrative_structure")


class RatingIndex:
    def __init__(self, users: dict[int, np.ndarray], liked: dict[int, np.ndarray], fans: dict[int, np.ndarray]):
        self.users = users
        self.liked = liked
        self.fans = fans

    def like_rate(self, query_id: int, recommendation_id: int, minimum_overlap: int) -> tuple[float, int] | None:
        fans = self.fans.get(query_id)
        users = self.users.get(recommendation_id)
        liked = self.liked.get(recommendation_id)
        if fans is None or users is None or len(fans) == 0 or len(users) == 0:
            return None
        positions = np.searchsorted(users, fans)
        in_range = positions < len(users)
        positions = positions[in_range]
        if len(positions) == 0:
            return None
        matched = users[positions] == fans[in_range]
        overlap = int(matched.sum())
        if overlap < minimum_overlap:
            return None
        return float(liked[positions][matched].mean()), overlap


def load_rating_index(movie_ids: set[int], like_threshold: float) -> RatingIndex:
    if not RATINGS_PATH.exists():
        raise SystemExit(f"Missing {RATINGS_PATH}. Download MovieLens 32M first.")
    print("Loading MovieLens ratings for the model catalog...", flush=True)
    ratings = pd.read_csv(
        RATINGS_PATH,
        usecols=["userId", "movieId", "rating"],
        dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32},
    )
    ratings = ratings[ratings["movieId"].isin(movie_ids)]
    ratings.sort_values(["movieId", "userId"], inplace=True)
    movie_values = ratings["movieId"].to_numpy()
    user_values = ratings["userId"].to_numpy()
    liked_values = ratings["rating"].to_numpy() >= like_threshold
    del ratings
    if len(movie_values) == 0:
        raise SystemExit("No MovieLens ratings matched the embedding catalog.")

    boundaries = np.flatnonzero(movie_values[1:] != movie_values[:-1]) + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), boundaries))
    ends = np.concatenate((boundaries, np.array([len(movie_values)], dtype=np.int64)))
    users: dict[int, np.ndarray] = {}
    liked: dict[int, np.ndarray] = {}
    fans: dict[int, np.ndarray] = {}
    for start, end in zip(starts, ends):
        movie_id = int(movie_values[start])
        users[movie_id] = user_values[start:end]
        liked[movie_id] = liked_values[start:end]
        fans[movie_id] = user_values[start:end][liked_values[start:end]]
    print(f"Ratings loaded: {len(users):,} catalog movies, {len(movie_values):,} ratings", flush=True)
    return RatingIndex(users, liked, fans)


def _top_indices(scores: np.ndarray, query_index: int, k: int) -> list[int]:
    scores = scores.copy()
    scores[query_index] = -np.inf
    count = min(k, len(scores) - 1)
    if count <= 0:
        return []
    candidates = np.argpartition(scores, -count)[-count:]
    return sorted((int(index) for index in candidates), key=lambda index: scores[index], reverse=True)


def _genre_scores(genres: list[set[str]], query_index: int) -> np.ndarray:
    query = genres[query_index]
    scores = np.zeros(len(genres), dtype=np.float32)
    for index, candidate in enumerate(genres):
        union = query | candidate
        if union:
            scores[index] = len(query & candidate) / len(union)
    return scores


def _popularity(popularity: np.ndarray, query_index: int, k: int) -> list[int]:
    order = np.argsort(-popularity, kind="stable")
    return [int(index) for index in order if index != query_index][:k]


def _matched_random(
    query_index: int,
    popularity: np.ndarray,
    recommendations: list[int],
    rng: random.Random,
) -> list[int]:
    banned = {query_index, *recommendations}
    result: list[int] = []
    for recommendation in recommendations:
        target = float(popularity[recommendation])
        low, high = target * 0.5, target * 2.0
        allowed = [
            index
            for index, count in enumerate(popularity)
            if low <= count <= high and index not in banned
        ]
        if not allowed:
            order = np.argsort(np.abs(np.log1p(popularity) - np.log1p(target)))
            allowed = [int(index) for index in order if int(index) not in banned]
        if not allowed:
            break
        choice = rng.choice(allowed)
        banned.add(choice)
        result.append(choice)
    return result


class RelevanceCache:
    """Memoizes co-like rates; sweeps score the same pairs many times."""

    def __init__(self, ratings: RatingIndex, metadata: list[dict], minimum_overlap: int) -> None:
        self.ratings = ratings
        self.metadata = metadata
        self.minimum_overlap = minimum_overlap
        self._cache: dict[tuple[int, int], tuple[float, int] | None] = {}

    def rate(self, query_index: int, recommendation_index: int) -> tuple[float, int] | None:
        key = (query_index, recommendation_index)
        if key not in self._cache:
            self._cache[key] = self.ratings.like_rate(
                int(self.metadata[query_index]["movie_id"]),
                int(self.metadata[recommendation_index]["movie_id"]),
                self.minimum_overlap,
            )
        return self._cache[key]


def _query_relevance(
    cache: RelevanceCache,
    query_index: int,
    recommendations: list[int],
) -> tuple[float | None, int, int]:
    values = []
    scored = 0
    overlaps = 0
    for index in recommendations:
        result = cache.rate(query_index, index)
        if result is not None:
            rate, overlap = result
            values.append(rate)
            overlaps += overlap
            scored += 1
    return (float(np.mean(values)) if values else None, scored, overlaps)


def _genre_diversity(indices: list[int], genres: list[set[str]]) -> tuple[float | None, int]:
    distances = []
    covered: set[str] = set()
    for index in indices:
        covered.update(genres[index])
    for offset, first in enumerate(indices):
        for second in indices[offset + 1 :]:
            union = genres[first] | genres[second]
            if union:
                distances.append(1.0 - len(genres[first] & genres[second]) / len(union))
    diversity = float(np.mean(distances)) if distances else None
    return diversity, len(covered)


_describe = describe

BASELINES = ("single", "genre", "popular", "random_matched")
PRIMARY_BASELINES = ("single", "genre")


def weight_profiles(args: argparse.Namespace) -> list[tuple[str, dict[str, float]]]:
    """Named weight settings to sweep: the CLI weights first, then presets, then random draws."""
    cli = {
        "plot": args.plot,
        "themes": args.themes,
        "characters": args.characters,
        "tone": args.tone,
        "narrative_structure": args.structure,
    }
    if any(value < 0 for value in cli.values()) or sum(cli.values()) == 0:
        raise SystemExit("Narrative weights must be nonnegative and at least one must be positive.")
    equal = {name: 1.0 for name in DIMENSIONS}
    profiles: list[tuple[str, dict[str, float]]] = [("equal" if cli == equal else "custom", cli)]
    if not args.no_sweep:
        if cli != equal:
            profiles.append(("equal", equal))
        for name in DIMENSIONS:
            profiles.append((f"{name}_only", {other: float(other == name) for other in DIMENSIONS}))
        rng = np.random.default_rng(args.seed + 1)
        for number in range(args.random_profiles):
            draw = rng.dirichlet(np.ones(len(DIMENSIONS)))
            profiles.append((f"random_{number + 1:02d}", {name: float(value) for name, value in zip(DIMENSIONS, draw)}))
    return profiles


def _score_lists(
    rows: dict[int, list[int]],
    cache: RelevanceCache,
    genres: list[set[str]],
) -> tuple[dict[int, float | None], dict[int, float | None], dict]:
    relevance: dict[int, float | None] = {}
    diversity: dict[int, float | None] = {}
    scored_pairs = 0
    possible_pairs = 0
    all_overlaps = 0
    genre_coverage = []
    for query_index, recommendations in rows.items():
        mean_rate, scored, overlaps = _query_relevance(cache, query_index, recommendations)
        relevance[query_index] = mean_rate
        scored_pairs += scored
        possible_pairs += len(recommendations)
        all_overlaps += overlaps
        value, genre_count = _genre_diversity(recommendations, genres)
        diversity[query_index] = value
        genre_coverage.append(float(genre_count))
    detail = {
        "recommendation_coverage": scored_pairs / possible_pairs if possible_pairs else 0.0,
        "scored_recommendations": scored_pairs,
        "possible_recommendations": possible_pairs,
        "mean_shared_raters_for_scored_recommendations": (
            all_overlaps / scored_pairs if scored_pairs else None
        ),
        "co_like_rate": describe([v for v in relevance.values() if v is not None]),
        "genre_intra_list_diversity": describe([v for v in diversity.values() if v is not None]),
        "mean_unique_genres_per_list": float(np.mean(genre_coverage)) if genre_coverage else None,
    }
    return relevance, diversity, detail


def _paired(
    treatment: dict[int, float | None],
    control: dict[int, float | None],
    seed: int,
    resamples: int,
) -> dict:
    keys = [key for key in treatment if treatment[key] is not None and control.get(key) is not None]
    return paired_comparison(
        [treatment[key] for key in keys],
        [control[key] for key in keys],
        seed=seed,
        resamples=resamples,
    )


def _compare_against(
    multi: tuple[dict, dict],
    baselines: dict[str, tuple[dict, dict]],
    names: tuple[str, ...],
    seed: int,
    resamples: int,
) -> dict:
    result = {}
    for name in names:
        result[name] = {
            "co_like_rate": _paired(multi[0], baselines[name][0], seed, resamples),
            "genre_diversity": _paired(multi[1], baselines[name][1], seed, resamples),
        }
    return result


def _apply_holm(comparisons: list[dict]) -> None:
    adjusted = holm_adjust([item["p_value"] for item in comparisons])
    for item, value in zip(comparisons, adjusted):
        item["p_value_holm"] = value


def _comparison_cells(block: dict) -> list[dict]:
    return [metric for baseline in block.values() for metric in baseline.values()]


def evaluate(args: argparse.Namespace) -> dict:
    artifact = Path(args.artifact)
    if not artifact.exists():
        raise SystemExit(f"Missing {artifact}. Build the narrative embedding artifact first.")
    metadata_path = artifact.with_suffix(".metadata.json")
    if not metadata_path.exists():
        raise SystemExit(f"Missing {metadata_path}.")
    arrays = np.load(artifact)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = {"single", *DIMENSIONS}
    if not required.issubset(arrays.files):
        raise SystemExit(f"Artifact must contain: {', '.join(sorted(required))}")
    if len(metadata) != arrays["single"].shape[0]:
        raise SystemExit("Metadata and embedding rows do not match.")
    if any(not item.get("movie_id") for item in metadata):
        raise SystemExit("Embedding metadata is missing MovieLens movie IDs; rebuild from MovieLens-labeled records.")

    genres = [set(item.get("genres", [])) for item in metadata]
    popularity = np.asarray([int(item.get("n_ratings") or 0) for item in metadata], dtype=np.int64)
    rating_index = load_rating_index({int(item["movie_id"]) for item in metadata}, args.like_threshold)
    eligible = [
        index
        for index, item in enumerate(metadata)
        if len(rating_index.fans.get(int(item["movie_id"]), ())) >= args.min_overlap
    ]
    if not eligible:
        raise SystemExit("No query movie has enough liked MovieLens ratings.")
    rng = random.Random(args.seed)
    queries = eligible if args.queries >= len(eligible) else rng.sample(eligible, args.queries)
    cache = RelevanceCache(rating_index, metadata, args.min_overlap)
    profiles = weight_profiles(args)
    ks = list(dict.fromkeys(args.k))

    single_matrix = np.asarray(arrays["single"], dtype=np.float32)
    dimension_matrices = [np.asarray(arrays[name], dtype=np.float32) for name in DIMENSIONS]
    # Per-query, per-dimension cosine scores; every weight profile is a dot product over these.
    dimension_scores = {
        query: np.stack([matrix @ matrix[query] for matrix in dimension_matrices])
        for query in queries
    }

    def multi_lists(weights: dict[str, float], k: int) -> dict[int, list[int]]:
        vector = np.asarray([weights[name] for name in DIMENSIONS], dtype=np.float32)
        vector = vector / vector.sum()
        return {query: _top_indices(vector @ dimension_scores[query], query, k) for query in queries}

    headline_name, headline_weights = profiles[0]
    report_by_k = {}
    sweep = []
    headline = None
    for k in ks:
        random_rng = random.Random(args.seed * 1000 + k)
        lists: dict[str, dict[int, list[int]]] = {name: {} for name in BASELINES}
        for query in queries:
            lists["single"][query] = _top_indices(single_matrix @ single_matrix[query], query, k)
            lists["genre"][query] = _top_indices(_genre_scores(genres, query), query, k)
            lists["popular"][query] = _popularity(popularity, query, k)
            lists["random_matched"][query] = _matched_random(
                query, popularity, lists["single"][query], random_rng
            )
        baseline_scores = {}
        baseline_detail = {}
        for name in BASELINES:
            relevance, diversity, detail = _score_lists(lists[name], cache, genres)
            baseline_scores[name] = (relevance, diversity)
            baseline_detail[name] = detail
        report_by_k[k] = baseline_detail

        for profile_name, weights in profiles:
            relevance, diversity, detail = _score_lists(multi_lists(weights, k), cache, genres)
            is_headline = profile_name == headline_name and k == ks[0]
            if is_headline:
                headline = {
                    "scores": {**baseline_scores, "multi": (relevance, diversity)},
                    "detail": {**baseline_detail, "multi": detail},
                }
            else:
                sweep.append(
                    {
                        "profile": profile_name,
                        "weights": weights,
                        "k": k,
                        "multi": detail,
                        "versus": _compare_against(
                            (relevance, diversity), baseline_scores, PRIMARY_BASELINES, args.seed, args.resamples
                        ),
                    }
                )

    assert headline is not None
    methods = headline["detail"]
    scores = headline["scores"]
    order = ("single", "multi", "genre", "popular", "random_matched")
    paired_queries = [
        query for query in queries if all(scores[name][0].get(query) is not None for name in order)
    ]
    for name in order:
        methods[name]["co_like_rate_paired"] = describe([float(scores[name][0][q]) for q in paired_queries])
        methods[name]["paired_query_count"] = len(paired_queries)
    primary = _compare_against(scores["multi"], {n: scores[n] for n in BASELINES}, BASELINES, args.seed, args.resamples)
    _apply_holm(_comparison_cells(primary))
    _apply_holm([cell for row in sweep for cell in _comparison_cells(row["versus"])])

    return {
        "artifact": str(artifact),
        "movies": len(metadata),
        "queries_requested": args.queries,
        "queries_evaluated": len(queries),
        "paired_queries": len(paired_queries),
        "k": ks[0],
        "k_values": ks,
        "like_threshold": args.like_threshold,
        "min_overlap": args.min_overlap,
        "seed": args.seed,
        "weights": headline_weights,
        "headline_profile": headline_name,
        "relevance_definition": "fraction of query fans who also rated a recommendation >= like_threshold",
        "diversity_definition": "pairwise Jaccard distance between recommended genre sets",
        "statistics": (
            "Paired by query. mean_difference = narrative minus baseline; 95% percentile bootstrap CI; "
            "two-sided sign-flip permutation p-value; Holm-adjusted within the headline family and "
            "separately across the exploratory sweep."
        ),
        "methods": methods,
        "comparisons": primary,
        "baselines_by_k": {str(k): detail for k, detail in report_by_k.items()},
        "sweep": sweep,
    }


def _print_summary(result: dict) -> None:
    print(f"\nHeadline ({result['headline_profile']} weights, k={result['k']}, {result['queries_evaluated']} queries)")
    for name, block in result["comparisons"].items():
        for metric, cell in block.items():
            if cell["n"] == 0:
                print(f"  narrative vs {name:<15} {metric:<16} no paired queries")
                continue
            print(
                f"  narrative vs {name:<15} {metric:<16} diff {cell['mean_difference']:+.4f} "
                f"[{cell['ci_low']:+.4f}, {cell['ci_high']:+.4f}] p_holm={cell['p_value_holm']:.4f} n={cell['n']}"
            )
    if result["sweep"]:
        print(f"\nSweep: {len(result['sweep'])} profile/k cells (narrative vs single co-like rate)")
        for row in result["sweep"]:
            cell = row["versus"]["single"]["co_like_rate"]
            if cell["n"]:
                print(
                    f"  k={row['k']:<3} {row['profile']:<26} diff {cell['mean_difference']:+.4f} "
                    f"[{cell['ci_low']:+.4f}, {cell['ci_high']:+.4f}] p_holm={cell['p_value_holm']:.4f}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare single, multi, genre, popularity, and random recommenders.")
    parser.add_argument("--artifact", default="artifacts/movie_embeddings.npz")
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--k", type=int, nargs="+", default=[10], help="One or more list lengths; the first is the headline.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--like-threshold", type=float, default=4.0)
    parser.add_argument("--min-overlap", type=int, default=30)
    parser.add_argument("--plot", type=float, default=1.0)
    parser.add_argument("--themes", type=float, default=1.0)
    parser.add_argument("--characters", type=float, default=1.0)
    parser.add_argument("--tone", type=float, default=1.0)
    parser.add_argument("--structure", type=float, default=1.0)
    parser.add_argument("--random-profiles", type=int, default=10, help="Random Dirichlet weightings to add to the sweep.")
    parser.add_argument("--no-sweep", action="store_true", help="Only evaluate the headline weights.")
    parser.add_argument("--resamples", type=int, default=10_000, help="Bootstrap and permutation resamples.")
    parser.add_argument("--output", default="reports/evaluation.json")
    args = parser.parse_args()
    if args.queries < 1 or any(k < 1 for k in args.k) or args.min_overlap < 1:
        parser.error("--queries, --k, and --min-overlap must be positive")
    if args.random_profiles < 0 or args.resamples < 100:
        parser.error("--random-profiles must be >= 0 and --resamples >= 100")
    result = evaluate(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _print_summary(result)
    print(f"Saved report to {output}")


if __name__ == "__main__":
    main()
