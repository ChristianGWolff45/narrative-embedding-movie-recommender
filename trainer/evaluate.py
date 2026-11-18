from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd


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


def _query_relevance(
    ratings: RatingIndex,
    query_id: int,
    recommendations: list[int],
    metadata: list[dict],
    minimum_overlap: int,
) -> tuple[float | None, int, int]:
    values = []
    scored = 0
    overlaps = 0
    for index in recommendations:
        result = ratings.like_rate(query_id, int(metadata[index]["movie_id"]), minimum_overlap)
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


def _describe(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "p10": None, "median": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
    }


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

    weights = {
        "plot": args.plot,
        "themes": args.themes,
        "characters": args.characters,
        "tone": args.tone,
        "narrative_structure": args.structure,
    }
    if any(value < 0 for value in weights.values()) or sum(weights.values()) == 0:
        raise SystemExit("Narrative weights must be nonnegative and at least one must be positive.")
    total_weight = sum(weights.values())

    query_lists: dict[str, dict[int, list[int]]] = {
        name: {} for name in ("single", "multi", "genre", "popular", "random_matched")
    }
    for query_index in queries:
        single_scores = arrays["single"] @ arrays["single"][query_index]
        multi_scores = sum(
            (weight / total_weight) * (arrays[name] @ arrays[name][query_index])
            for name, weight in weights.items()
        )
        query_lists["single"][query_index] = _top_indices(single_scores, query_index, args.k)
        query_lists["multi"][query_index] = _top_indices(multi_scores, query_index, args.k)
        query_lists["genre"][query_index] = _top_indices(_genre_scores(genres, query_index), query_index, args.k)
        query_lists["popular"][query_index] = _popularity(popularity, query_index, args.k)
        query_lists["random_matched"][query_index] = _matched_random(
            query_index, popularity, query_lists["single"][query_index], rng
        )

    relevance: dict[str, dict[int, float | None]] = {name: {} for name in query_lists}
    detail: dict[str, dict] = {}
    for name, rows in query_lists.items():
        scored_pairs = 0
        possible_pairs = 0
        all_overlaps = 0
        diversity_values = []
        genre_coverage = []
        for query_index, recommendations in rows.items():
            query_id = int(metadata[query_index]["movie_id"])
            mean_rate, scored, overlaps = _query_relevance(
                rating_index, query_id, recommendations, metadata, args.min_overlap
            )
            relevance[name][query_index] = mean_rate
            scored_pairs += scored
            possible_pairs += len(recommendations)
            all_overlaps += overlaps
            diversity, genre_count = _genre_diversity(recommendations, genres)
            if diversity is not None:
                diversity_values.append(diversity)
            genre_coverage.append(float(genre_count))
        detail[name] = {
            "recommendation_coverage": scored_pairs / possible_pairs if possible_pairs else 0.0,
            "scored_recommendations": scored_pairs,
            "possible_recommendations": possible_pairs,
            "mean_shared_raters_for_scored_recommendations": (
                all_overlaps / scored_pairs if scored_pairs else None
            ),
            "genre_intra_list_diversity": _describe(diversity_values),
            "mean_unique_genres_per_list": float(np.mean(genre_coverage)) if genre_coverage else None,
        }

    paired_queries = [
        index
        for index in queries
        if all(relevance[name][index] is not None for name in query_lists)
    ]
    for name in query_lists:
        paired_rates = [float(relevance[name][index]) for index in paired_queries]
        detail[name]["co_like_rate_paired"] = _describe(paired_rates)
        detail[name]["paired_query_count"] = len(paired_queries)

    return {
        "artifact": str(artifact),
        "movies": len(metadata),
        "queries_requested": args.queries,
        "queries_evaluated": len(queries),
        "paired_queries": len(paired_queries),
        "k": args.k,
        "like_threshold": args.like_threshold,
        "min_overlap": args.min_overlap,
        "weights": weights,
        "relevance_definition": "fraction of query fans who also rated a recommendation >= like_threshold",
        "diversity_definition": "pairwise Jaccard distance between recommended genre sets",
        "methods": detail,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare single, multi, genre, popularity, and random recommenders.")
    parser.add_argument("--artifact", default="artifacts/movie_embeddings.npz")
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--like-threshold", type=float, default=4.0)
    parser.add_argument("--min-overlap", type=int, default=30)
    parser.add_argument("--plot", type=float, default=1.0)
    parser.add_argument("--themes", type=float, default=1.0)
    parser.add_argument("--characters", type=float, default=1.0)
    parser.add_argument("--tone", type=float, default=1.0)
    parser.add_argument("--structure", type=float, default=1.0)
    parser.add_argument("--output", default="reports/evaluation.json")
    args = parser.parse_args()
    if args.queries < 1 or args.k < 1 or args.min_overlap < 1:
        parser.error("--queries, --k, and --min-overlap must be positive")
    result = evaluate(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved report to {output}")


if __name__ == "__main__":
    main()
