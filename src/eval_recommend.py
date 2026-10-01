"""Score title recommendations by whether fans of the query movie also like them.

A fan rated the query at least --like-threshold (default 4.0). Each recommendation
scores as the fraction of those fans who also rated it and rated it at least the
threshold. Pairs with fewer than --min-overlap shared raters are left unscored.

The same score is computed for the most-rated catalog movies and for random
catalog movies with a similar rating count. Reported means use only queries that
all three methods scored, so the lift is a paired comparison.

    python src/eval_recommend.py
    python src/eval_recommend.py --queries 50 -k 10 --seed 0
"""

from __future__ import annotations

import argparse
import random
import sys

import numpy as np
import pandas as pd

from config import RATINGS_PATH
from recommend import Recommender


def _configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


def _label(record: dict) -> str:
    release_date = record.get("release_date") or ""
    year = release_date[:4] if len(release_date) >= 4 else ""
    title = record.get("title") or "Untitled"
    return f"{title} ({year})" if year else title


class RatingIndex:
    """Per-movie user ids, aligned with a liked flag. Fans are users who liked it."""

    def __init__(
        self,
        users: dict[int, np.ndarray],
        liked: dict[int, np.ndarray],
        fans: dict[int, np.ndarray],
    ) -> None:
        self.users = users
        self.liked = liked
        self.fans = fans

    def like_rate(self, query_id: int, recommended_id: int, min_overlap: int) -> tuple[float, int] | None:
        fans = self.fans.get(query_id)
        rec_users = self.users.get(recommended_id)
        rec_liked = self.liked.get(recommended_id)
        if fans is None or rec_users is None or len(fans) == 0 or len(rec_users) == 0:
            return None
        positions = np.searchsorted(rec_users, fans)
        in_range = positions < len(rec_users)
        positions = positions[in_range]
        if len(positions) == 0:
            return None
        equal = rec_users[positions] == fans[in_range]
        overlap = int(equal.sum())
        if overlap < min_overlap:
            return None
        likes = int(rec_liked[positions][equal].sum())
        return likes / overlap, overlap


def load_rating_index(catalog_ids: set[int], like_threshold: float) -> RatingIndex:
    if not RATINGS_PATH.exists():
        raise SystemExit(f"Missing {RATINGS_PATH}.")
    print("Loading MovieLens ratings...", flush=True)
    ratings = pd.read_csv(
        RATINGS_PATH,
        usecols=["userId", "movieId", "rating"],
        dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32},
    )
    ratings = ratings[ratings["movieId"].isin(catalog_ids)]
    ratings.sort_values(["movieId", "userId"], inplace=True)
    ratings.reset_index(drop=True, inplace=True)

    movie_ids = ratings["movieId"].to_numpy()
    user_ids = ratings["userId"].to_numpy()
    liked_flags = ratings["rating"].to_numpy() >= like_threshold
    del ratings

    if len(movie_ids) == 0:
        raise SystemExit("No MovieLens ratings matched the catalog.")

    change = np.flatnonzero(movie_ids[1:] != movie_ids[:-1]) + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), change))
    ends = np.concatenate((change, np.array([len(movie_ids)], dtype=np.int64)))

    users: dict[int, np.ndarray] = {}
    liked: dict[int, np.ndarray] = {}
    fans: dict[int, np.ndarray] = {}
    for start, end in zip(starts, ends):
        movie_id = int(movie_ids[start])
        movie_users = user_ids[start:end]
        movie_liked = liked_flags[start:end]
        users[movie_id] = movie_users
        liked[movie_id] = movie_liked
        fans[movie_id] = movie_users[movie_liked]

    print(
        f"Ratings index: {len(users):,} movies, {len(user_ids):,} ratings, "
        f"{int(liked_flags.sum()):,} likes (>= {like_threshold:g})",
        flush=True,
    )
    return RatingIndex(users, liked, fans)


def _most_rated(query_index: int, k: int, popularity: list[int]) -> list[int]:
    picked = []
    for index in popularity:
        if index == query_index:
            continue
        picked.append(index)
        if len(picked) == k:
            break
    return picked


def _matched_random(
    query_index: int,
    recommendations: list[int],
    n_ratings: np.ndarray,
    rng: random.Random,
) -> list[int]:
    banned = {query_index, *recommendations}
    picked = []
    for rec_index in recommendations:
        target = float(n_ratings[rec_index])
        lo, hi = target * 0.5, target * 2.0
        candidates = np.flatnonzero((n_ratings >= lo) & (n_ratings <= hi))
        allowed = [int(index) for index in candidates if int(index) not in banned]
        if allowed:
            choice = rng.choice(allowed)
        else:
            order = np.argsort(np.abs(np.log1p(n_ratings) - np.log1p(target)))
            choice = next(int(index) for index in order if int(index) not in banned)
        picked.append(choice)
        banned.add(choice)
    return picked


def _score(
    index: RatingIndex,
    query_id: int,
    movie_ids: list[int],
    min_overlap: int,
) -> list[tuple[float, int] | None]:
    return [index.like_rate(query_id, movie_id, min_overlap) for movie_id in movie_ids]


def _query_mean(scores: list[tuple[float, int] | None]) -> float | None:
    rates = [score[0] for score in scores if score is not None]
    if not rates:
        return None
    return float(np.mean(rates))


def _coverage(score_rows: list[list[tuple[float, int] | None]]) -> tuple[int, int]:
    possible = sum(len(row) for row in score_rows)
    scored = sum(score is not None for row in score_rows for score in row)
    return scored, possible


def _summarize(name: str, means: list[float], scored: int, possible: int) -> None:
    coverage = scored / possible if possible else 0.0
    values = np.asarray(means, dtype=np.float64)
    print(name)
    print(f"  coverage:      {coverage:.1%} ({scored:,}/{possible:,})")
    if len(values) == 0:
        print("  mean like-rate: n/a")
        return
    print(
        "  like-rate:     "
        f"mean {values.mean():.3f}  "
        f"p50 {float(np.percentile(values, 50)):.3f}  "
        f"p10 {float(np.percentile(values, 10)):.3f}  "
        f"p90 {float(np.percentile(values, 90)):.3f}"
    )


def _print_example(
    records: list[dict],
    query_index: int,
    fan_count: int,
    recommendations: list[int],
    scores: list[tuple[float, int] | None],
) -> None:
    print(f"{_label(records[query_index])} — {fan_count:,} fans")
    for rec_index, score in zip(recommendations, scores):
        if score is None:
            print(f"  unscored   {_label(records[rec_index])}")
            continue
        rate, overlap = score
        print(f"  {rate:.3f}  n={overlap:,}  {_label(records[rec_index])}")


def main() -> None:
    _configure_stdout()
    parser = argparse.ArgumentParser(
        description="Evaluate title recommendations by co-like rate on MovieLens.",
    )
    parser.add_argument("--queries", type=int, default=200, help="Catalog movies to sample as queries")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--like-threshold", type=float, default=4.0)
    parser.add_argument("--min-overlap", type=int, default=30, help="Minimum fans who also rated the recommendation")
    parser.add_argument("--examples", type=int, default=3, help="Query movies to print with per-title scores")
    args = parser.parse_args()
    if args.queries < 1 or args.k < 1 or args.min_overlap < 1:
        parser.error("--queries, -k, and --min-overlap must be positive")

    engine = Recommender()
    records = engine.records
    catalog_ids = {int(record["movieId"]) for record in records}
    rating_index = load_rating_index(catalog_ids, args.like_threshold)

    eligible = [
        index
        for index, record in enumerate(records)
        if len(rating_index.fans.get(int(record["movieId"]), ())) >= args.min_overlap
    ]
    if not eligible:
        raise SystemExit("No catalog movie has enough likes to use as a query.")
    rng = random.Random(args.seed)
    if args.queries >= len(eligible):
        print(f"Using all {len(eligible):,} eligible catalog movies.", flush=True)
        queries = eligible
    else:
        queries = rng.sample(eligible, args.queries)

    n_ratings = np.asarray([record.get("n_ratings") or 0 for record in records], dtype=np.float64)
    popularity = sorted(range(len(records)), key=lambda index: n_ratings[index], reverse=True)

    recommender_scores: list[list[tuple[float, int] | None]] = []
    random_scores: list[list[tuple[float, int] | None]] = []
    popular_scores: list[list[tuple[float, int] | None]] = []
    examples: list[tuple[int, int, list[int], list[tuple[float, int] | None]]] = []

    for query_number, query_index in enumerate(queries, start=1):
        query_id = int(records[query_index]["movieId"])
        neighbors = engine.neighbors(engine.embeddings[query_index], args.k, {query_index})
        recommendations = [index for _score_value, index in neighbors]
        rec_ids = [int(records[index]["movieId"]) for index in recommendations]
        random_picks = _matched_random(query_index, recommendations, n_ratings, rng)
        popular_picks = _most_rated(query_index, args.k, popularity)

        rec_scores = _score(rating_index, query_id, rec_ids, args.min_overlap)
        rand_scores = _score(
            rating_index,
            query_id,
            [int(records[index]["movieId"]) for index in random_picks],
            args.min_overlap,
        )
        pop_scores = _score(
            rating_index,
            query_id,
            [int(records[index]["movieId"]) for index in popular_picks],
            args.min_overlap,
        )
        recommender_scores.append(rec_scores)
        random_scores.append(rand_scores)
        popular_scores.append(pop_scores)
        if len(examples) < args.examples:
            examples.append((query_index, len(rating_index.fans[query_id]), recommendations, rec_scores))
        if query_number % 25 == 0 or query_number == len(queries):
            print(f"Scored {query_number:,}/{len(queries):,} queries", flush=True)

    paired = [
        index
        for index in range(len(queries))
        if _query_mean(recommender_scores[index]) is not None
        and _query_mean(random_scores[index]) is not None
        and _query_mean(popular_scores[index]) is not None
    ]
    recommender_means = [_query_mean(recommender_scores[index]) for index in paired]
    random_means = [_query_mean(random_scores[index]) for index in paired]
    popular_means = [_query_mean(popular_scores[index]) for index in paired]

    print()
    print(
        f"Queries: {len(queries):,}   k: {args.k}   "
        f"paired: {len(paired):,}   like >= {args.like_threshold:g}   "
        f"min overlap {args.min_overlap}"
    )
    print("Like-rate is the mean of per-query means, on queries all three methods scored.")
    print()
    _summarize("Recommender", recommender_means, *_coverage(recommender_scores))
    _summarize("Random, similar rating count", random_means, *_coverage(random_scores))
    _summarize("Most-rated catalog movies", popular_means, *_coverage(popular_scores))
    print()
    if recommender_means and float(np.mean(random_means)) > 0 and float(np.mean(popular_means)) > 0:
        print(f"Lift vs random:     {float(np.mean(recommender_means)) / float(np.mean(random_means)):.2f}")
        print(f"Lift vs most-rated: {float(np.mean(recommender_means)) / float(np.mean(popular_means)):.2f}")
    else:
        print("Lift: n/a")

    if examples:
        print()
        print("Examples")
        for query_index, fan_count, recommendations, scores in examples:
            print()
            _print_example(records, query_index, fan_count, recommendations, scores)


if __name__ == "__main__":
    main()
