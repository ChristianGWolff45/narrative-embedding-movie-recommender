"""Synthetic catalog, ratings, and artifact so the pipeline runs without real data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DIMENSIONS = ("plot", "themes", "characters", "tone", "narrative_structure")
GENRES = ["Drama", "Comedy", "Action", "Horror", "Romance", "Thriller", "Animation", "Documentary"]


def _unit(rows: np.ndarray) -> np.ndarray:
    return (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(np.float32)


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory) -> dict:
    """300 films in 6 latent taste clusters; users like films in their cluster.

    The narrative vectors carry the cluster signal more cleanly than the single
    vector, so a correct evaluator should find narrative >= single.
    """
    root: Path = tmp_path_factory.mktemp("synthetic")
    rng = np.random.default_rng(7)
    films, clusters, dim = 300, 6, 16
    cluster = rng.integers(0, clusters, size=films)
    centers = rng.normal(size=(clusters, dim))
    arrays = {"single": _unit(centers[cluster] + rng.normal(scale=2.0, size=(films, dim)))}
    for name in DIMENSIONS:
        arrays[name] = _unit(centers[cluster] + rng.normal(scale=0.8, size=(films, dim)))
    artifact = root / "artifacts" / "movie_embeddings.npz"
    artifact.parent.mkdir()
    np.savez_compressed(artifact, **arrays)
    metadata = []
    for index in range(films):
        picked = rng.choice(len(GENRES), size=rng.integers(1, 4), replace=False)
        metadata.append(
            {
                "tmdb_id": 1000 + index,
                "movie_id": 1 + index,
                "n_ratings": int(rng.integers(500, 20000)),
                "title": f"Film {index}",
                "genres": [GENRES[g] for g in picked],
                "release_date": f"{1960 + index % 60}-01-01",
            }
        )
    artifact.with_suffix(".metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    rows = []
    for user in range(1, 1201):
        taste = rng.integers(0, clusters)
        seen = rng.choice(films, size=60, replace=False)
        for film in seen:
            liked = cluster[film] == taste
            rating = rng.choice([4.0, 4.5, 5.0]) if liked or rng.random() < 0.15 else rng.choice([1.0, 2.0, 3.0])
            rows.append((user, film + 1, rating))
    ratings = root / "ratings.csv"
    pd.DataFrame(rows, columns=["userId", "movieId", "rating"]).to_csv(ratings, index=False)
    baseline = root / "embeddings.npy"
    np.save(baseline, arrays["single"])
    return {"root": root, "artifact": artifact, "ratings": ratings, "baseline": baseline,
            "metadata": metadata, "arrays": arrays}
