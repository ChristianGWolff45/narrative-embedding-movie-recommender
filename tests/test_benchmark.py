import argparse
import json

import numpy as np

from trainer import benchmark as bm


def _args(synthetic, tmp_path, **overrides):
    values = dict(
        artifact=str(synthetic["artifact"]), baseline_embeddings=str(synthetic["baseline"]),
        queries=50, seed=1, k_values=[5, 10, 500], scale_sizes=[100, 600], skip_scaling=False,
        latency_budget_ms=100.0, memory_budget_mb=256.0, report=str(tmp_path / "bench.json"),
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_multi_query_matches_weighted_cosine(synthetic):
    arrays = synthetic["arrays"]
    dims = [arrays[name] for name in bm.DIMENSIONS]
    weights = np.array([2.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    expected = sum(w / weights.sum() * (m @ m[3]) for w, m in zip(weights, dims))
    expected[3] = -np.inf
    top = bm.multi_query(dims, weights, 3, 10)
    assert list(top) == list(np.argsort(-expected)[:10])
    assert 3 not in top


def test_random_slider_weights_stay_in_ui_range():
    rng = np.random.default_rng(0)
    for _ in range(500):
        weights = bm.random_slider_weights(rng)
        assert weights.min() >= 0 and weights.max() <= 3 and weights.sum() > 0


def test_benchmark_report(synthetic, tmp_path):
    args = _args(synthetic, tmp_path)
    bm.run(args)
    report = json.loads((tmp_path / "bench.json").read_text())
    assert report["movies"] == 300
    assert set(report["latency_k10"]) == {"single", "multi_equal_weights", "multi_random_weights"}
    assert set(report["p95_ms_by_k"]) == {"5", "10"}  # k=500 exceeds the catalog and is skipped
    assert [row["source"] for row in report["scaling"]] == ["subsample", "subsample", "synthetic"]
    storage = report["storage"]
    assert storage["narrative_bytes"] > storage["single_bytes"] > 0
    assert 3 < storage["narrative_to_single_ratio"] < 7
    assert report["budget"]["pass"] is True
    assert report["memory"]["total_vectors_mb"] == np.float32().itemsize * 300 * 16 * 6 / 2**20


def test_budget_failure_is_reported(synthetic, tmp_path):
    report = bm.benchmark(_args(synthetic, tmp_path, latency_budget_ms=0.0, memory_budget_mb=0.001, skip_scaling=True))
    assert report["budget"]["latency_pass"] is False
    assert report["budget"]["memory_pass"] is False
    assert report["budget"]["pass"] is False
