"""Paired statistics for comparing recommenders on the same queries.

Only numpy is used so the evaluation runs in the same environment as the rest
of the pipeline.
"""

from __future__ import annotations

import numpy as np


def describe(values) -> dict:
    """Mean and spread for a list of numbers, with None when empty."""
    if len(values) == 0:
        return {"mean": None, "p10": None, "median": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
    }


def latency_summary(seconds) -> dict:
    """Latency percentiles in milliseconds plus serial throughput."""
    array = np.asarray(seconds, dtype=np.float64) * 1000.0
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "max_ms": float(array.max()),
        "serial_qps": float(1000.0 / array.mean()) if array.mean() > 0 else None,
    }


def paired_comparison(
    treatment,
    control,
    *,
    resamples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict:
    """Compare two methods scored on the same queries.

    Returns the mean difference (treatment minus control), a percentile
    bootstrap confidence interval, a two-sided sign-flip permutation p-value,
    and win/tie/loss counts. Pairs must already be aligned by query.
    """
    a = np.asarray(treatment, dtype=np.float64)
    b = np.asarray(control, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Paired samples must have the same length.")
    n = int(a.size)
    if n == 0:
        return {"n": 0, "mean_difference": None, "ci_low": None, "ci_high": None, "p_value": None,
                "wins": 0, "ties": 0, "losses": 0, "effect_size_dz": None}
    diff = a - b
    observed = float(diff.mean())
    rng = np.random.default_rng(seed)

    # Bootstrap the mean difference by resampling queries.
    picks = rng.integers(0, n, size=(resamples, n))
    boot = diff[picks].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(boot, [alpha, 1.0 - alpha])

    # Sign-flip randomization test: under H0 each paired difference is
    # equally likely to have either sign.
    signs = rng.choice(np.array([-1.0, 1.0]), size=(resamples, n))
    null = (signs * diff).mean(axis=1)
    extreme = int(np.sum(np.abs(null) >= abs(observed) - 1e-12))
    p_value = (extreme + 1) / (resamples + 1)

    spread = float(diff.std(ddof=1)) if n > 1 else 0.0
    return {
        "n": n,
        "mean_difference": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence": confidence,
        "p_value": float(p_value),
        "wins": int(np.sum(diff > 1e-12)),
        "ties": int(np.sum(np.abs(diff) <= 1e-12)),
        "losses": int(np.sum(diff < -1e-12)),
        "effect_size_dz": observed / spread if spread > 0 else None,
    }


def holm_adjust(p_values: list[float | None]) -> list[float | None]:
    """Holm-Bonferroni adjustment for a family of tests; None entries pass through."""
    indexed = [(p, i) for i, p in enumerate(p_values) if p is not None]
    indexed.sort()
    m = len(indexed)
    adjusted: list[float | None] = [None] * len(p_values)
    running = 0.0
    for rank, (p, i) in enumerate(indexed):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[i] = running
    return adjusted
