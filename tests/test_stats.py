import numpy as np
import pytest

from trainer.stats import holm_adjust, latency_summary, paired_comparison


def test_clear_improvement_is_significant():
    rng = np.random.default_rng(0)
    control = rng.normal(0.3, 0.05, size=200)
    result = paired_comparison(control + 0.05, control, resamples=2000)
    assert result["mean_difference"] == pytest.approx(0.05)
    assert result["ci_low"] > 0
    assert result["p_value"] < 0.01
    assert result["wins"] == 200


def test_no_difference_is_not_significant():
    rng = np.random.default_rng(1)
    a = rng.normal(size=200)
    b = a + rng.normal(scale=0.5, size=200) * rng.choice([-1, 1], size=200)
    result = paired_comparison(a, b, resamples=2000)
    assert result["ci_low"] < 0 < result["ci_high"] or result["p_value"] > 0.05


def test_permutation_p_matches_scipy_wilcoxon_order_of_magnitude():
    scipy = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(2)
    b = rng.normal(size=80)
    a = b + rng.normal(0.15, 0.5, size=80)
    ours = paired_comparison(a, b, resamples=20000)["p_value"]
    reference = scipy.ttest_rel(a, b).pvalue
    assert abs(ours - reference) < 0.02


def test_empty_and_mismatched_inputs():
    assert paired_comparison([], [])["n"] == 0
    with pytest.raises(ValueError):
        paired_comparison([1, 2], [1])


def test_holm_adjust():
    assert holm_adjust([0.01, 0.04, None, 0.03]) == pytest.approx([0.03, 0.06, None, 0.06])


def test_latency_summary_units():
    summary = latency_summary([0.001] * 10)
    assert summary["p95_ms"] == pytest.approx(1.0)
    assert summary["serial_qps"] == pytest.approx(1000.0)
