import argparse

import pytest

from trainer import evaluate as ev


def _args(synthetic, **overrides):
    values = dict(
        artifact=str(synthetic["artifact"]), queries=40, k=[10, 5], seed=0, like_threshold=4.0,
        min_overlap=5, plot=1.0, themes=1.0, characters=1.0, tone=1.0, structure=1.0,
        random_profiles=3, no_sweep=False, resamples=500,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.fixture
def report(synthetic, monkeypatch):
    monkeypatch.setattr(ev, "RATINGS_PATH", synthetic["ratings"])
    return ev.evaluate(_args(synthetic))


def test_report_keeps_fields_the_web_app_reads(report):
    for name in ("single", "multi"):
        assert report["methods"][name]["co_like_rate_paired"]["mean"] is not None
    assert report["k"] == 10 and report["k_values"] == [10, 5]


def test_sweep_covers_every_profile_and_k(report):
    # 1 headline + 5 single-dimension + 3 random = 9 profiles, 2 k values, minus the headline cell.
    assert len(report["sweep"]) == 9 * 2 - 1
    assert {row["k"] for row in report["sweep"]} == {5, 10}


def test_comparisons_are_paired_and_holm_adjusted(report):
    for block in report["comparisons"].values():
        for cell in block.values():
            assert cell["n"] > 0
            assert cell["ci_low"] <= cell["mean_difference"] <= cell["ci_high"]
            assert cell["p_value"] <= cell["p_value_holm"] <= 1.0


def test_narrative_beats_noisy_single_on_planted_signal(report):
    # The fixture plants the taste signal mostly in the narrative vectors.
    cell = report["comparisons"]["single"]["co_like_rate"]
    assert cell["mean_difference"] > 0
    assert cell["p_value_holm"] < 0.05


def test_no_sweep_evaluates_headline_only(synthetic, monkeypatch):
    monkeypatch.setattr(ev, "RATINGS_PATH", synthetic["ratings"])
    result = ev.evaluate(_args(synthetic, no_sweep=True, k=[10]))
    assert result["sweep"] == []


def test_rejects_all_zero_weights(synthetic):
    with pytest.raises(SystemExit):
        ev.weight_profiles(_args(synthetic, plot=0, themes=0, characters=0, tone=0, structure=0))


def test_summary_prints_headline_and_sweep(report, capsys):
    ev._print_summary(report)
    out = capsys.readouterr().out
    assert "narrative vs single" in out and "Sweep:" in out
