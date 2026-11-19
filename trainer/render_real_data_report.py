"""Render the checked-in real-data evaluation report and SVG charts."""

from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def read_json(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def svg_text(x: float, y: float, value: str, *, size: int = 14, fill: str = "#263238",
             weight: str = "400", anchor: str = "start") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{html.escape(value)}</text>')


def write_outcomes_chart(evaluation: dict) -> None:
    comparisons = evaluation["comparisons"]
    baselines = [
        ("single", "Single-vector"),
        ("genre", "Genre overlap"),
        ("popular", "Popularity"),
        ("random_matched", "Matched random"),
    ]
    panels = [
        ("co_like_rate", "Co-like rate", "Difference in the share of query fans who also liked a recommendation."),
        ("genre_diversity", "Genre diversity", "Difference in mean pairwise genre distance within each list."),
    ]
    width, height = 1160, 510
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Narrative recommendations compared with four baselines</title>',
        '<desc id="desc">Paired mean differences with 95 percent bootstrap confidence intervals. Positive values favor narrative recommendations.</desc>',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        '<style>text{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}</style>',
        svg_text(42, 42, "Recommendation outcomes", size=25, weight="650"),
        svg_text(42, 69, "Equal narrative weights · k = 10 · positive values favor narrative recommendations", size=14, fill="#627078"),
    ]
    panel_xs = [42, 604]
    panel_w = 514
    for panel_x, (metric, heading, description) in zip(panel_xs, panels):
        data = [(key, label, comparisons[key][metric]) for key, label in baselines]
        values = [num for _, _, row in data for num in (row["ci_low"], row["ci_high"])]
        low = min(values)
        high = max(values)
        pad = (high - low) * 0.12
        low -= pad
        high += pad
        # Keep the origin and at least a little scale on each side visible.
        low = min(low, -0.01)
        high = max(high, 0.01)
        plot_left = panel_x + 154
        plot_right = panel_x + panel_w - 18
        plot_width = plot_right - plot_left
        y_top = 168
        y_bottom = 405
        x_pos = lambda value: plot_left + (value - low) / (high - low) * plot_width
        parts.extend([
            svg_text(panel_x, 113, heading, size=18, weight="650"),
            svg_text(panel_x, 136, description, size=11, fill="#627078"),
        ])
        ticks = [low, 0.0, high]
        if high - low > 0.45:
            ticks = [low, 0.0, high / 2, high]
        for tick in ticks:
            x = x_pos(tick)
            color = "#6e7b81" if abs(tick) < 1e-12 else "#e2e4e1"
            stroke = "#59656a" if abs(tick) < 1e-12 else "#dfe3e0"
            parts.append(f'<line x1="{x:.1f}" y1="{y_top-8}" x2="{x:.1f}" y2="{y_bottom+4}" stroke="{stroke}" stroke-width="1"/>')
            parts.append(svg_text(x, y_bottom + 26, f"{tick * 100:+.0f} pp", size=11, fill=color, anchor="middle"))
        for index, (_, label, row) in enumerate(data):
            y = y_top + index * 59
            x1, x2, point = x_pos(row["ci_low"]), x_pos(row["ci_high"]), x_pos(row["mean_difference"])
            color = "#236b69" if row["mean_difference"] >= 0 else "#b4563e"
            parts.append(svg_text(panel_x, y + 4, label, size=13, weight="550"))
            parts.append(svg_text(panel_x, y + 22, f"n = {row['n']}", size=10, fill="#768188"))
            parts.append(f'<line x1="{x1:.1f}" y1="{y}" x2="{x2:.1f}" y2="{y}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>')
            parts.append(f'<line x1="{x1:.1f}" y1="{y-6}" x2="{x1:.1f}" y2="{y+6}" stroke="{color}" stroke-width="1.5"/>')
            parts.append(f'<line x1="{x2:.1f}" y1="{y-6}" x2="{x2:.1f}" y2="{y+6}" stroke="{color}" stroke-width="1.5"/>')
            parts.append(f'<circle cx="{point:.1f}" cy="{y}" r="5" fill="{color}" stroke="#fbfaf7" stroke-width="1.5"/>')
        parts.append(svg_text(panel_x, 475, "95% percentile bootstrap interval", size=11, fill="#627078"))
    parts.append(svg_text(42, 503, "Holm-adjusted permutation p-values and unadjusted confidence intervals are reported in the accompanying table.", size=11, fill="#627078"))
    parts.append("</svg>")
    (REPORTS / "paired_outcomes.svg").write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_latency_chart(benchmark: dict, loadtest: dict) -> None:
    latency = benchmark["latency_k10"]
    http = {}
    for row in loadtest["results"]:
        if row["offered_rps"] == 100:
            http[row["mode"]] = row["latency"]["p95_ms"]
    entries = [
        ("Single vector · in process", latency["single"]["p95_ms"], "#365f73"),
        ("Narrative, equal weights · in process", latency["multi_equal_weights"]["p95_ms"], "#236b69"),
        ("Narrative, random weights · in process", latency["multi_random_weights"]["p95_ms"], "#42817d"),
        ("Single vector · HTTP at 100 req/s", http["single"], "#365f73"),
        ("Narrative · HTTP at 100 req/s", http["multi"], "#236b69"),
    ]
    width, height = 1020, 440
    left, plot_width, max_ms = 365, 590, max(12, max(value for _, value, _ in entries) * 1.08)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Recommendation latency on COMECH-2080</title>',
        '<desc id="desc">Serial in-process p95 latency and HTTP p95 latency at 100 requests per second.</desc>',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        '<style>text{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}</style>',
        svg_text(40, 42, "Latency at the 6,194-film catalog", size=24, weight="650"),
        svg_text(40, 68, "p95 response time · lower is better · in-process and HTTP paths use separate measurements", size=13, fill="#627078"),
    ]
    for tick in range(0, 13, 2):
        x = left + tick / max_ms * plot_width
        parts.append(f'<line x1="{x:.1f}" y1="100" x2="{x:.1f}" y2="374" stroke="#e2e4e1" stroke-width="1"/>')
        parts.append(svg_text(x, 398, f"{tick} ms", size=11, fill="#627078", anchor="middle"))
    for index, (label, value, color) in enumerate(entries):
        y = 125 + index * 51
        bar_w = max(1.5, value / max_ms * plot_width)
        parts.append(svg_text(40, y + 17, label, size=13, weight="500"))
        parts.append(f'<rect x="{left}" y="{y}" width="{bar_w:.2f}" height="22" rx="4" fill="{color}"/>')
        parts.append(svg_text(left + bar_w + 8, y + 17, f"{value:.2f} ms", size=12, fill="#37474f", weight="600"))
        if index == 2:
            parts.append(f'<line x1="40" y1="{y+39}" x2="970" y2="{y+39}" stroke="#d9ddda"/>')
    parts.append(svg_text(40, 430, "HTTP run: 5 seconds at each offered rate; 100 req/s was the highest rate tested, with zero errors.", size=11, fill="#627078"))
    parts.append("</svg>")
    (REPORTS / "latency.svg").write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_capacity_chart(benchmark: dict) -> None:
    memory = benchmark["memory"]
    storage = benchmark["storage"]
    actual_mb = memory["total_vectors_mb"]
    projected_mb = memory["projected_total_mb_at_tmdb_scale"]
    budget_mb = benchmark["budget"]["memory_mb_limit"]
    width, height = 1020, 390
    left, plot_width, max_mb = 320, 570, 820
    budget_x = left + budget_mb / max_mb * plot_width
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Vector memory and artifact storage</title>',
        '<desc id="desc">Memory for the real catalog and an 87,585-title synthetic projection compared with a 256-megabyte vector budget.</desc>',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        '<style>text{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}</style>',
        svg_text(40, 42, "Vector memory and on-disk size", size=24, weight="650"),
        svg_text(40, 68, "The 87,585-title point is a synthetic projection, not a measured real catalog.", size=13, fill="#627078"),
    ]
    for tick in [0, 200, 400, 600, 800]:
        x = left + tick / max_mb * plot_width
        parts.append(f'<line x1="{x:.1f}" y1="116" x2="{x:.1f}" y2="245" stroke="#e2e4e1"/>')
        parts.append(svg_text(x, 267, f"{tick} MiB", size=11, fill="#627078", anchor="middle"))
    parts.append(f'<line x1="{budget_x:.1f}" y1="105" x2="{budget_x:.1f}" y2="246" stroke="#b4563e" stroke-width="2" stroke-dasharray="5 5"/>')
    parts.append(svg_text(budget_x + 7, 103, f"{budget_mb:.0f} MiB vector budget", size=11, fill="#9b4935"))
    bars = [("Real catalog · 6,194 films", actual_mb, "#236b69"), ("Synthetic scale · 87,585 films", projected_mb, "#b77742")]
    for i, (label, value, color) in enumerate(bars):
        y = 145 + i * 64
        bar_w = value / max_mb * plot_width
        parts.append(svg_text(40, y + 18, label, size=13, weight="500"))
        parts.append(f'<rect x="{left}" y="{y}" width="{bar_w:.2f}" height="25" rx="4" fill="{color}"/>')
        parts.append(svg_text(left + bar_w + 8, y + 18, f"{value:.1f} MiB", size=12, weight="600"))
    parts.append('<line x1="40" y1="294" x2="980" y2="294" stroke="#d9ddda"/>')
    parts.append(svg_text(40, 324, f"Compressed artifact: {storage['artifact_bytes'] / (1024**2):.1f} MiB ({storage['artifact_bytes'] / 1_000_000:.1f} MB)", size=13, weight="600"))
    parts.append(svg_text(40, 348, f"Single-vector member: {storage['single_bytes'] / (1024**2):.1f} MiB · narrative members: {storage['narrative_bytes'] / (1024**2):.1f} MiB · {storage['narrative_to_single_ratio']:.2f}× single-vector size", size=12, fill="#53636a"))
    parts.append(svg_text(40, 374, f"Benchmark process peak RSS: {memory['peak_process_rss_mb']:.0f} MiB, including Python/runtime overhead; the plotted budget applies to vectors only.", size=11, fill="#627078"))
    parts.append("</svg>")
    (REPORTS / "capacity.svg").write_text("\n".join(parts) + "\n", encoding="utf-8")


def render_markdown(evaluation: dict, benchmark: dict, loadtest: dict, build: dict, samples: dict) -> str:
    comparisons = evaluation["comparisons"]
    method_names = {
        "single": "Single-vector",
        "genre": "Genre overlap",
        "popular": "Popularity",
        "random_matched": "Rating-count-matched random",
    }
    lines = [
        "# Real-data evaluation: narrative movie recommender",
        "",
        "**Run date:** 2026-11-19  ·  **Source:** `main` at `40783b1`  ·  **Machine:** COMECH-2080",
        "",
        "## Result",
        "",
        "The real 6,194-film catalog is well inside the current serving budgets: the measured in-process p95 is below 1 ms, the HTTP path stayed below 13 ms through the highest tested rate of 100 requests/second, and the six resident vector arrays use 54.4 MiB against a 256 MiB vector budget. The multi-vector artifact adds useful genre diversity and beats rating-count-matched random recommendations, but this run does not establish a quality improvement over the single-vector or genre-only baselines. Popularity remains much stronger on the co-like metric.",
        "",
        "The broader 87,585-title capacity point is synthetic. It projects 769.8 MiB of resident vectors, so the 256 MiB memory target would require a storage/loading strategy change at that scale. Its measured array-search latency is still low, but it is not a real full-catalog validation.",
        "",
        "## Data and run",
        "",
        f"- Catalog: **{evaluation['movies']:,} films** with **{evaluation['movies']:,} labels** already available; the evaluator loaded **29,782,953 MovieLens ratings**.",
        f"- Quality evaluation: {evaluation['queries_evaluated']} query films, equal narrative weights; swept `k` values {', '.join(map(str, evaluation['k_values']))} and 10 random weight profiles (47 profile/`k` cells total).",
        f"- A co-like outcome means the share of people who rated the seed at least {evaluation['like_threshold']:.0f}/5 and also rated the recommendation at least {evaluation['like_threshold']:.0f}/5. The evaluator requires at least {evaluation['min_overlap']} shared ratings.",
        "- Pairwise differences are narrative minus baseline. Intervals are 95% percentile bootstrap intervals; p-values are two-sided sign-flip permutation tests with Holm correction within the headline family. The number of valid seed pairs differs by metric and baseline.",
        "- The Qwen2.5-3B annotations pre-date this benchmark. This run did not fine-tune model weights: it reused the baseline MiniLM vectors and rebuilt the five narrative embedding dimensions from existing labels in {:.1f} seconds ({:.2f} ms per film).".format(build["narrative_encode_seconds"], build["narrative_ms_per_film"]),
        "",
        "## Quality: equal weights, `k=10`",
        "",
        "| Baseline | Paired `n` | Co-like difference (95% CI) | Holm adjusted `p` |",
        "|---|---:|---:|---:|",
    ]
    for key in ("single", "genre", "popular", "random_matched"):
        row = comparisons[key]["co_like_rate"]
        lines.append(f"| {method_names[key]} | {row['n']} | {row['mean_difference'] * 100:+.2f} pp ({row['ci_low'] * 100:+.2f}, {row['ci_high'] * 100:+.2f}) | {row['p_value_holm']:.4f} |")
    lines.extend([
        "",
        "The point estimate versus single-vector is −1.90 percentage points; its unadjusted interval is below zero, but the Holm-adjusted p-value is 0.0552, just above the 0.05 familywise threshold. Narrative versus genre overlap is inconclusive (`p=0.1615`). Narrative exceeds the rating-count-matched random baseline by 4.83 points (`p=0.0008`) and trails popularity by 25.05 points (`p=0.0008`). Read the adjusted p-values for significance decisions; the displayed intervals are not multiplicity-adjusted.",
        "",
        "For genre diversity, narrative minus single-vector is **−5.64 pp** (Holm `p=0.0008`), narrative minus genre overlap is **+53.88 pp** (`p=0.0008`), and narrative is less diverse than both popularity and matched random by about **25 pp**. This shows a trade-off: the narrative representation is much broader than a genre match, while single-vector recommendations are slightly more diverse in this sample.",
        "",
        "![Paired quality differences with bootstrap intervals](paired_outcomes.svg)",
        "",
        "## Serving cost and capacity",
        "",
        "| Measurement | Result |",
        "|---|---:|",
        f"| Serial single-vector p95, `k=10` | {benchmark['latency_k10']['single']['p95_ms']:.3f} ms |",
        f"| Serial narrative, equal weights p95, `k=10` | {benchmark['latency_k10']['multi_equal_weights']['p95_ms']:.3f} ms |",
        f"| Serial narrative, random weights p95, `k=10` | {benchmark['latency_k10']['multi_random_weights']['p95_ms']:.3f} ms |",
        f"| HTTP single-vector p95 at 100 offered req/s | {next(row['latency']['p95_ms'] for row in loadtest['results'] if row['mode'] == 'single' and row['offered_rps'] == 100):.2f} ms |",
        f"| HTTP narrative p95 at 100 offered req/s | {next(row['latency']['p95_ms'] for row in loadtest['results'] if row['mode'] == 'multi' and row['offered_rps'] == 100):.2f} ms |",
        f"| Six vector arrays in memory, actual catalog | {benchmark['memory']['total_vectors_mb']:.1f} MiB |",
        f"| Compressed `.npz` artifact | {benchmark['storage']['artifact_bytes'] / (1024**2):.1f} MiB ({benchmark['storage']['artifact_bytes'] / 1_000_000:.1f} MB) |",
        f"| Peak benchmark-process RSS | {benchmark['memory']['peak_process_rss_mb']:.0f} MiB, including runtime overhead |",
        "",
        "The HTTP load test offered 5, 20, 50, and 100 requests/second for five seconds each and had zero errors in both modes. **100 requests/second is the highest rate tested, not a measured saturation limit.** The server was reached over its own Tailscale address, so this includes HTTP and queueing but not WAN latency from another machine.",
        "",
        f"The vector budget passes on the real catalog. The benchmark estimates only **{benchmark['budget']['largest_catalog_within_memory']:,}** titles fit under 256 MiB when all six 384-dimensional float32 arrays are resident. The 87,585-title point and its {benchmark['memory']['projected_total_mb_at_tmdb_scale']:.1f} MiB vector requirement are generated from synthetic vectors; no semantic quality claim is made at that scale.",
        "",
        "![Observed serial and HTTP latency](latency.svg)",
        "",
        "![Vector memory, artifact size, and the resident-vector budget](capacity.svg)",
        "",
        "## Qualitative samples",
        "",
        f"[`output_samples.json`](output_samples.json) contains {samples['sample_count']} seed films, each with top-five single-vector and equal-weight narrative recommendations and the existing Qwen labels for the seeds and recommendations. It is intended for the teammate's output review; it contains no user-level rating rows.",
        "",
        "## Limits",
        "",
        "- Offline cosine-search timings are array-search microbenchmarks; they exclude model inference, index/network fetches, and cold process startup.",
        "- The HTTP run used short five-second rate windows on the same host as the server and stopped at 100 offered requests/second. It should not be read as an external-client or saturation test.",
        "- Only the 6,194-title catalog uses real movie records and labels. Larger scaling rows use synthetic vectors; the memory projection is useful for sizing but not a substitute for measuring a real expanded catalog.",
        "- The current relevance metric is a MovieLens co-like proxy, not a user study or a measure of narrative coherence. The qualitative samples are provided for human inspection.",
        "",
        "## Reproduction",
        "",
        "On COMECH-2080, with the project environment and the web app running in a second shell:",
        "",
        "```sh",
        "bin/web --host 127.0.0.1 --port 8765",
        "# In another shell:",
        "bin/reconstruct build --input data/processed/movie_lens_labels.jsonl --output artifacts/movie_embeddings.npz",
        "bin/evaluate --queries 200 --k 10 5 20 --random-profiles 10 --resamples 10000 --output reports/evaluation.json",
        "bin/reconstruct benchmark --artifact artifacts/movie_embeddings.npz --queries 500 --report reports/reconstruction_benchmark.json",
        "bin/loadtest --url http://127.0.0.1:8765 --artifact artifacts/movie_embeddings.npz --modes single multi --rates 5 20 50 100 --duration 5 --workers 128 --timeout 10 --report reports/loadtest.json",
        "python -m trainer.render_real_data_report",
        "```",
        "",
        "The generated report source is [`trainer/render_real_data_report.py`](../trainer/render_real_data_report.py).",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    evaluation = read_json("evaluation.json")
    benchmark = read_json("reconstruction_benchmark.json")
    loadtest = read_json("loadtest.json")
    build = read_json("movie_embeddings.build.json")
    samples = read_json("output_samples.json")
    write_outcomes_chart(evaluation)
    write_latency_chart(benchmark, loadtest)
    write_capacity_chart(benchmark)
    (REPORTS / "real_data_report.md").write_text(
        render_markdown(evaluation, benchmark, loadtest, build, samples), encoding="utf-8"
    )
    print(f"Rendered report and charts in {REPORTS}")


if __name__ == "__main__":
    main()
