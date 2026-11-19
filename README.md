# Reel Notes — Narrative Movie Recommender

Reel Notes finds films by comparing the story details viewers care about. It pairs a conventional whole-film embedding with five separate narrative dimensions: plot, themes, characters, tone, and narrative structure. The local web app supports film search, reference lists, an adjustable story mix, and an offline evaluation report.

## Current collection

The MovieLens 32M data selects every title with at least 500 ratings and a TMDB match. On the COMECH-2080 build, this produced **6,223 candidates**; TMDB returned usable metadata for **6,194**. The remaining 29 were unavailable or filtered by TMDB. MovieLens ratings and TMDB metadata are local inputs; generated labels, vectors, credentials, and reports are excluded from Git.

## Run the website

Use Python 3.11 or 3.12. From the project folder:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
bin/web
```

Then open `http://127.0.0.1:8765`. On COMECH-2080, run as `jvaught2` from `~/marshall/csce585-movie-recommender` and bind to its Tailscale address to keep the preview on the private tailnet:

```bash
bin/web --host 100.64.182.86 --port 8765
```

The app starts with the single-film profile. Once the full narrative artifact and evaluation report exist, it enables the story-mix controls automatically. Genre and year filters apply to both profiles.

## Build or refresh the collection

Place MovieLens 32M `ratings.csv` and `links.csv` under `data/raw/ml-32m/`. TMDB credentials belong in `~/.config/csce585-labeler/.env`, with mode `600`; use `TMDB_BEARER_TOKEN` or `TMDB_API_KEY`.

Fetch every eligible MovieLens title and build the baseline vectors:

```bash
bin/train-embeddings --min-ratings 500
bin/backfill-artwork
```

`bin/train-embeddings` resumes catalog fetching and encodes the full fetched catalog with frozen `sentence-transformers/all-MiniLM-L6-v2` vectors. The artwork step fills TMDB poster and backdrop paths for older cached rows.

To run the complete, resumable data, labeling, model, and evaluation sequence on the GPU server:

```bash
MOVIE_LABEL_CUDA=0 bin/finish-project
```

This command reuses saved TMDB labels, labels only uncovered titles, then rebuilds the vectors, comparison report, and retrieval benchmark. It keeps a copy of the previous 8,298-film artifact before replacing the default artifact with the MovieLens-aligned collection.

## Build the narrative model

The labeling command resumes from its JSONL output. Existing TMDB labels can first be matched to MovieLens IDs with:

```bash
bin/seed-movielens-labels
```

On the GPU server, use one available GPU for local Qwen labeling:

```bash
CUDA_VISIBLE_DEVICES=0 bin/label-catalog \
  --backend transformers \
  --model Qwen/Qwen2.5-3B-Instruct \
  --input data/processed/movie_catalog.jsonl \
  --output data/processed/movie_lens_labels.jsonl
```

Then build narrative vectors and run the research comparison:

```bash
bin/reconstruct build \
  --input data/processed/movie_lens_labels.jsonl \
  --output artifacts/movie_embeddings.npz
bin/evaluate --queries 200 --k 10 5 20
bin/reconstruct benchmark --artifact artifacts/movie_embeddings.npz
```

## Evaluation

The evaluator compares single and weighted narrative retrieval against genre, popularity, and rating-count-matched random baselines. Relevance is an offline co-like proxy: the share of users who rated a query film at least 4/5 who also rated a recommendation at least 4/5. Diversity is pairwise Jaccard distance over genre sets. These measures do not replace a viewer study, and no viewer study was run.

**Relevance and diversity** (`bin/evaluate`, written to `reports/evaluation.json`):

- *Headline test:* equal narrative weights at the first `--k` (10). The narrative profile is compared against every baseline on the same query films. Each comparison reports the mean paired difference, a 95% bootstrap confidence interval, a sign-flip permutation p-value, win/tie/loss counts, and a Holm-adjusted p-value across the headline family.
- *Exploratory sweep:* every `--k` value crossed with the weight profiles: equal, each dimension alone, and `--random-profiles` random weightings (default 10). Each cell is compared against the single and genre baselines, with Holm adjustment across the whole sweep. Use `--no-sweep` for the headline only.

**Systems cost** (`bin/reconstruct benchmark`, written to `reports/reconstruction_benchmark.json`):

- Per-request latency (mean, p50, p95, p99) for single, equal-weight narrative, and random slider weightings. Each request is timed separately, the way the web server answers it.
- p95 latency for k = 5, 10, 20 and 50.
- Catalog scaling up to 87,585 films (the TMDB catalog size from Milestone 1). Sizes above the real catalog use random unit vectors and are labeled `synthetic`.
- Vector memory, peak process memory, and compressed storage of the narrative arrays relative to the single array.
- One-time costs: LLM labeling speed from `runs/*/status.json`, and per-dimension encoding time, which `bin/reconstruct build` records in `artifacts/movie_embeddings.build.json`.
- **Budget check:** pass/fail against p95 ≤ 100 ms for a narrative request with random weights and ≤ 256 MB of vectors in RAM. Change these with `--latency-budget-ms` and `--memory-budget-mb`.

**Load test** (`bin/loadtest`, written to `reports/loadtest.json`): start `bin/web`, then run `bin/loadtest`. It sends open-loop `/api/recommend` requests at 5–400 requests per second in both modes. Latency includes HTTP and queueing time. The report gives the highest rate that stays within the latency budget.

**Tests:** `uv pip install --python .venv/bin/python -e '.[dev]'`, then `.venv/bin/python -m pytest`. The tests build a synthetic catalog, ratings and artifact, so no real data is needed.

## Design and data credits

The interface uses an editorial type hierarchy, responsive layouts, keyboard focus styling, visible labels, and reduced-motion support. It follows WCAG 2.2 guidance for contrast, keyboard focus, and target sizing, with the Material 3 typography guidance for readable body text and clear type roles.

Film descriptions and images are provided by TMDB. This product uses the TMDB API but is not endorsed or certified by TMDB. TMDB requires its logo and attribution to appear in an About or Credits section; see its [API attribution guidance](https://developer.themoviedb.org/docs/faq).

The unmodified TMDB wordmark in the Credits section is the [approved blue short logo](https://www.themoviedb.org/assets/v4/logos/v2/blue_short-8e7b30f73a4020692ccca9c88bafe5dcb6f8a62a4c6bc55cd9ba82bb2cd95f6c.svg).

- [WCAG 2.2](https://www.w3.org/TR/WCAG22/)
- [Material Design 3 typography](https://m3.material.io/styles/typography/applying-type)
- [TMDB image URL guidance](https://developer.themoviedb.org/docs/image-basics)

## Source layout

- `src/` — MovieLens/TMDB catalog ingestion and the single-embedding baseline.
- `labeler/` — resumable local LLM labeling and progress tracking.
- `trainer/` — narrative embedding construction, recommendation evaluation, and benchmarks.
- `webapp/` — the local web server and responsive Reel Notes interface.
- `bin/` — project commands for fetching, labeling, building, evaluating, and serving.
