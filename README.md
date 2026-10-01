# Single-embedding movie recommender

Each movie is one vector. The text that gets embedded is:

```
Title: {title}
Overview: {overview}
Genres: {genres}
Keywords: {keywords}
Director: {director}
Top cast: {top_5_cast}
```

Recommendations are the nearest movies in that vector space. The encoder is `sentence-transformers/all-MiniLM-L6-v2`, run with fastembed.

## Setup

Python 3.11 or newer. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Git Bash, activate with `source .venv/Scripts/activate`.

MovieLens 32M must be unpacked at `data/raw/ml-32m/` so these files exist:

- `data/raw/ml-32m/links.csv`
- `data/raw/ml-32m/ratings.csv`

Download it from [GroupLens](https://grouplens.org/datasets/movielens/32m/). The ratings file has to be named `ratings.csv`.

Create a `.env` file in the repository root with a TMDB read token:

```
TMDB_READ_TOKEN=your_read_access_token
```

Create the token at [TMDB settings](https://www.themoviedb.org/settings/api).

## Build the model

This fetches TMDB metadata for MovieLens titles, writes `data/processed/movie_catalog.jsonl`, and saves the embedding matrix to `data/processed/embeddings.npy`. Movies with the most ratings are fetched first. A rerun skips titles already in the catalog.

The index used for development is the 3,000 most-rated movies that have at least 500 MovieLens ratings:

```powershell
python src/build_single_embedding.py --min-ratings 500 --limit 3000
```

Leave `--limit` off to embed every title that meets `--min-ratings` (the default minimum is 500). `--skip-fetch` rebuilds vectors from the catalog already on disk.

## Test the model

Similar movies for a title already in the catalog:

```powershell
python src/recommend.py "Fight Club"
```

A free-text description. This encodes the query with the same model:

```powershell
python src/recommend.py --text "an insomniac and a soap salesman start an underground fight club"
```

Combine a few liked titles:

```powershell
python src/recommend.py --like "Toy Story" --like "Finding Nemo"
```

`-k` sets how many movies to return. The default is 10.

## Test latency and throughput

`src/bench_recommend.py` loads the index once, then times steady-state queries. Title search is a nearest-neighbor lookup. Text search also encodes the description. Encoder load time is reported separately and is not included in the per-query latency.

```powershell
python src/bench_recommend.py
```

Defaults are 200 title queries and 40 text queries, with 10 warmup calls dropped before timing. Override them with `--title-requests`, `--text-requests`, and `--warmup`.
