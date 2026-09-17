# Movie Recommender Data Sources

This reference summarizes the main data sources for the CSCE 585 controllable multi-embedding movie recommender and the role each can play in the experiment.

| Source | What you get | Best use in the project |
|---|---|---|
| **MovieLens 32M** | User ratings, timestamps, movie IDs, genres, and tag data | Main **evaluation dataset**. Create train/validation/test splits, measure NDCG and Recall, and compare the multi-embedding recommender with genre-based and single-embedding baselines. |
| **TMDB API** | Movie titles, overviews, release dates, posters, genres, keywords, cast/crew, and TMDB IDs | Primary **metadata-enrichment source** for building the five facet texts: plot, themes, characters, tone, and narrative structure. |
| **MovieLens Tag Genome** | Dense relevance scores for many descriptive tags | Strong source for the **themes** and **tone** facets. It provides more specific semantic signals than broad genre labels alone. |
| **IMDb datasets** | Titles, ratings, vote counts, cast/crew, and title links | Useful as a backup or enrichment source for credits and popularity/quality features. Document its non-commercial licensing constraints. |
| **Wikidata / Wikipedia** | Structured entities and occasional richer plot/context information | Helpful for resolving IDs, franchises, countries, languages, and metadata gaps. Coverage and formatting vary, so it should not be the only source. |
| **CMU Movie Summary Corpus** | Plot summaries with metadata | Optional research dataset for testing richer plot-based embeddings or validating the plot-facet pipeline. It is smaller and less current than TMDB. |

## Recommended starting combination

Start with **MovieLens 32M** and **TMDB**:

- **MovieLens** tells you what users liked, which lets you evaluate recommendation quality.
- **TMDB** tells you what each movie is about, which lets you create the embeddings and controllable facets.

Add **MovieLens Tag Genome** next, if available with your MovieLens download, to strengthen the themes and tone facets.

## Suggested use by facet

| Facet | Primary inputs |
|---|---|
| Plot | TMDB overview or synopsis |
| Themes | TMDB keywords and MovieLens Tag Genome |
| Characters | TMDB cast/crew plus extracted role and relationship descriptions |
| Tone | MovieLens tags plus controlled labels derived from metadata |
| Narrative structure | Structured labels extracted from plot and keyword information |

## Important integration task

Build a mapping table between MovieLens `movieId`, TMDB `tmdbId`, and IMDb `imdbId`. This lets you join user ratings to movie metadata correctly. Before generating embeddings, filter to movies that have both sufficient MovieLens interaction data and usable metadata.
