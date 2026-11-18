# Embedding reconstruction and evaluation

`reconstruct.py` reuses the GitHub baseline's single-embedding matrix, aligning rows by MovieLens ID, then creates five additional vectors from plot, themes, characters, tone, and narrative-structure text with the same frozen `all-MiniLM-L6-v2` FastEmbed model. It does not fine-tune the transformer.

The generated NPZ stores normalized vectors; the adjacent metadata JSON preserves MovieLens IDs, rating counts, titles, TMDB IDs, and genres for reproducible evaluation.

`evaluate.py` compares single and weighted multi retrieval with Jaccard genre, popularity, and rating-count-matched random baselines. It reports MovieLens co-like relevance, scored recommendation coverage, and genre-based intra-list diversity.
