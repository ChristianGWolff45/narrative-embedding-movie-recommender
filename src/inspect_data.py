from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw/ml-32m")

movies = pd.read_csv(DATA_DIR / "movies.csv")
links = pd.read_csv(DATA_DIR / "links.csv")
ratings = pd.read_csv(DATA_DIR / "ratings.csv")
tags = pd.read_csv(DATA_DIR / "tags.csv")

print("Movies:")
print(movies.head())
print()

print("Links:")
print(links.head())
print()

print("Ratings:")
print(ratings.head())
print()

print("Tags:")
print(tags.head())
print()

print(f"Number of movies: {len(movies):,}")
print(f"Number of ratings: {len(ratings):,}")
print(f"Number of tags: {len(tags):,}")