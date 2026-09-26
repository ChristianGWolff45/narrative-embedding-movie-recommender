from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "ml-32m"
PROCESSED_DIR = ROOT / "data" / "processed"

LINKS_PATH = RAW_DIR / "links.csv"
RATINGS_PATH = RAW_DIR / "ratings.csv"
RATING_COUNTS_PATH = PROCESSED_DIR / "rating_counts.csv"
CATALOG_PATH = PROCESSED_DIR / "movie_catalog.jsonl"
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings.npy"
MODEL_INFO_PATH = PROCESSED_DIR / "model.json"

# One encoder for the full movie document. Movie-to-movie similarity uses the
# same vector space on both sides, so queries are encoded without a task prefix.
# fastembed runs the ONNX weights, which avoids a PyTorch install.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 256
