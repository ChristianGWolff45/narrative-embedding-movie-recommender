"""Embed the movie catalog with one sentence-embedding model."""

from __future__ import annotations

import json
import sys

import numpy as np

from config import CATALOG_PATH, EMBEDDINGS_PATH, MAX_SEQ_LENGTH, MODEL_INFO_PATH, MODEL_NAME
from movie_document import document_from_fields


def _configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


def load_records() -> list[dict]:
    if not CATALOG_PATH.exists():
        raise SystemExit(f"Missing {CATALOG_PATH}. Run src/fetch_tmdb.py first.")
    records = []
    with CATALOG_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise SystemExit(f"{CATALOG_PATH} is empty.")
    return records


def load_tokenizer():
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(hf_hub_download(MODEL_NAME, "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return tokenizer


def _token_length(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text).ids)


def fit_document(tokenizer, max_length: int, record: dict) -> str:
    """Shorten overview, then keywords, so director and cast stay inside the window."""

    def build(overview: str, keywords: str) -> str:
        return document_from_fields(
            title=record.get("title") or "",
            overview=overview,
            genres=record.get("genres") or "",
            keywords=keywords,
            director=record.get("director") or "",
            top_cast=record.get("top_cast") or "",
        )

    overview = record.get("overview") or ""
    keywords = record.get("keywords") or ""
    document = build(overview, keywords)
    if _token_length(tokenizer, document) <= max_length:
        return document

    words = overview.split()
    best_words = 0
    low, high = 0, len(words)
    while low <= high:
        mid = (low + high) // 2
        if _token_length(tokenizer, build(" ".join(words[:mid]), keywords)) <= max_length:
            best_words = mid
            low = mid + 1
        else:
            high = mid - 1
    if best_words > 0 or _token_length(tokenizer, build("", keywords)) <= max_length:
        return build(" ".join(words[:best_words]), keywords)

    keyword_items = [item.strip() for item in keywords.split(",") if item.strip()]
    best_keywords = 0
    low, high = 0, len(keyword_items)
    while low <= high:
        mid = (low + high) // 2
        candidate = ", ".join(keyword_items[:mid])
        if _token_length(tokenizer, build("", candidate)) <= max_length:
            best_keywords = mid
            low = mid + 1
        else:
            high = mid - 1
    return build("", ", ".join(keyword_items[:best_keywords]))


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    from fastembed import TextEmbedding

    model = TextEmbedding(MODEL_NAME)
    vectors = [np.asarray(vector, dtype=np.float32) for vector in model.embed(texts, batch_size=batch_size)]
    embeddings = np.vstack(vectors)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def embed_catalog(batch_size: int = 64) -> None:
    _configure_stdout()
    records = load_records()
    print(f"Loading {MODEL_NAME}", flush=True)
    tokenizer = load_tokenizer()
    documents = [fit_document(tokenizer, MAX_SEQ_LENGTH, record) for record in records]
    truncated = sum(
        1
        for record, document in zip(records, documents)
        if document != (record.get("document") or "")
    )
    print(f"Encoding {len(documents):,} documents ({truncated:,} shortened to fit {MAX_SEQ_LENGTH} tokens)", flush=True)
    embeddings = embed_texts(documents, batch_size=batch_size)
    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    MODEL_INFO_PATH.write_text(
        json.dumps(
            {
                "model_name": MODEL_NAME,
                "num_movies": int(embeddings.shape[0]),
                "embedding_dim": int(embeddings.shape[1]),
                "max_seq_length": MAX_SEQ_LENGTH,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {embeddings.shape[0]:,} x {embeddings.shape[1]} embeddings to {EMBEDDINGS_PATH}", flush=True)


def main() -> None:
    embed_catalog()


if __name__ == "__main__":
    main()
