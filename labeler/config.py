from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


SECRET_FILE = Path.home() / ".config" / "csce585-labeler" / ".env"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


@dataclass(frozen=True)
class Settings:
    tmdb_api_key: str
    tmdb_bearer_token: str | None
    model_id: str
    tmdb_language: str = "en-US"


def load_settings(model_id: str | None = None) -> Settings:
    load_dotenv(SECRET_FILE, override=False)
    api_key = os.environ.get("TMDB_API_KEY", "").strip()
    bearer = os.environ.get("TMDB_BEARER_TOKEN", "").strip() or None
    if not api_key and not bearer:
        raise RuntimeError(
            f"No TMDB credential found. Add TMDB_API_KEY to {SECRET_FILE} "
            "or set TMDB_BEARER_TOKEN in the process environment."
        )
    return Settings(
        tmdb_api_key=api_key,
        tmdb_bearer_token=bearer,
        model_id=model_id or os.environ.get("LABELER_MODEL_ID", DEFAULT_MODEL_ID),
        tmdb_language=os.environ.get("TMDB_LANGUAGE", "en-US"),
    )
