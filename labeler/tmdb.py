from __future__ import annotations

import time
from typing import Any

import httpx


class TMDBClient:
    base_url = "https://api.themoviedb.org/3"

    def __init__(
        self,
        api_key: str = "",
        bearer_token: str | None = None,
        language: str = "en-US",
        timeout: float = 45.0,
    ) -> None:
        self.api_key = api_key
        self.bearer_token = bearer_token
        self.language = language
        self.client = httpx.Client(timeout=timeout, headers={"Accept": "application/json"})

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "TMDBClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _get(self, path: str, **params: Any) -> dict[str, Any]:
        params.setdefault("language", self.language)
        headers: dict[str, str] = {}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        else:
            params["api_key"] = self.api_key

        for attempt in range(4):
            try:
                response = self.client.get(f"{self.base_url}{path}", params=params, headers=headers)
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                if attempt == 3:
                    raise RuntimeError("TMDB request timed out or could not connect after retries") from exc
                time.sleep(2**attempt)
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 3:
                    raise RuntimeError(
                        f"TMDB request failed with HTTP {response.status_code}: "
                        f"{response.text[:200]}"
                    )
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
                time.sleep(min(delay, 30.0))
                continue
            if response.status_code >= 400:
                raise RuntimeError(
                    f"TMDB request failed with HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )
            return response.json()
        raise RuntimeError("TMDB request failed after retries")

    def discover_page(self, page: int) -> dict[str, Any]:
        return self._get(
            "/discover/movie",
            page=page,
            sort_by="popularity.desc",
            include_adult="false",
            include_video="false",
        )

    def movie_bundle(self, movie_id: int) -> dict[str, Any]:
        raw = self._get(f"/movie/{movie_id}", append_to_response="credits")
        return {
            "tmdb_id": raw.get("id"),
            "title": raw.get("title") or raw.get("original_title") or "",
            "overview": raw.get("overview") or "",
            "release_date": raw.get("release_date") or "",
            "genres": [genre.get("name", "") for genre in raw.get("genres", [])],
            "vote_average": raw.get("vote_average"),
            "cast": [
                {"name": person.get("name", ""), "character": person.get("character", "")}
                for person in raw.get("credits", {}).get("cast", [])[:12]
            ],
        }
