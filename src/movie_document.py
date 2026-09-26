"""Turn a TMDB movie payload into the single-embedding document."""

from __future__ import annotations


def document_from_fields(
    *,
    title: str,
    overview: str,
    genres: str,
    keywords: str,
    director: str,
    top_cast: str,
) -> str:
    return (
        f"Title: {title}\n"
        f"Overview: {overview}\n"
        f"Genres: {genres}\n"
        f"Keywords: {keywords}\n"
        f"Director: {director}\n"
        f"Top cast: {top_cast}"
    )


def _clean(value) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _join_names(items, *, limit: int | None = None) -> str:
    names: list[str] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        name = _clean(item.get("name"))
        if not name:
            continue
        names.append(name)
        if limit is not None and len(names) >= limit:
            break
    return ", ".join(names)


def _top_cast(cast, limit: int = 5) -> str:
    people = [person for person in cast or [] if isinstance(person, dict)]
    people.sort(key=lambda person: person.get("order") if isinstance(person.get("order"), int) else 10**9)
    return _join_names(people, limit=limit)


def _directors(crew) -> str:
    people = [person for person in crew or [] if isinstance(person, dict)]
    directors = [person for person in people if person.get("job") == "Director"]
    if not directors:
        directors = [person for person in people if person.get("job") == "Co-Director"]
    return _join_names(directors)


def _keywords(movie: dict) -> str:
    raw = movie.get("keywords") or []
    if isinstance(raw, dict):
        raw = raw.get("keywords") or raw.get("results") or []
    return _join_names(raw)


def fields_from_tmdb(movie: dict) -> dict[str, str]:
    credits = movie.get("credits") or {}
    fields = {
        "title": _clean(movie.get("title")) or _clean(movie.get("original_title")),
        "overview": _clean(movie.get("overview")),
        "genres": _join_names(movie.get("genres")),
        "keywords": _keywords(movie),
        "director": _directors(credits.get("crew")),
        "top_cast": _top_cast(credits.get("cast")),
        "release_date": _clean(movie.get("release_date")),
    }
    fields["document"] = document_from_fields(
        title=fields["title"],
        overview=fields["overview"],
        genres=fields["genres"],
        keywords=fields["keywords"],
        director=fields["director"],
        top_cast=fields["top_cast"],
    )
    return fields
