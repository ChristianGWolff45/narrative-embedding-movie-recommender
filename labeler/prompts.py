PROMPT_VERSION = "v1"


def labeling_messages(movie: dict) -> list[dict[str, str]]:
    cast = movie.get("cast", [])
    cast_text = ", ".join(
        (
            f"{person.get('name', '').strip()} as {person.get('character', '').strip()}"
            if person.get("character")
            else person.get("name", "").strip()
        )
        for person in cast[:12]
    )
    genres = ", ".join(movie.get("genres", [])) or "unknown"
    keywords = ", ".join(movie.get("keywords", [])) or "not supplied"
    overview = movie.get("overview") or "No synopsis supplied."
    user_prompt = f"""Label this movie for a controllable recommendation system.

Title: {movie.get('title', '')}
Genres: {genres}
TMDB keywords: {keywords}
Director: {movie.get('director', '') or 'not supplied'}
Release date: {movie.get('release_date', '')}
Synopsis: {overview}
Principal cast and characters: {cast_text or 'not supplied'}

Return ONLY one valid JSON object with exactly these keys:
{{
  "themes": ["3-8 concise thematic concepts"],
  "tone": ["2-5 concise tone descriptors"],
  "characters": ["3-8 important character roles, types, or relationships"],
  "narrative_structure": ["3-6 concise story-structure descriptors"],
  "rationale": "one short sentence explaining the labels"
}}

Use only evidence from the supplied metadata. Do not invent names or plot
events. Narrative structure may describe patterns such as investigation,
coming-of-age, quest, ensemble, nonlinear, or revenge when supported by the
synopsis. Keep labels useful for semantic similarity."""
    return [
        {
            "role": "system",
            "content": "You are a careful movie metadata annotator. Output valid JSON only.",
        },
        {"role": "user", "content": user_prompt},
    ]
