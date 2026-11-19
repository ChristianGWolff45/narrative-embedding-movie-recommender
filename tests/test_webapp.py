import argparse
import json
import threading
from http.server import ThreadingHTTPServer
import urllib.request

import numpy as np
import pytest

from webapp import server


def test_parse_filters_validates():
    genres = {"Drama", "Comedy"}
    assert server.parse_filters(None, genres) == {}
    assert server.parse_filters({"genres": ["Drama"], "yearMin": "1990"}, genres) == {"genres": ["Drama"], "yearMin": 1990}
    for bad in (
        {"genres": ["Western"]},
        {"yearMin": 2000, "yearMax": 1990},
        {"yearMin": "abc"},
        {"yearMax": 3000},
        {"genres": ["Drama"], "excludeGenres": ["Drama"]},
        "Drama",
    ):
        with pytest.raises(ValueError):
            server.parse_filters(bad, genres)


def test_filter_mask():
    sets = [{"Drama"}, {"Comedy"}, {"Drama", "Horror"}, set()]
    years = np.array([1990, 2005, 2010, 0])
    assert server.filter_mask(sets, years, {}) is None
    assert list(server.filter_mask(sets, years, {"genres": ["Drama"]})) == [True, False, True, False]
    assert list(server.filter_mask(sets, years, {"excludeGenres": ["Horror"]})) == [True, True, False, True]
    assert list(server.filter_mask(sets, years, {"yearMin": 2000, "yearMax": 2008})) == [False, True, False, False]


@pytest.fixture(scope="module")
def live_app(synthetic, tmp_path_factory):
    """Run the real RecommenderApp and HTTP handler over the synthetic catalog."""
    import config, embed_movies, recommend

    root = tmp_path_factory.mktemp("web")
    catalog = root / "movie_catalog.jsonl"
    catalog.write_text(
        "\n".join(json.dumps({**item, "tmdbId": item["tmdb_id"], "movieId": item["movie_id"]}) for item in synthetic["metadata"]),
        encoding="utf-8",
    )
    patch = pytest.MonkeyPatch()
    patch.setattr(embed_movies, "CATALOG_PATH", catalog)
    patch.setattr(recommend, "EMBEDDINGS_PATH", synthetic["baseline"])
    patch.setattr(recommend, "MODEL_INFO_PATH", root / "missing.json")
    patch.setattr(server, "EMBEDDINGS_PATH", synthetic["baseline"])
    patch.setattr(server, "MODEL_INFO_PATH", root / "missing.json")
    patch.setattr(server, "ARTIFACT", synthetic["artifact"])
    patch.setattr(server, "ARTIFACT_METADATA", synthetic["artifact"].with_suffix(".metadata.json"))
    patch.setattr(server, "ROOT", root)
    app = server.RecommenderApp()
    patch.setattr(server, "APP", app)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    patch.setattr(server.Handler, "log_message", lambda *a, **k: None)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield app, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    patch.undo()


def _post(url, payload):
    request = urllib.request.Request(f"{url}/api/recommend", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_status_lists_genres_and_years(live_app):
    app, url = live_app
    with urllib.request.urlopen(f"{url}/api/status") as response:
        status = json.loads(response.read())
    assert status["multiReady"] is True
    assert "Drama" in status["genres"]
    assert status["yearRange"] == [1960, 2019]


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_filters_constrain_results(live_app, mode):
    app, url = live_app
    filters = {"genres": ["Drama"], "excludeGenres": ["Comedy"], "yearMin": 1980, "yearMax": 2000}
    code, body = _post(url, {"seeds": [1000], "mode": mode, "count": 24, "filters": filters})
    assert code == 200
    assert body["results"] and body["matchingCount"] < len(app.records)
    for movie in body["results"]:
        assert "Drama" in movie["genres"] and "Comedy" not in movie["genres"]
        assert 1980 <= int(movie["year"]) <= 2000
        assert movie["tmdbId"] != 1000


def test_filters_that_match_nothing_return_empty(live_app):
    app, url = live_app
    code, body = _post(url, {"seeds": [1000], "mode": "multi", "filters": {"yearMin": 2099}})
    assert code == 200 and body["results"] == [] and body["matchingCount"] == 0


def test_bad_filter_is_a_400(live_app):
    app, url = live_app
    code, body = _post(url, {"seeds": [1000], "filters": {"genres": ["Western"]}})
    assert code == 400 and "Unknown genre" in body["error"]


def test_loadtest_against_live_server(live_app, synthetic, tmp_path):
    from trainer import loadtest

    app, url = live_app
    args = argparse.Namespace(url=url, artifact=str(synthetic["artifact"]), modes=["single", "multi"],
                              rates=[20, 40], duration=1.0, workers=16, timeout=5.0, seed=1,
                              latency_budget_ms=1000.0)
    result = loadtest.loadtest(args)
    assert len(result["results"]) == 4
    for row in result["results"]:
        assert row["errors"] == 0 and row["latency"]["count"] == row["requests"]
    assert result["max_rate_within_budget"] == {"single": 40, "multi": 40}
