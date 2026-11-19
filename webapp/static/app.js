(() => {
  "use strict";

  const $ = (selector, parent = document) => parent.querySelector(selector);
  const $$ = (selector, parent = document) => [...parent.querySelectorAll(selector)];
  const IMAGE_BASE = "https://image.tmdb.org/t/p/w500";
  const weights = { plot: 1, themes: 1, characters: 1, tone: 1, narrative_structure: 1 };
  const state = { seeds: [], mode: "single", status: null, searchTimer: null, recommendationTimer: null };

  const searchInput = $("#film-search");
  const suggestions = $("#search-results");
  const selectedFilms = $("#selected-films");
  const resultsGrid = $("#results-grid");
  const findButton = $("#find-button");
  const announcer = $("#announcer");

  function announce(message) {
    announcer.textContent = "";
    window.setTimeout(() => { announcer.textContent = message; }, 30);
  }

  async function getJson(url, options) {
    const response = await fetch(url, { cache: "no-store", ...options });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Something went wrong.");
    return data;
  }

  function posterUrl(path) {
    if (typeof path !== "string" || !/^\/[A-Za-z0-9_./-]+$/.test(path) || path.includes("..")) return null;
    return `${IMAGE_BASE}${path}`;
  }

  function makePoster(movie, className = "poster-wrap") {
    const frame = document.createElement("div");
    frame.className = className;
    frame.setAttribute("aria-hidden", "true");
    const url = posterUrl(movie.backdropPath || movie.posterPath);
    if (url) {
      const image = document.createElement("img");
      image.src = url;
      image.alt = "";
      image.loading = "lazy";
      image.decoding = "async";
      image.addEventListener("error", () => {
        frame.replaceChildren();
        const fallback = document.createElement("div");
        fallback.className = "poster-placeholder";
        const monogram = document.createElement("span");
        monogram.textContent = (movie.title || "R").trim().charAt(0).toUpperCase();
        fallback.append(monogram);
        frame.append(fallback);
      }, { once: true });
      frame.append(image);
    } else {
      const fallback = document.createElement("div");
      fallback.className = "poster-placeholder";
      const monogram = document.createElement("span");
      monogram.textContent = (movie.title || "R").trim().charAt(0).toUpperCase();
      fallback.append(monogram);
      frame.append(fallback);
    }
    return frame;
  }

  function hideSuggestions() {
    suggestions.hidden = true;
    searchInput.setAttribute("aria-expanded", "false");
  }

  function addSeed(movie) {
    if (state.seeds.some(seed => seed.tmdbId === movie.tmdbId)) {
      announce(`${movie.title} is already selected.`);
      hideSuggestions();
      searchInput.value = "";
      return;
    }
    if (state.seeds.length >= 3) {
      announce("Select up to three movies.");
      return;
    }
    state.seeds.push(movie);
    searchInput.value = "";
    hideSuggestions();
    renderSeeds();
    findButton.disabled = false;
    announce(`${movie.title} selected.`);
  }

  function renderSeeds() {
    selectedFilms.replaceChildren();
    $("#seed-count").textContent = `${state.seeds.length} / 3`;
    if (!state.seeds.length) {
      const empty = document.createElement("p");
      empty.className = "empty-selection";
      empty.textContent = "No movies selected.";
      selectedFilms.append(empty);
      findButton.disabled = true;
      return;
    }
    state.seeds.forEach(movie => {
      const chip = document.createElement("span");
      chip.className = "seed-chip";
      const title = document.createElement("span");
      title.textContent = movie.year ? `${movie.title} (${movie.year})` : movie.title;
      const remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "×";
      remove.setAttribute("aria-label", `Remove ${movie.title}`);
      remove.addEventListener("click", () => {
        state.seeds = state.seeds.filter(seed => seed.tmdbId !== movie.tmdbId);
        renderSeeds();
        if (state.seeds.length) runRecommendations();
        else showEmptyResults("Select a movie to get recommendations.");
        announce(`${movie.title} removed.`);
      });
      chip.append(title, remove);
      selectedFilms.append(chip);
    });
  }

  function renderSearchResults(movies, query) {
    suggestions.replaceChildren();
    if (!movies.length) {
      const empty = document.createElement("div");
      empty.className = "search-empty";
      empty.textContent = `No matches for “${query}”. Try another title.`;
      suggestions.append(empty);
    } else {
      movies.forEach(movie => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "suggestion";
        if (movie.posterPath) {
          const image = document.createElement("img");
          image.src = posterUrl(movie.posterPath) || "";
          image.alt = "";
          image.loading = "lazy";
          button.append(image);
        } else {
          const image = document.createElement("span");
          image.className = "suggestion-poster";
          image.setAttribute("aria-hidden", "true");
          button.append(image);
        }
        const info = document.createElement("span");
        const title = document.createElement("span");
        title.className = "suggestion-title";
        title.textContent = movie.title;
        const meta = document.createElement("span");
        meta.className = "suggestion-meta";
        meta.textContent = [movie.year, ...(movie.genres || []).slice(0, 2)].filter(Boolean).join(" · ");
        info.append(title, meta);
        button.append(info);
        button.addEventListener("click", () => addSeed(movie));
        suggestions.append(button);
      });
    }
    suggestions.hidden = false;
    searchInput.setAttribute("aria-expanded", "true");
  }

  async function searchMovies(query) {
    if (query.trim().length < 2) {
      hideSuggestions();
      return;
    }
    try {
      const data = await getJson(`/api/search?q=${encodeURIComponent(query.trim())}`);
      if (searchInput.value.trim() !== query.trim()) return;
      renderSearchResults(data.results || [], query.trim());
    } catch (error) {
      renderSearchResults([], query.trim());
      announce(error.message);
    }
  }

  function createMovieCard(movie) {
    const article = document.createElement("article");
    article.className = "movie-card";
    article.append(makePoster(movie));

    const body = document.createElement("div");
    body.className = "movie-card-body";
    const heading = document.createElement("div");
    heading.className = "movie-title-row";
    const title = document.createElement("h3");
    title.className = "movie-title";
    title.textContent = movie.title;
    const year = document.createElement("span");
    year.className = "movie-year";
    year.textContent = movie.year || "";
    heading.append(title, year);
    body.append(heading);

    const genres = document.createElement("p");
    genres.className = "movie-genres";
    genres.textContent = (movie.genres || []).slice(0, 3).join(" · ") || "Feature film";
    body.append(genres);

    const overview = document.createElement("p");
    overview.className = "movie-overview";
    overview.textContent = movie.overview || "Synopsis unavailable for this film.";
    body.append(overview);

    const bottom = document.createElement("div");
    bottom.className = "movie-card-bottom";
    const ratingCount = document.createElement("span");
    ratingCount.className = "rating-count";
    ratingCount.textContent = `${Number(movie.ratingCount || 0).toLocaleString()} MovieLens ratings`;
    const add = document.createElement("button");
    add.type = "button";
    add.className = "add-reference";
    const selected = state.seeds.some(seed => seed.tmdbId === movie.tmdbId);
    add.textContent = selected ? "Added" : "+ Add";
    add.disabled = selected || state.seeds.length >= 3;
    add.setAttribute("aria-label", selected ? `${movie.title} is already selected` : `Add ${movie.title}`);
    add.addEventListener("click", () => {
      addSeed(movie);
      renderResults(state.lastResults || []);
    });
    bottom.append(ratingCount, add);
    body.append(bottom);
    article.append(body);

    if (movie.score !== null && movie.score !== undefined) {
      const score = document.createElement("span");
      score.className = "match-score";
      score.textContent = `Score ${Number(movie.score).toFixed(3)}`;
      article.querySelector(".poster-wrap").append(score);
    }
    return article;
  }

  function showEmptyResults(message, title = "Select at least one movie.") {
    resultsGrid.replaceChildren();
    const panel = document.createElement("div");
    panel.className = "results-empty";
    const headline = document.createElement("p");
    headline.className = "empty-title";
    headline.textContent = title;
    const copy = document.createElement("p");
    copy.textContent = message;
    panel.append(headline, copy);
    resultsGrid.append(panel);
  }

  function renderResults(movies) {
    state.lastResults = movies;
    resultsGrid.replaceChildren();
    if (!movies.length) {
      showEmptyResults("Try another movie.", "No recommendations.");
      return;
    }
    movies.forEach(movie => resultsGrid.append(createMovieCard(movie)));
  }

  async function runRecommendations() {
    if (!state.seeds.length) return;
    findButton.disabled = true;
    findButton.classList.add("is-loading");
    findButton.firstChild.textContent = "Working… ";
    resultsGrid.setAttribute("aria-busy", "true");
    showEmptyResults("Generating recommendations…", "Please wait");
    const selectedWeights = {};
    $$('[data-weight]').forEach(input => { selectedWeights[input.dataset.weight] = Number(input.value); });
    try {
      const data = await getJson("/api/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          seeds: state.seeds.map(movie => movie.tmdbId),
          mode: state.mode,
          count: 12,
          weights: selectedWeights,
          filters: readFilters(),
        }),
      });
      renderResults(data.results || []);
      const seedNames = state.seeds.map(movie => movie.title).join(", ");
      const profile = data.mode === "multi" ? "Narrative profile" : "Single profile";
      const filtered = data.filters && Object.keys(data.filters).length
        ? ` · ${Number(data.matchingCount || 0).toLocaleString()} films match filters`
        : "";
      $("#result-description").textContent = `${profile}: ${seedNames}${filtered}`;
      announce(`${(data.results || []).length} recommendations loaded.`);
    } catch (error) {
      showEmptyResults(error.message, "Recommendation error");
      announce(error.message);
    } finally {
      resultsGrid.setAttribute("aria-busy", "false");
      findButton.disabled = state.seeds.length === 0;
      findButton.classList.remove("is-loading");
      findButton.firstChild.textContent = "Recommend";
    }
  }

  function readFilters() {
    const filters = {};
    const include = $("#filter-genre").value;
    const exclude = $("#filter-exclude").value;
    if (include) filters.genres = [include];
    if (exclude) filters.excludeGenres = [exclude];
    ["yearMin", "yearMax"].forEach(key => {
      const input = $(`[data-filter="${key}"]`);
      const value = input.value.trim();
      if (value && input.checkValidity()) filters[key] = Number(value);
    });
    return filters;
  }

  function updateFilterSummary() {
    const filters = readFilters();
    const parts = [];
    if (filters.genres) parts.push(filters.genres[0]);
    if (filters.excludeGenres) parts.push(`no ${filters.excludeGenres[0]}`);
    if (filters.yearMin || filters.yearMax) parts.push(`${filters.yearMin || "…"}–${filters.yearMax || "…"}`);
    $("#filter-summary").textContent = parts.length ? parts.join(" · ") : "No filters";
    $("#filter-clear").disabled = parts.length === 0;
    $$("[data-filter^='year']").forEach(input => {
      input.setAttribute("aria-invalid", String(Boolean(input.value) && !input.checkValidity()));
    });
  }

  function populateFilters(data) {
    const genres = data.genres || [];
    [["#filter-genre", "Any genre"], ["#filter-exclude", "None"]].forEach(([selector, emptyLabel]) => {
      const select = $(selector);
      if (select.options.length > 1) return;
      select.replaceChildren(new Option(emptyLabel, ""), ...genres.map(name => new Option(name, name)));
    });
    if (Array.isArray(data.yearRange)) {
      $$("[data-filter^='year']").forEach(input => {
        input.min = String(data.yearRange[0]);
        input.max = String(data.yearRange[1]);
      });
    }
  }

  function renderFeatured(movies) {
    const host = $("#featured-films");
    host.replaceChildren();
    movies.slice(0, 6).forEach(movie => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "quick-pick";
      button.textContent = movie.title;
      button.addEventListener("click", () => addSeed(movie));
      host.append(button);
    });
  }

  function updateWeights() {
    $$('[data-weight]').forEach(input => {
      const value = Number(input.value);
      const output = $(`#value-${input.dataset.weight}`);
      output.value = value.toFixed(1);
      output.textContent = value.toFixed(1);
      input.setAttribute("aria-valuetext", `${value.toFixed(1)} times weight`);
    });
  }

  function setMode(mode) {
    if (mode === "multi" && !state.status?.multiReady) return;
    state.mode = mode;
    $$(".mode-button").forEach(button => {
      const active = button.dataset.mode === mode;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    $$("[data-weight]").forEach(input => { input.disabled = mode !== "multi" || !state.status?.multiReady; });
    $("#mix-status").textContent = state.status?.multiReady ? "Ready" : "Loading";
    $("#mix-status").classList.toggle("is-ready", Boolean(state.status?.multiReady));
    $("#progress-detail").textContent = mode === "multi"
      ? "Adjust weights and select Recommend."
      : (state.status?.multiReady ? "Profiles loaded." : "Single profile is ready.");
    if (state.seeds.length) runRecommendations();
  }

  function updateEvaluation(report) {
    const target = $("#evaluation-state");
    if (!report?.methods) {
      target.textContent = "Not yet evaluated";
      return;
    }
    const single = report.methods.single?.co_like_rate_paired?.mean;
    const multi = report.methods.multi?.co_like_rate_paired?.mean;
    if (Number.isFinite(single) && Number.isFinite(multi)) {
      target.textContent = `${single.toFixed(2)} single · ${multi.toFixed(2)} narrative`;
    } else {
      target.textContent = `${report.queries_evaluated || 0} queries scored`;
    }
  }

  function updateStatus(data) {
    state.status = data;
    $("#movie-count").textContent = Number(data.movieCount || 0).toLocaleString();
    $("#model-name").textContent = data.baselineModel || "all-MiniLM-L6-v2";
    const status = $("#collection-status");
    status.classList.toggle("is-ready", Number(data.movieCount) > 0);
    $("#collection-label").textContent = Number(data.movieCount) > 0 ? "Ready" : "Loading";
    renderFeatured(data.featured || []);
    populateFilters(data);

    const percent = Math.min(100, Math.max(0, Number(data.labelPercent || 0)));
    $("#progress-value").textContent = `${percent}%`;
    $("#progress-fill").style.width = `${percent}%`;
    $("#progressbar").setAttribute("aria-valuenow", String(percent));
    $("#progress-label").textContent = `${Number(data.labelCount || 0).toLocaleString()} / ${Number(data.movieCount || 0).toLocaleString()} narrative profiles`;
    const run = data.labelRun;
    if (data.multiReady) {
      $("#progress-detail").textContent = "Profiles loaded.";
      $("#mix-status").textContent = "Ready";
      $("#mix-status").classList.add("is-ready");
      $("#multi-mode").disabled = false;
      $$("[data-weight]").forEach(input => { input.disabled = state.mode !== "multi"; });
    } else {
      $("#multi-mode").disabled = true;
      $$("[data-weight]").forEach(input => { input.disabled = true; });
      if (run?.status === "running") {
        const done = Number(run.completed || 0) + Number(run.failed || 0);
        const rate = Number(run.itemsPerMinute || 0);
        const eta = rate > 0 ? ` · about ${Math.ceil(Number(run.remaining || 0) / rate)} min left` : "";
        $("#progress-detail").textContent = `Processing ${done.toLocaleString()} of ${Number(run.total || 0).toLocaleString()}${eta}.`;
      } else if (Number(data.labelCount || 0)) {
        $("#progress-detail").textContent = `${Number(data.labelCount).toLocaleString()} profiles loaded.`;
      } else {
        $("#progress-detail").textContent = "Single profile is ready.";
      }
    }
    updateEvaluation(data.evaluation);
  }

  async function refreshStatus() {
    try {
      const data = await getJson("/api/status");
      updateStatus(data);
    } catch (_error) {
      $("#collection-label").textContent = "Unavailable";
    }
  }

  searchInput.addEventListener("input", () => {
    window.clearTimeout(state.searchTimer);
    const query = searchInput.value;
    state.searchTimer = window.setTimeout(() => searchMovies(query), 180);
  });
  searchInput.addEventListener("keydown", event => {
    if (event.key === "Escape") hideSuggestions();
    if (event.key === "Enter" && !suggestions.hidden) {
      const first = $(".suggestion", suggestions);
      if (first) { event.preventDefault(); first.click(); }
    }
  });
  document.addEventListener("click", event => {
    if (!event.target.closest(".search-wrap")) hideSuggestions();
  });
  document.addEventListener("keydown", event => {
    if (event.key === "Escape") hideSuggestions();
  });
  document.addEventListener("keydown", event => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      searchInput.focus();
    }
  });
  findButton.addEventListener("click", runRecommendations);
  $$(".mode-button").forEach(button => button.addEventListener("click", () => setMode(button.dataset.mode)));
  $$('[data-weight]').forEach(input => input.addEventListener("input", () => {
    weights[input.dataset.weight] = Number(input.value);
    updateWeights();
    window.clearTimeout(state.recommendationTimer);
    state.recommendationTimer = window.setTimeout(() => {
      if (state.mode === "multi" && state.seeds.length) runRecommendations();
    }, 250);
  }));

  $$("[data-filter]").forEach(control => control.addEventListener("input", () => {
    updateFilterSummary();
    window.clearTimeout(state.recommendationTimer);
    state.recommendationTimer = window.setTimeout(() => {
      if (state.seeds.length) runRecommendations();
    }, 350);
  }));
  $("#filter-clear").addEventListener("click", () => {
    $$("[data-filter]").forEach(control => { control.value = ""; });
    updateFilterSummary();
    announce("Filters cleared.");
    if (state.seeds.length) runRecommendations();
  });

  updateWeights();
  updateFilterSummary();
  refreshStatus();
  window.setInterval(refreshStatus, 15_000);
})();
