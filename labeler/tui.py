#!/usr/bin/env python3
"""A small, dependency-free terminal UI for the CSCE 585 recommender."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import time
from pathlib import Path

import numpy as np


DIMENSIONS = ("plot", "themes", "characters", "tone", "narrative_structure")
DIMENSION_LABELS = {
    "plot": "Plot",
    "themes": "Themes",
    "characters": "Characters",
    "tone": "Tone",
    "narrative_structure": "Narrative structure",
}
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def normalize_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


class MovieRecommenderTUI:
    def __init__(self, artifact: Path, report: Path) -> None:
        self.artifact_path = artifact
        self.report_path = report
        if not artifact.exists():
            raise SystemExit(
                f"Artifact not found: {artifact}\n"
                "Build it first with: bin/reconstruct build"
            )
        self.data = np.load(artifact)
        self.metadata = json.loads(
            artifact.with_suffix(".metadata.json").read_text(encoding="utf-8")
        )
        self.titles = [item.get("title", "") for item in self.metadata]
        self.title_lookup: dict[str, list[int]] = {}
        for index, title in enumerate(self.titles):
            self.title_lookup.setdefault(normalize_title(title), []).append(index)

    def run(self) -> None:
        print("\nCSCE 585 MOVIE RECOMMENDER")
        print("First-pass multi-embedding demonstration")
        print(f"Loaded {len(self.metadata):,} movies from {self.artifact_path}")
        print("Type b at a movie prompt to go back.\n")
        while True:
            try:
                self.print_menu()
                choice = input("Choose an option: ").strip().lower()
                if choice in {"6", "q", "quit", "exit"}:
                    print("Goodbye.")
                    return
                if choice == "1":
                    self.multi_recommendation()
                elif choice == "2":
                    self.single_recommendation()
                elif choice == "3":
                    self.compare_movies()
                elif choice == "4":
                    self.run_benchmark()
                elif choice == "5":
                    self.show_model_info()
                else:
                    print("Please choose 1, 2, 3, 4, 5, or 6.\n")
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                return

    @staticmethod
    def print_menu() -> None:
        print("\n" + "=" * 72)
        print("1. Recommend with the multi-embedding model")
        print("2. Recommend with the single-embedding baseline")
        print("3. Compare two films across narrative dimensions")
        print("4. Benchmark single vs. multi retrieval")
        print("5. Show model and corpus information")
        print("6. Exit")
        print("=" * 72)

    def prompt_movie(self, prompt: str) -> int | None:
        while True:
            raw = input(prompt).strip()
            if raw.casefold() in {"b", "back", "q", "quit"}:
                return None
            if not raw:
                print("Please type a film title.")
                continue

            normalized = normalize_title(raw)
            exact = self.title_lookup.get(normalized, [])
            if len(exact) == 1:
                return exact[0]
            if len(exact) > 1:
                return self.choose_match(raw, exact)

            matches = [
                index
                for index, title in enumerate(self.titles)
                if normalized in normalize_title(title)
            ]
            if len(matches) == 1:
                return matches[0]
            if matches:
                return self.choose_match(raw, matches[:12])

            suggestions = difflib.get_close_matches(raw, self.titles, n=6, cutoff=0.25)
            print(f"No exact match for {raw!r}.")
            if suggestions:
                print("Possible matches: " + "; ".join(suggestions))
            print("Try again, or type b to return.")

    def choose_match(self, query: str, matches: list[int]) -> int | None:
        print(f"Matches for {query!r}:")
        for number, index in enumerate(matches, 1):
            item = self.metadata[index]
            genres = ", ".join(item.get("genres", []))
            suffix = f" ({genres})" if genres else ""
            print(f"  {number:2d}. {self.titles[index]}{suffix}")
        while True:
            choice = input("Select a number, or type another title: ").strip()
            if choice.casefold() in {"b", "back", "q", "quit"}:
                return None
            if choice.isdigit() and 1 <= int(choice) <= len(matches):
                return matches[int(choice) - 1]
            if choice:
                normalized = normalize_title(choice)
                exact = self.title_lookup.get(normalized, [])
                if len(exact) == 1:
                    return exact[0]
            print("Please select one of the listed numbers or type a title.")

    @staticmethod
    def prompt_int(prompt: str, default: int, minimum: int = 1, maximum: int = 100) -> int:
        while True:
            raw = input(f"{prompt} [{default}]: ").strip()
            if not raw:
                return default
            try:
                value = int(raw)
            except ValueError:
                print("Please enter a whole number.")
                continue
            if minimum <= value <= maximum:
                return value
            print(f"Enter a number from {minimum} to {maximum}.")

    @staticmethod
    def prompt_float(prompt: str, default: float) -> float:
        while True:
            raw = input(f"{prompt} [{default:g}]: ").strip()
            if not raw:
                return default
            try:
                value = float(raw)
            except ValueError:
                print("Please enter a number.")
                continue
            if value >= 0:
                return value
            print("Weights cannot be negative.")

    def prompt_weights(self) -> dict[str, float] | None:
        choice = input("Use equal weights for all five dimensions? [Y/n]: ").strip().lower()
        if choice in {"b", "back", "q", "quit"}:
            return None
        weights = {name: 1.0 for name in DIMENSIONS}
        if choice not in {"n", "no"}:
            return weights
        print("Enter 0 to ignore a dimension, or a larger number to emphasize it.")
        for name in DIMENSIONS:
            weights[name] = self.prompt_float(f"  {DIMENSION_LABELS[name]} weight", 1.0)
        if sum(weights.values()) == 0:
            print("All weights were zero; using equal weights instead.")
            return {name: 1.0 for name in DIMENSIONS}
        return weights

    def multi_scores(self, query_index: int, weights: dict[str, float]) -> np.ndarray:
        total = sum(weights.values()) or 1.0
        scores = np.zeros(len(self.metadata), dtype=np.float32)
        for name, weight in weights.items():
            scores += (weight / total) * (self.data[name] @ self.data[name][query_index])
        return scores

    def top_indices(self, scores: np.ndarray, query_index: int, count: int) -> np.ndarray:
        scores = scores.copy()
        scores[query_index] = -np.inf
        return np.argsort(scores)[-count:][::-1]

    def print_results(self, heading: str, indices: np.ndarray, scores: np.ndarray) -> None:
        print(f"\n{heading}")
        print("-" * 72)
        print(f"{'#':>2}  {'Film':<47} {'Score':>8}  Genres")
        print("-" * 72)
        for rank, index in enumerate(indices, 1):
            title = self.titles[index]
            genres = ", ".join(self.metadata[index].get("genres", []))
            print(f"{rank:2d}  {title[:47]:<47} {scores[index]:8.4f}  {genres[:35]}")

    def multi_recommendation(self) -> None:
        query_index = self.prompt_movie("Type a film title: ")
        if query_index is None:
            return
        count = self.prompt_int("How many recommendations", 10, 1, 50)
        weights = self.prompt_weights()
        if weights is None:
            return
        scores = self.multi_scores(query_index, weights)
        indices = self.top_indices(scores, query_index, count)
        weight_text = ", ".join(
            f"{DIMENSION_LABELS[name]}={value:g}" for name, value in weights.items() if value
        )
        self.print_results(
            f"Multi-embedding recommendations for {self.titles[query_index]}\n"
            f"Weights: {weight_text}",
            indices,
            scores,
        )
        self.pause()

    def single_recommendation(self) -> None:
        query_index = self.prompt_movie("Type a film title: ")
        if query_index is None:
            return
        count = self.prompt_int("How many recommendations", 10, 1, 50)
        scores = self.data["single"] @ self.data["single"][query_index]
        indices = self.top_indices(scores, query_index, count)
        self.print_results(
            f"Single-embedding recommendations for {self.titles[query_index]}",
            indices,
            scores,
        )
        self.pause()

    def compare_movies(self) -> None:
        first = self.prompt_movie("First film title: ")
        if first is None:
            return
        second = self.prompt_movie("Second film title: ")
        if second is None:
            return
        print(f"\n{self.titles[first]}  vs.  {self.titles[second]}")
        print("-" * 52)
        print(f"{'Dimension':<24} Similarity")
        print("-" * 52)
        dimension_scores: dict[str, float] = {}
        for name in DIMENSIONS:
            value = float(self.data[name][first] @ self.data[name][second])
            dimension_scores[name] = value
            print(f"{DIMENSION_LABELS[name]:<24} {value: .4f}")
        single = float(self.data["single"][first] @ self.data["single"][second])
        multi = sum(dimension_scores.values()) / len(DIMENSIONS)
        print("-" * 52)
        print(f"{'Single embedding':<24} {single: .4f}")
        print(f"{'Multi equal-weight':<24} {multi: .4f}")
        print("\nHigher scores mean the two films are closer in that representation.")
        self.pause()

    def run_benchmark(self) -> None:
        queries = self.prompt_int("Number of benchmark queries", 200, 1, 5000)
        count = len(self.metadata)
        rng = np.random.default_rng(585)
        query_indices = rng.integers(0, count, size=queries)

        started = time.perf_counter()
        single_scores = self.data["single"][query_indices] @ self.data["single"].T
        np.argpartition(single_scores, -10, axis=1)[:, -10:]
        single_seconds = time.perf_counter() - started

        started = time.perf_counter()
        multi_scores = sum(
            self.data[name][query_indices] @ self.data[name].T for name in DIMENSIONS
        ) / len(DIMENSIONS)
        np.argpartition(multi_scores, -10, axis=1)[:, -10:]
        multi_seconds = time.perf_counter() - started

        result = {
            "movies": count,
            "queries": queries,
            "single_mean_ms": single_seconds / queries * 1000,
            "multi_mean_ms": multi_seconds / queries * 1000,
            "single_qps": queries / single_seconds,
            "multi_qps": queries / multi_seconds,
        }
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("\nBenchmark complete")
        print("-" * 52)
        print(f"Movies tested:                 {count:,}")
        print(f"Queries:                       {queries:,}")
        print(f"Single mean latency:           {result['single_mean_ms']:.3f} ms")
        print(f"Multi mean latency:            {result['multi_mean_ms']:.3f} ms")
        print(f"Single throughput:             {result['single_qps']:.1f} queries/sec")
        print(f"Multi throughput:              {result['multi_qps']:.1f} queries/sec")
        print(f"Saved report:                  {self.report_path}")
        self.pause()

    def show_model_info(self) -> None:
        vector_size = self.data["single"].shape[1]
        print("\nModel and corpus information")
        print("-" * 52)
        print("Encoder:                      " + ENCODER)
        print(f"Movies in artifact:           {len(self.metadata):,}")
        print(f"Embedding dimensions:         {vector_size}")
        print("Stored representations:       single + 5 narrative vectors")
        print("Narrative vectors:             " + ", ".join(DIMENSION_LABELS[name] for name in DIMENSIONS))
        print("Ranking:                      weighted cosine similarity")
        print("Labeling status:              first pass; automated labels")
        print("\nThis is the presentation/demo artifact, not a final validated model.")
        self.pause()

    @staticmethod
    def pause() -> None:
        try:
            input("\nPress Enter to return to the menu...")
        except (EOFError, KeyboardInterrupt):
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive CSCE 585 movie recommender")
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/movie_embeddings.npz"))
    parser.add_argument("--report", type=Path, default=Path("reports/reconstruction_benchmark.json"))
    args = parser.parse_args()
    MovieRecommenderTUI(args.artifact, args.report).run()


if __name__ == "__main__":
    main()
