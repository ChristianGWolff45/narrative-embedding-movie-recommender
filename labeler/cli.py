from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from .config import load_settings
from .runner import run_labeling


def _format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "calibrating"
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).astimezone().strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )


def _latest_status(root: Path) -> Path | None:
    paths = sorted(root.glob("runs/*/status.json"), key=lambda path: path.stat().st_mtime)
    return paths[-1] if paths else None


def print_status(path: Path | None) -> None:
    if path is None or not path.exists():
        print("No labeling run has started.")
        return
    status = json.loads(path.read_text(encoding="utf-8"))
    print(f"run:       {status['run_id']}")
    print(f"status:    {status['status']} ({status['phase']})")
    print(f"progress:  {status['completed'] + status['failed']}/{status['total']} processed")
    print(f"completed: {status['completed']}   failed: {status['failed']}")
    print(f"rate:      {status['items_per_minute']} items/min")
    print(f"ETA:       {_format_eta(status.get('eta_seconds'))}")
    print(f"updated:   {status['updated_at']}")
    print(f"last ID:   {status.get('last_movie_id')}")
    if status.get("last_error"):
        print(f"last error: {status['last_error']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CSCE 585 TMDB movie labeler")
    subparsers = parser.add_subparsers(dest="command", required=True)

    label = subparsers.add_parser("label", help="fetch TMDB movies and label them locally")
    label.add_argument("--limit", type=int, default=20, help="movies to process; 0 uses max-pages")
    label.add_argument("--max-pages", type=int, default=1, help="TMDB discover pages to scan")
    label.add_argument("--start-page", type=int, default=1, help="first TMDB page, inclusive")
    label.add_argument("--end-page", type=int, default=None, help="last TMDB page, inclusive")
    label.add_argument("--output", default=None, help="JSONL output path")
    label.add_argument("--model", default=None, help="Hugging Face model ID")
    label.add_argument("--run-id", default=None)

    status = subparsers.add_parser("status", help="show the latest run and ETA")
    status.add_argument("--run-id", default=None)

    args = parser.parse_args()
    root = Path.cwd()
    if args.command == "status":
        path = root / "runs" / args.run_id / "status.json" if args.run_id else _latest_status(root)
        print_status(path)
        return

    settings = load_settings(args.model)
    status_path = run_labeling(
        settings,
        limit=args.limit,
        max_pages=args.max_pages,
        run_id=args.run_id,
        start_page=args.start_page,
        end_page=args.end_page,
        output_file=args.output,
    )
    print_status(status_path)
