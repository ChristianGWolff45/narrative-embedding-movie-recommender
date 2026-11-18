from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path


def _eta(seconds: float | None) -> str:
    if seconds is None:
        return "calibrating"
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).astimezone().strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate sharded labeler status")
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    paths = sorted(Path("runs").glob(f"{args.prefix}-gpu*/status.json"))
    if not paths:
        print(f"No worker status files found for {args.prefix}")
        return
    statuses = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    total = sum(int(item.get("total", 0)) for item in statuses)
    completed = sum(int(item.get("completed", 0)) for item in statuses)
    failed = sum(int(item.get("failed", 0)) for item in statuses)
    processed = completed + failed
    started = min(datetime.fromisoformat(item["started_at"]) for item in statuses)
    elapsed = max((datetime.now(timezone.utc) - started).total_seconds(), 0.001)
    rate = processed / elapsed * 60.0
    remaining = max(total - processed, 0)
    eta_seconds = remaining / (processed / elapsed) if processed else None
    state = "completed" if all(item.get("status") == "completed" for item in statuses) else "running"
    print(f"run:       {args.prefix}")
    print(f"status:    {state}")
    print(f"progress:  {processed}/{total} processed")
    print(f"completed: {completed}   failed: {failed}")
    print(f"rate:      {rate:.2f} items/min aggregate")
    print(f"remaining: {remaining}")
    print(f"ETA:       {_eta(eta_seconds)}")
    print(f"updated:   {max(item['updated_at'] for item in statuses)}")
    for item in statuses:
        print(
            f"gpu {item['run_id'].rsplit('-gpu', 1)[-1]}: "
            f"{item['completed'] + item['failed']}/{item['total']} "
            f"{item['status']} at {item['items_per_minute']} items/min"
        )


if __name__ == "__main__":
    main()
