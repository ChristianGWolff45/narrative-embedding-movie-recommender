from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(path: Path, status: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def update_metrics(status: dict[str, Any]) -> None:
    started = datetime.fromisoformat(status["started_at"])
    elapsed = max((datetime.now(timezone.utc) - started).total_seconds(), 0.001)
    processed = status.get("completed", 0) + status.get("failed", 0)
    status["updated_at"] = now_iso()
    status["items_per_minute"] = round(processed / elapsed * 60.0, 3)
    remaining = max(status["total"] - processed, 0)
    status["remaining"] = remaining
    status["eta_seconds"] = (
        round(remaining / (processed / elapsed), 1) if processed else None
    )


def new_status(run_id: str, total: int, model_id: str, output_file: str) -> dict[str, Any]:
    started = now_iso()
    return {
        "run_id": run_id,
        "status": "starting",
        "phase": "initializing",
        "started_at": started,
        "updated_at": started,
        "model_id": model_id,
        "total": total,
        "completed": 0,
        "failed": 0,
        "remaining": total,
        "items_per_minute": 0.0,
        "eta_seconds": None,
        "last_movie_id": None,
        "last_error": None,
        "output_file": output_file,
    }
