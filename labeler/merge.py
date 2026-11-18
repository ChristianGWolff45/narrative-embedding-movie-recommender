from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> None:
    data_dir = Path("data")
    source = data_dir / "labels.jsonl"
    shards = sorted((data_dir / "shards").glob("labels_gpu*.jsonl"))
    if not shards:
        raise SystemExit("No shard files found")
    temporary = data_dir / "labels.merged.jsonl"
    seen: set[int] = set()
    count = 0
    with temporary.open("w", encoding="utf-8") as output:
        for path in [source, *shards]:
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                    movie_id = int(record["movie"]["tmdb_id"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                if movie_id in seen:
                    continue
                seen.add(movie_id)
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    os.replace(temporary, source)
    print(f"merged {count} unique records from {len(shards)} shards into {source}")


if __name__ == "__main__":
    main()
