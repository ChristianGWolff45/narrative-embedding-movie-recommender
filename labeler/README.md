# CSCE 585 Movie Recommender Labeler

This project fetches movie metadata from TMDB and uses a local Hugging Face
instruction model to label themes, tone, characters, and narrative structure.
It writes checkpointed JSONL output and a durable status file for ETA reporting.

## One-time setup

The remote machine already has a CUDA-enabled PyTorch installation. Create the
project environment with access to that installation:

```bash
uv venv --system-site-packages .venv
uv pip install --python .venv/bin/python -e .
```

The TMDB credential belongs outside this repository at:

```text
~/.config/csce585-labeler/.env
```

with mode `600` and this content:

```text
TMDB_BEARER_TOKEN=your_v4_access_token_here
```

The labeler also accepts a v3 key as `TMDB_API_KEY`, but v4 access tokens must
be stored as `TMDB_BEARER_TOKEN` so they are sent in an Authorization header.

## Pilot run

The first run deliberately labels only 20 movies so model output and speed can
be checked:

```bash
bin/run-labeling --limit 20
```

The default local model is `Qwen/Qwen2.5-3B-Instruct`. Override it with:

```bash
bin/run-labeling --limit 20 --model Qwen/Qwen2.5-7B-Instruct
```

Results are written to `data/labels.jsonl`. Each run has its own status file in
`runs/<run-id>/status.json`, and the latest status can be inspected with:

```bash
bin/status
```

For a long-running job, start it in `tmux` so it survives an SSH disconnect:

```bash
tmux new -s csce585-labeler
bin/run-labeling --limit 0 --max-pages 25
```

Detach with `Ctrl-b d` and reconnect with `tmux attach -t csce585-labeler`.
