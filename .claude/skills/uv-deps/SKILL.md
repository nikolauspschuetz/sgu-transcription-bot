---
name: uv-deps
description: Correct uv commands for this repo's dependency groups (local, local-mlx, db). Use when installing/syncing deps, adding a package, or when imports break after a uv sync (torch / pyannote / mlx_whisper / psycopg missing).
---

# uv dependency management (this repo)

Groups in `pyproject.toml` → `[dependency-groups]`:
- `local` — faster-whisper, torch, torchaudio, pyannote.audio, soundfile
- `local-mlx` — mlx-whisper (Apple Silicon GPU ASR; no wheels on Intel/Linux)
- `db` — psycopg, pgvector (metadata store)
- `dev`, `scripts` — tooling / legacy server

## ⚠️ GOTCHA: sync ALL runtime groups together

`uv sync --group X` syncs the environment to **base + X only**, and **removes**
packages from any group you didn't name. So `uv sync --group db` silently uninstalls
torch/pyannote/mlx — breaking the pipeline.

**On the M-series box, always sync the full runtime set:**
```
uv sync --group local --group local-mlx --group db
```
(Intel/Linux: drop `--group local-mlx` — mlx has no wheels there.)

**Symptom of getting it wrong:** `ModuleNotFoundError: No module named 'torch' /
'pyannote' / 'mlx_whisper' / 'psycopg'` immediately after a sync. Fix = re-sync with the
full group set above.

## Add a package
Put it in the correct group in `[dependency-groups]`, then re-sync with the full set
above. Don't `uv add` a group-scoped dep into the base `dependencies`.

## Run
`uv run python -m transcription_bot.entrypoints.<name>` — `uv run` uses the synced `.venv`.
