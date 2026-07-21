# Setup / porting guide

This fork develops on a low-power Mac and runs full-quality on a GPU box. Only the
sizing changes between them — the code and pipeline are identical. Everything here is
free / local / open-source; no paid services.

See `PLAN.md` for the architecture and the GitHub issues for the work backlog.

## Clone (on the GPU box)

```bash
git clone https://github.com/nikolauspschuetz/transcription-bot.git
cd transcription-bot
git checkout local-backend        # the working branch
```

## Prerequisites

- **Python 3.12** (pinned in `pyproject.toml`; `uv` will fetch it)
- **ffmpeg** on PATH — Whisper needs it
  - macOS: `brew install ffmpeg`
  - Debian/Ubuntu: `sudo apt-get install -y ffmpeg`
- **uv** — https://docs.astral.sh/uv/
- A **HuggingFace token** (free) with the `pyannote/speaker-diarization-3.1` terms
  accepted: https://huggingface.co/settings/tokens

## Install

```bash
uv sync                 # base deps
uv sync --group local   # + faster-whisper, pyannote.audio, torch, torchaudio, soundfile
```

### The target box is an Apple Silicon M4 mini

Two Apple-specific facts drive the config:

- **ASR:** faster-whisper (CTranslate2) has **no Metal backend**, so `local_whisper` is
  CPU-only on Apple Silicon. To use the 10-core GPU, use the **MLX** backend:
  ```bash
  uv sync --group local --group local-mlx   # adds mlx-whisper (Apple Silicon only)
  ```
  then set `TB_TRANSCRIPTION_BACKEND=local_mlx_whisper` (Profile B).
- **Diarization:** pyannote runs on PyTorch, which supports Apple Silicon via the `mps`
  device. Set `TB_DIARIZATION_DEVICE=mps`. A few pyannote ops aren't MPS-implemented, so
  export the fallback so they drop to CPU instead of erroring:
  ```bash
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  ```
  If MPS gives trouble, `TB_DIARIZATION_DEVICE=cpu` is a reliable fallback (diarization
  is far lighter than ASR).

Verify torch sees the GPU: `uv run python -c "import torch; print(torch.backends.mps.is_available())"` → `True`.

> Note: `uv sync --group local` (no `local-mlx`) also works on Intel Macs / Linux for
> CPU transcription — the MLX group is what's Apple-only.

## Configure

```bash
cp .env.template .env
```

Then edit `.env`:
- Set `TB_HF_TOKEN=...`
- **Dev Mac (Intel, CPU):** keep Profile A (`medium.en` / `cpu` / `int8`).
- **M4 mini:** uncomment Profile B (`local_mlx_whisper` on the GPU + `mps` diarization,
  optional Ollama).

Credentials for paid services stay blank — they're only validated if you switch a
backend back to a paid option, which we don't.

## Status: what runs today vs. what's next

The **config layer is done** (backend switches + sizing). Not yet implemented (tracked
as issues — do these on the GPU box):

- **#2** local Whisper ASR backend
- **#3** local pyannote diarization backend
- **#4** backend dispatch wiring
- **#5** local speaker identification
- **#6** free/optional LLM helper
- **#7** WebVTT + Markdown output
- **#8** review gate
- **#9** end-to-end dry run on episode 1096

Until #2–#4 land there is no runnable local transcription path yet — that's the first
work on the big box. The seam is confirmed in `PLAN.md`: implement two functions
(`get_transcription`, `create_diarization`) to the documented shapes and everything
downstream works unchanged.
