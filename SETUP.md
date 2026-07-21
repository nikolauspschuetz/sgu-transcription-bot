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

### CUDA note

`uv sync --group local` pulls a default torch wheel. On a CUDA box, install the
matching build first so GPU is used, e.g.:

```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify: `uv run python -c "import torch; print(torch.cuda.is_available())"` → `True`.

## Configure

```bash
cp .env.template .env
```

Then edit `.env`:
- Set `TB_HF_TOKEN=...`
- **Dev Mac (CPU):** keep Profile A (`medium.en` / `cpu` / `int8`).
- **GPU box:** uncomment Profile B (`large-v3` / `cuda` / `float16`, optional Ollama).

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
