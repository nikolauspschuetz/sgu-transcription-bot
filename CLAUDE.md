# transcription-bot (SGU) — Claude Code guide

Fan project: a **local, free/open-source** pipeline that transcribes *The Skeptics' Guide
to the Universe* and (optionally) publishes to the community wiki. Fork/modernization of
the archived `mheguy/transcription-bot`. **Not affiliated with SGU Productions, LLC.**

## Hard guardrail — free / local / open-source only
NO paid services in the pipeline: no Azure, pyannote.ai, OpenAI, or rented GPU. Every
paid step has a free/local switch via the `*_backend` config settings. If a step demands
a paid credential, you're on the wrong backend. (Building *products on top of* the
finished corpus is fine — the rule governs pipeline **dependencies**, not products.)

## Environment (Apple Silicon / M4)
- Python via `uv`. Deps: `uv sync --group local --group local-mlx`.
- `ffmpeg` required (Whisper + diarization WAV transcode). Export `PYTORCH_ENABLE_MPS_FALLBACK=1` for pyannote on `mps`.
- HuggingFace: `hf auth login` (cached token is honored; `TB_HF_TOKEN` optional). Accept the gated terms for the pyannote pipeline (`speaker-diarization-community-1`, `segmentation-3.0`, `wespeaker-voxceleb-resnet34-LM`).
- Config: `transcription_bot/data/config.toml`, overridden by `.env` (`TB_` prefix; gitignored). Profile B = Apple Silicon (MLX GPU ASR + pyannote on mps).

## Backends (config switches)
- `transcription_backend`: `local_mlx_whisper` (Apple GPU — use this) | `local_whisper` (faster-whisper, CPU) | `azure` (paid — don't).
- `diarization_backend`: `local_pyannote` (pyannote.audio 4.x) | `pyannote_ai` (paid — don't).
- `llm_backend`: `none` (heuristics) | `ollama` (local) | `openai` (paid — don't).

## Running (dev loop)
Publish-free transcript (RSS → ASR → diarization → merge → `output/`):
```
PYTORCH_ENABLE_MPS_FALLBACK=1 TB_MLX_WHISPER_REPO=mlx-community/whisper-medium.en-mlx \
  uv run python -m transcription_bot.entrypoints.create_local_transcript <EP>
```
- **Use `whisper-medium.en-mlx` for the dev loop** — `large-v3` is too slow on the M4 (>55 min/episode). Reserve large-v3 for a final quality pass.
- **Long runs must be detached** (`nohup ... & disown` → logfile), not foreground or `run_in_background` — long jobs otherwise get reaped mid-run. Then tail/Monitor the log.
- Caches (all under gitignored `/data/`): audio `data/audio/<ep>.mp3`, per-step `data/cache/`, cluster embeddings `data/embeddings/<ep>.npz`. Re-runs skip cached stages; delete a step's cache file to force re-run.
- Outputs (gitignored `/output/`): `SGU_Episode_<ep>.vtt` + `.md`.

## Architecture
- `entrypoints/create_local_transcript.py` — publish-free VTT/MD (no LLM, no wiki).
- `handlers/transcription_handler/_diarized_transcript.py` — backend dispatch + merge + the speaker-naming hook.
- `interfaces/local_mlx_whisper.py`, `interfaces/local_diarization.py` — the local engines.
- `serializers/local_outputs.py` (VTT/MD), `serializers/wiki.py` (MediaWiki markup).
- `utils/roster.py` + `data/roster.toml` — cast/guest identity registry for local speaker naming (#5).

## pyannote.audio 4.x notes (these bit us — don't re-discover)
- `Pipeline.from_pretrained(..., token=...)` — NOT `use_auth_token=` (removed in huggingface_hub 1.x).
- Diarization needs a **16 kHz mono WAV**; pyannote's strict `Audio.crop` rejects MP3 decode drift. `_ensure_wav16k()` transcodes.
- The pipeline returns a `DiarizeOutput` wrapper: use `.exclusive_speaker_diarization` for merging, `.speaker_embeddings` for #5.
- Default pipeline is `speaker-diarization-community-1` (4.x-native; better turn boundaries than 3.1).

## Speaker naming (#5)
Diarization emits `SPEAKER_00..`. `utils/roster.py` + `local_diarization.name_speakers()`
cosine-match cluster embeddings against enrolled member voiceprints
(`data/voiceprints/<id>.npy`) and relabel via `display_name`. It's a **no-op until a
voice is enrolled**. Enroll from a self-identifying cluster
(`roster.enroll_embedding(id, vec)`); each human correction becomes a durable reference.

## Wiki (publish target)
`sgutranscripts.org` is MediaWiki 1.43; "version control" is its native page-revision
history. A `transcription-bot` already seeds pages; a `Sts.` status model
(open→machine→bot→incomplete→proofread→verified) + `{{Editing required}}` flags encode
the review workflow. Publish is **dry-run by default** and gated by human review (#8) —
talk to wiki maintainers before any real publish.

## Conventions
- Commit only when asked. Never commit `/data/` or `/output/` (gitignored).
- Human-in-the-loop, output-first: prove quality locally before wiring any publish.
- `docs/` = show/members/segments reference. `HANDOFF.md` (untracked) = on-box runbook.
