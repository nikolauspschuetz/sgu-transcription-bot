---
name: transcribe-episode
description: Run the local publish-free transcription pipeline for an SGU episode (RSS → MLX Whisper → pyannote → VTT/MD). Use when asked to transcribe an episode number, (re)generate a transcript, or produce output/SGU_Episode_<N>.{vtt,md}.
---

# Transcribe an SGU episode (local, publish-free)

Produces `output/SGU_Episode_<EP>.{vtt,md}` via RSS → MLX Whisper → pyannote → merge. No
LLM, no wiki. See CLAUDE.md for environment/guardrails.

## Steps
1. **Preflight:** `.env` present (Profile B), `hf auth login` done + gated pyannote terms
   accepted, `ffmpeg` on PATH. Confirm the episode exists in the RSS feed.
2. **Launch detached** (long runs get reaped otherwise), medium.en for the dev loop:
   ```
   LOG=<scratchpad>/run_<EP>.log; : > "$LOG"
   PYTORCH_ENABLE_MPS_FALLBACK=1 TB_MLX_WHISPER_REPO=mlx-community/whisper-medium.en-mlx \
     nohup uv run python -m transcription_bot.entrypoints.create_local_transcript <EP> >> "$LOG" 2>&1 &
   disown
   ```
3. **Monitor** the log with the Monitor tool, filtering for milestones AND failures:
   `Transcribing episode`, `MLX transcription complete`, `Diarizing`,
   `Local diarization complete`, `Wrote ...` — and `Traceback`, `GatedRepoError`,
   `MemoryError`, `Killed`. Silence is not success — cover the failure signatures too.
4. **First-run costs** (both cache to disk): audio download (~3 min) + model weights.
   Re-runs skip cached stages. Memory holds fine on 24 GB (~40-60% free).
5. **Inspect** `output/SGU_Episode_<EP>.vtt` — speaker labels are `<v ...>`.

## Gotchas
- `large-v3` is too slow on the M4 (>55 min) — use `medium.en` for iteration.
- Diarization errors → see the pyannote 4.x notes in CLAUDE.md (WAV transcode, `token=`,
  `DiarizeOutput`). A `GatedRepoError` means the HF account hasn't accepted a gated
  pipeline component's terms.
- Speakers are `SPEAKER_NN` until voices are enrolled — see the `enroll-voice` skill (#5).
