# SGU Transcriber — modernization plan

A maintained, local-first successor to the archived
[`mheguy/transcription-bot`](https://github.com/mheguy/transcription-bot) (MIT), which
produced the AI first-draft transcripts on
[sgutranscripts.org](https://www.sgutranscripts.org) until it was **discontinued
2026-02-28** ("lack of interest from the podcast producers"). Its death is almost
certainly why recent episodes (1096, 1097 …) have no transcript.

This is a fan project. Not endorsed by or affiliated with SGU Productions, LLC.

## Goal

Ingest an episode's audio from the public RSS feed, produce an accurate,
speaker-labeled, segmented transcript, put a **human review gate** in front of
publishing, and emit standards-friendly artifacts (Markdown + WebVTT) — with an
optional path back to the sgutranscripts wiki once quality is trusted.

## What we keep vs. replace

The upstream pipeline is cleanly modular. Downstream of one function everything is
backend-agnostic, so our change is surgical.

| Stage | Upstream | Our plan |
|---|---|---|
| RSS → episode list | `parsers/rss_feed.py` | **keep** (public feed, MP3 enclosures) |
| **ASR (audio → words+times)** | `interfaces/azure.py` (Azure Speech, cloud) | **replace → local Whisper** (`faster-whisper` large-v3) |
| **Diarization (who speaks when)** | `interfaces/pyannote.py` (pyannote.ai API + ngrok webhook) | **replace → local `pyannote.audio`** |
| **Speaker ID (name the speakers)** | pyannote.ai voiceprints (proprietary blobs) | **replace → local embeddings + cosine match** to a retrainable reference set |
| Merge transcript+diarization | `handlers/transcription_handler/_diarized_transcript.py` | **keep** (backend-agnostic merge) |
| Segment gathering (show notes + mp3 lyrics) | `parsers/show_notes.py`, `parsers/lyrics.py` | **keep** |
| Segment transition detection | OpenAI GPT (`interfaces/llm_interface.py`) | **keep**, option to swap → Claude |
| Wiki markup / templates | `serializers/wiki.py`, `data/templates/` | **keep**, add Markdown + VTT serializers |
| Publish | `interfaces/wiki.py` (MediaWiki API) | **keep, gated** behind human review; default target = our own output first |

### The seam (the whole re-architecture in one paragraph)

`get_transcript(rss_entry)` returns a `DiarizedTranscript` = `list[{start, end, text,
speaker}]`. It is built from two functions we swap:

- `get_transcription(rss_entry) -> list[{start, end, text}]`  ← local Whisper
- `create_diarization(rss_entry) -> DataFrame[start, end, speaker]`  ← local pyannote

Match those two shapes and **every downstream stage works unchanged.**

## Key facts discovered

- Public audio feed: `https://feed.theskepticsguide.org/feed/rss.aspx?feed=sgu`
- Wiki is MediaWiki with an API (`https://sgutranscripts.org/w/api.php`); bot pages
  auto-carry a "created by transcription-bot" banner + `needs proofreading` notice.
- Roster in the frozen voiceprints: Bob, Cara, Evan, Jay, Steve. **Evan left (ep 1094)**
  and guest rogues now appear — so speaker ID must be **retrainable**, not frozen.
- Spotify is a dead end as a *source*: auto-transcripts are in-app only, no API/export.
  Standards path for distribution is the Podcasting 2.0 `<podcast:transcript>` RSS tag
  (host a VTT, reference it in the feed → Apple et al. render it). Not our call to make
  on SGU's feed, but VTT is the right canonical artifact regardless.

## Decisions (locked)

1. **Fork & modernize** the archived MIT repo (keep proven RSS/segment/wiki plumbing).
2. **Local Whisper + pyannote** for ASR/diarization (free, no per-minute cost, no
   webhook/ngrok dance). Needs a GPU for large-v3 at reasonable speed; HF token for
   pyannote model download.
3. **Our own output first** (Markdown/VTT + a review diff). Wiki auto-publish is a
   later, opt-in, human-gated step pending community/producer buy-in.

## Open questions / risks

- **Python 3.14** on this machine; torch/pyannote lag → pin project to **3.12** via uv.
- **ffmpeg** not installed → `brew install ffmpeg` (Whisper needs it).
- **GPU**: mac = Metal/MPS (faster-whisper via CTranslate2 is CPU/CUDA; on Apple Silicon
  consider `whisper.cpp` or MLX-whisper). Decide ASR runtime per hardware.
- **Speaker-ID cold start**: need labeled reference audio per rogue to build embeddings.
  Bootstrap from a couple of older episodes that already have verified transcripts.
- **Segmentation** still calls an LLM; keep OpenAI seam, evaluate Claude swap.

## Roadmap

- [x] Recon upstream, confirm the seam, fork to `local-backend` branch
- [ ] Pin Python 3.12, add `local` optional-dep group (faster-whisper, pyannote.audio, torch), install ffmpeg
- [ ] Implement `interfaces/local_whisper.py` → `get_transcription` (contract-compatible)
- [ ] Implement `interfaces/local_diarization.py` → `create_diarization` (contract-compatible)
- [ ] Add `TB_TRANSCRIPTION_BACKEND=local|azure` switch; make cloud creds required only for `azure`
- [ ] Build local voiceprint/embedding reference set + cosine speaker-ID
- [ ] Add Markdown + WebVTT serializers alongside the wiki serializer
- [ ] Add review gate: emit a draft + diff, require approval before any publish
- [ ] End-to-end dry run on **episode 1096** (the AI episode), review output
- [ ] (Later) wire gated MediaWiki publish; talk to wiki maintainers/producers
