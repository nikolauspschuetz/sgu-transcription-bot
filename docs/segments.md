# Recurring segments

SGU episodes are built from recurring segments. The pipeline already models these
as Jinja templates in
[`../transcription_bot/data/templates/`](../transcription_bot/data/templates/) —
this table maps each segment to its template so the docs and code stay in sync.

| Segment | What it is | Template |
|---------|-----------|----------|
| Intro | Cold open / episode header | `intro.j2x` |
| News Items | Discussion of recent science news | `news.j2x` |
| Who's That Noisy | Audio-clip guessing game | `noisy.j2x` |
| Science or Fiction | Host presents 3 items; panel picks the fake one | `science_or_fiction.j2x` |
| What's the Word | Etymology / meaning of a science term | `whats_the_word.j2x` |
| Quickie | Short single-topic science item | `quickie.j2x` |
| Interview | Guest interview | `interview.j2x` |
| Email / Questions | Listener mail and follow-ups | `email.j2x` |
| Logical Fallacy | Named-fallacy explainer | `logical_fallacy.j2x` |
| Dumbest Thing of the Week | Notably bad pseudoscience/news item | `dumbest.j2x` |
| From TikTok | Debunking a viral clip | `tiktok.j2x` |
| Skeptical Quote | Closing quotation read by a host | `quote.j2x` |
| Outro | Sign-off | `outro.j2x` |

Supporting templates: `base.j2x` (shared layout), `unknown.j2x` (unclassified
segment fallback), `episode_list_entry.j2x` (wiki episode-list row).

## Pipeline relevance

- The **VTT/MD dry-run path** (`create_local_transcript`) does raw ASR +
  diarization only — **no** segmentation, so these templates are not exercised.
- The **full-wiki path** uses an LLM (or heuristics with `llm_backend=none`) to
  detect segment transitions and fill these templates. Segment-transition
  detection, image captions, and Science-or-Fiction extraction are the parts
  that still want an LLM — see issue #6 (add `ollama` / heuristics-only paths).
