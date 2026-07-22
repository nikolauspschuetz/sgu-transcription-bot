# SGU Transcriber — supporting docs

Reference material about **The Skeptics' Guide to the Universe (SGU)** — the show,
its cast, and its recurring segments — used to inform transcription, speaker
naming, and wiki templating.

> **Fan project.** Not affiliated with SGU Productions, LLC. Sourced from public
> pages (the show's official site, Wikipedia, sgutranscripts.org). Free / local /
> open-source only — no paid services.

## Contents

- [`show.md`](show.md) — what the show is, format, cadence, feeds.
- [`members.md`](members.md) — the cast ("rogues") and their real identities, current and former.
- [`segments.md`](segments.md) — recurring segments and how they map to the pipeline's templates.

## Machine-readable config

- [`../transcription_bot/data/roster.toml`](../transcription_bot/data/roster.toml) —
  the identity roster that drives local speaker naming (issue #5). Every host,
  rogue, and guest is one `[[member]]` block with a stable `id`, real name,
  aliases, and a slot for a reference voice embedding. Add a guest by copying the
  template block; nothing else in the pipeline changes.

_Last researched: 2026-07-21._
