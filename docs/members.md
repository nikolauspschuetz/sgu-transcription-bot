# The Rogues — cast & identities

Real identities behind the transcript labels. The machine-readable version is
[`../transcription_bot/data/roster.toml`](../transcription_bot/data/roster.toml),
which is what the pipeline reads for local speaker naming (issue #5).

> Roster changes over time — treat this as a living document. Evan Bernstein
> resigned in June 2026; guest rogues appear regularly. Adding a voice should be
> a one-block edit to `roster.toml` + a reference clip.

## Current cast

| Label | Real name | Role | On the show since | Background |
|-------|-----------|------|-------------------|------------|
| **Steve** | Steven Novella | Host / founder | 2005 | Academic neurologist (Yale); president & co-founder of the NESS |
| **Bob** | Bob Novella | Co-host | 2005 | NESS co-founder / VP; astronomy, cosmology, particle physics |
| **Jay** | Jay Novella | Co-host | 2005 | Live-show producer; co-founder of NECSS |
| **Cara** | Cara Santa Maria | Co-host | July 2015 | Emmy-winning science journalist & communicator; former biology/psychology lecturer |

## Former cast

| Label | Real name | Role | Years | Departure |
|-------|-----------|------|-------|-----------|
| **Evan** | Evan Bernstein | Co-host | 2005 – June 2026 (~ep 1094) | Resigned |
| **Perry** | Perry DeAngelis | Co-host / co-founder | 2005 – 2007 | Died (2007) |
| **Rebecca** | Rebecca Watson | Panelist | 2006 – 2014 | Departed; founder of Skepchick |

## Guests

Guests (scientists, authors, musicians such as George Hrab, and rotating "guest
rogues") appear per-episode. They are modeled with the **same schema** as the
core cast — a `[[member]]` block with `role = "guest"` and the episodes they
appeared on. See the template at the bottom of `roster.toml`.

## Notes for speaker naming (#5)

- `display_name` is the transcript label; `full_name` + `aliases` help map spoken
  self-introductions ("I'm Steven Novella") back to the right `id`.
- The old bot's biggest weakness was speaker attribution. Human corrections on
  proposed transcripts (e.g. "this is Cara, not SPEAKER_02") double as labeled
  voice samples — feed them back into a member's `reference_audio` to improve
  future matches.

_Sources: theskepticsguide.org/about, en.wikipedia.org/wiki/The_Skeptics'_Guide_to_the_Universe. Researched 2026-07-21._
