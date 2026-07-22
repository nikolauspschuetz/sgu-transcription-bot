"""Canonical metadata/identity layer (Postgres + pgvector) — issue #13.

Formalizes the file-based bootstrap (roster.toml + data/voiceprints/*.npy) into a
queryable store: a `members` registry plus `voiceprints` (embedding vectors keyed by
member id) that re-load per episode to jump-start speaker naming, with matching done
in-DB via pgvector cosine search.
"""

from transcription_bot.metadata import store

__all__ = ["store"]
