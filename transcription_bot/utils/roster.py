"""Roster loader — the SGU cast/guest identity registry (issue #10, feeds #5).

Loads ``transcription_bot/data/roster.toml`` into typed :class:`Member` objects,
resolves spoken names/aliases to a stable member id, and owns the reference
*voiceprint* embeddings used for local speaker naming.

Diarization emits anonymous clusters (``SPEAKER_00``..). Local speaker naming (#5)
cosine-matches each cluster's voice embedding against members' enrolled reference
embeddings and relabels the cluster with the member's ``display_name``. This module
owns the roster + embedding I/O; the matching itself lives with the diarization
backend.

Embeddings are stored one ``.npy`` per member id under the voiceprint directory
(``config.voiceprint_dir`` if set, else ``data/voiceprints/``). A member is
"enrolled" once such a file exists; until then the cluster stays ``SPEAKER_NN`` for
human review. Enrollment is retrainable: append a labelled sample and recompute the
member's centroid (see :func:`enroll_embedding`).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

from transcription_bot.utils.config import DATA_FOLDER, config

ROSTER_FILE = DATA_FOLDER / "roster.toml"
_DEFAULT_VOICEPRINT_DIR = DATA_FOLDER / "voiceprints"


@dataclass(frozen=True)
class Member:
    """One person on the show — host, rogue, or guest — from ``roster.toml``."""

    id: str
    display_name: str
    full_name: str = ""
    role: str = "rogue"
    status: str = "current"
    aliases: tuple[str, ...] = ()
    bio: str = ""
    active_from: str | None = None
    active_to: str | None = None
    active_from_episode: int | None = None
    active_to_episode: int | None = None
    guest_episodes: tuple[int, ...] = ()
    reference_audio: tuple[str, ...] = ()
    embedding: str = ""  # optional explicit path; else <voiceprint_dir>/<id>.npy

    def _names(self) -> set[str]:
        return {n.casefold() for n in (self.display_name, self.full_name, *self.aliases) if n}

    def matches(self, spoken_name: str) -> bool:
        """True if ``spoken_name`` equals the display/full name or a known alias."""
        return spoken_name.casefold().strip() in self._names()

    def embedding_path(self) -> Path:
        """Where this member's reference voiceprint lives on disk."""
        if self.embedding:
            p = Path(self.embedding)
            return p if p.is_absolute() else voiceprint_dir() / p
        return voiceprint_dir() / f"{self.id}.npy"

    def load_embedding(self) -> np.ndarray | None:
        """Load the member's reference embedding, or None if not yet enrolled."""
        path = self.embedding_path()
        if not path.exists():
            return None
        return np.load(path)


@dataclass(frozen=True)
class Roster:
    """The full cast + guests, with id/name lookup and enrollment queries."""

    members: tuple[Member, ...]

    def by_id(self, member_id: str) -> Member | None:
        return next((m for m in self.members if m.id == member_id), None)

    def resolve(self, spoken_name: str) -> Member | None:
        """Map a spoken/written name ("Steven Novella") to its Member, or None."""
        return next((m for m in self.members if m.matches(spoken_name)), None)

    def enrolled(self) -> list[Member]:
        """Members that have a reference embedding on disk (usable for naming)."""
        return [m for m in self.members if m.embedding_path().exists()]

    def active_for_episode(self, episode: int) -> list[Member]:
        """Members plausibly on a given episode (by episode-number bounds).

        Best-effort: only filters on ``active_from_episode`` / ``active_to_episode``
        when present, so members without episode bounds are always included.
        """
        out = []
        for m in self.members:
            if m.active_from_episode is not None and episode < m.active_from_episode:
                continue
            if m.active_to_episode is not None and episode > m.active_to_episode:
                continue
            out.append(m)
        return out


def voiceprint_dir() -> Path:
    """Directory holding reference voiceprint ``.npy`` files."""
    configured = getattr(config, "voiceprint_dir", "") or ""
    return Path(configured) if configured else _DEFAULT_VOICEPRINT_DIR


def _coerce_member(raw: dict) -> Member:
    def as_tuple(key: str) -> tuple:
        val = raw.get(key) or []
        return tuple(val)

    return Member(
        id=raw["id"],
        display_name=raw["display_name"],
        full_name=raw.get("full_name", ""),
        role=raw.get("role", "rogue"),
        status=raw.get("status", "current"),
        aliases=as_tuple("aliases"),
        bio=raw.get("bio", ""),
        active_from=raw.get("active_from"),
        active_to=raw.get("active_to"),
        active_from_episode=raw.get("active_from_episode"),
        active_to_episode=raw.get("active_to_episode"),
        guest_episodes=as_tuple("guest_episodes"),
        reference_audio=as_tuple("reference_audio"),
        embedding=raw.get("embedding", ""),
    )


@cache
def load_roster(path: Path | None = None) -> Roster:
    """Load and validate ``roster.toml`` into a :class:`Roster` (memoized)."""
    roster_path = path or ROSTER_FILE
    with roster_path.open("rb") as fh:
        data = tomllib.load(fh)

    members = tuple(_coerce_member(m) for m in data.get("member", []))

    ids = [m.id for m in members]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"Duplicate member ids in {roster_path}: {sorted(dupes)}")

    return Roster(members=members)


def enroll_embedding(member_id: str, sample: np.ndarray) -> np.ndarray:
    """Fold a new labelled voice sample into a member's reference centroid.

    Retrainable enrollment: averages the incoming ``sample`` with any existing
    reference, L2-normalizes, and persists it. Returns the stored centroid. This is
    how a human "this cluster is Steve" correction becomes a durable reference.
    """
    member = load_roster().by_id(member_id)
    if member is None:
        raise ValueError(f"Unknown member id: {member_id!r}")

    sample = _l2_normalize(np.asarray(sample, dtype=np.float32))
    existing = member.load_embedding()
    centroid = sample if existing is None else _l2_normalize(existing + sample)

    path = member.embedding_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, centroid)
    return centroid


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm
