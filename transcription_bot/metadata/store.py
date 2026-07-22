"""Metadata store — canonical identity/attribution layer (Postgres + pgvector).

The file-based roster (`roster.toml`) seeds this store once; thereafter the DB is the
source of truth. `members` is the canonical registry; `voiceprints` holds each member's
reference embedding (the "essential coefficients") as a pgvector column, keyed by member
id, so it re-loads per episode to jump-start speaker naming (#13). Speaker matching is an
in-DB cosine search — replacing the numpy `.npy` + hand-rolled cosine in
`interfaces/local_diarization.name_speakers()`.

Connection via `TB_DATABASE_URL` (default points at the docker-compose Postgres).
Embeddings are 256-d (pyannote community-1 / wespeaker). Vectors are L2-normalized on the
way in so pgvector's cosine distance (`<=>`) behaves like a plain dot product.
"""

import os
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

DATABASE_URL = os.environ.get("TB_DATABASE_URL", "postgresql://sgu:sgu@localhost:5432/sgu")
EMBED_DIM = 256

# DDL lives in versioned SQL files (repo-root ./sql), the single source of truth. They
# run automatically on first DB init via docker-entrypoint-initdb.d, and are applied
# idempotently here for existing / non-container databases.
_SQL_DIR = Path(__file__).resolve().parents[2] / "sql"


def connect() -> psycopg.Connection:
    """Open a pgvector-aware connection.

    Ensures the ``vector`` extension exists before registering the type — otherwise
    ``register_vector`` raises on a fresh database (the type wouldn't exist yet).
    """
    conn = psycopg.connect(DATABASE_URL)
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    return conn


def init_schema() -> None:
    """Apply the versioned DDL in ./sql (idempotent) — for existing/non-container DBs."""
    with connect() as conn, conn.cursor() as cur:
        for sql_file in sorted(_SQL_DIR.glob("*.sql")):
            cur.execute(sql_file.read_text())
        conn.commit()


def sync_roster() -> int:
    """Upsert every roster.toml member into the `members` registry. Returns the count."""
    from transcription_bot.utils.roster import load_roster  # noqa: PLC0415

    members = load_roster().members
    with connect() as conn, conn.cursor() as cur:
        for m in members:
            cur.execute(
                """
                INSERT INTO members (id, display_name, full_name, role, status, aliases, bio)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    display_name = EXCLUDED.display_name,
                    full_name    = EXCLUDED.full_name,
                    role         = EXCLUDED.role,
                    status       = EXCLUDED.status,
                    aliases      = EXCLUDED.aliases,
                    bio          = EXCLUDED.bio
                """,
                (m.id, m.display_name, m.full_name, m.role, m.status, list(m.aliases), m.bio),
            )
        conn.commit()
    return len(members)


def upsert_voiceprint(member_id: str, embedding: np.ndarray, n_samples: int = 1) -> None:
    """Store/replace a member's reference voiceprint (L2-normalized)."""
    vec = _l2_normalize(np.asarray(embedding, dtype=np.float32))
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO voiceprints (member_id, embedding, n_samples)
            VALUES (%s, %s, %s)
            ON CONFLICT (member_id) DO UPDATE SET
                embedding = EXCLUDED.embedding, n_samples = EXCLUDED.n_samples, updated_at = now()
            """,
            (member_id, vec, n_samples),
        )
        conn.commit()


def match_speaker(embedding: np.ndarray, threshold: float = 0.5) -> dict | None:
    """Best voiceprint match for a cluster embedding via pgvector cosine search.

    Returns {"id", "display_name", "similarity"} above ``threshold``, else None.
    """
    vec = _l2_normalize(np.asarray(embedding, dtype=np.float32))
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT m.id, m.display_name, 1 - (v.embedding <=> %s) AS similarity
            FROM voiceprints v
            JOIN members m ON m.id = v.member_id
            ORDER BY v.embedding <=> %s
            LIMIT 1
            """,
            (vec, vec),
        )
        row = cur.fetchone()
    if row and row[2] >= threshold:
        return {"id": row[0], "display_name": row[1], "similarity": float(row[2])}
    return None


def save_transcript(episode_number: int, transcript: list[dict]) -> int:
    """Persist a diarized transcript as the episode's timestream segments (replace-per-episode).

    ``transcript`` is a list of {start, end, text, speaker}. Returns the row count.
    """
    rows = [
        (episode_number, seq, float(c["start"]), float(c["end"]), str(c["speaker"]), str(c["text"]))
        for seq, c in enumerate(transcript)
    ]
    with connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM transcript_segments WHERE episode_number = %s", (episode_number,))
        cur.executemany(
            """INSERT INTO transcript_segments (episode_number, seq, start_s, end_s, speaker, text)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            rows,
        )
        conn.commit()
    return len(rows)


def search_transcript(query: str, episode_number: int | None = None, limit: int = 10) -> list[dict]:
    """Full-text search across the transcript timestream (ranked). Optionally scope to one episode."""
    sql = """
        SELECT episode_number, start_s, end_s, speaker, text,
               ts_rank(tsv, websearch_to_tsquery('english', %s)) AS rank
        FROM transcript_segments
        WHERE tsv @@ websearch_to_tsquery('english', %s)
    """
    params: list = [query, query]
    if episode_number is not None:
        sql += " AND episode_number = %s"
        params.append(episode_number)
    sql += " ORDER BY rank DESC, episode_number, start_s LIMIT %s"
    params.append(limit)

    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [
        {"episode": r[0], "start": r[1], "end": r[2], "speaker": r[3], "text": r[4], "rank": float(r[5])}
        for r in rows
    ]


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec if norm == 0 else vec / norm
