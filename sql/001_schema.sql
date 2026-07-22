-- SGU metadata store — canonical identity/attribution layer (#13).
--
-- Applied two ways from this single source of truth:
--   1. Automatically on first DB init, via the postgres image's
--      /docker-entrypoint-initdb.d hook (see docker-compose.yml).
--   2. Idempotently by metadata.store.init_schema() for existing / non-container DBs.
-- Keep every statement IF NOT EXISTS so both paths are safe to re-run.

CREATE EXTENSION IF NOT EXISTS vector;

-- Canonical registry of everyone who speaks on the show (seeded from roster.toml).
CREATE TABLE IF NOT EXISTS members (
    id           text PRIMARY KEY,
    display_name text NOT NULL,
    full_name    text,
    role         text,
    status       text,
    aliases      text[],
    bio          text
);

-- Reference voiceprints ("essential coefficients"), one per member, keyed by id.
-- 256-d = pyannote community-1 / wespeaker embedding. Matching = cosine (<=>).
CREATE TABLE IF NOT EXISTS voiceprints (
    member_id  text PRIMARY KEY REFERENCES members(id) ON DELETE CASCADE,
    embedding  vector(256) NOT NULL,
    n_samples  int NOT NULL DEFAULT 1,
    updated_at timestamptz NOT NULL DEFAULT now()
);
