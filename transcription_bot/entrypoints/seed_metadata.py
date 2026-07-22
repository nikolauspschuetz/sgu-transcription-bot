"""Seed the metadata store from roster.toml + local voiceprints (#13 bootstrap).

    docker compose up -d          # start Postgres + pgvector
    uv sync --group db
    uv run python -m transcription_bot.entrypoints.seed_metadata

Idempotent: creates the schema, upserts the roster into `members`, and migrates any
file-based reference voiceprints (`data/voiceprints/<id>.npy`) into the `voiceprints`
table. Safe to re-run.
"""

from loguru import logger

from transcription_bot.metadata import store
from transcription_bot.utils.roster import load_roster


def main() -> None:
    store.init_schema()
    logger.info("Schema ready.")

    n_members = store.sync_roster()
    logger.success(f"Synced {n_members} members into the metadata store.")

    migrated = 0
    for member in load_roster().members:
        embedding = member.load_embedding()
        if embedding is not None:
            store.upsert_voiceprint(member.id, embedding)
            migrated += 1
            logger.info(f"Migrated voiceprint: {member.id}")
    logger.success(f"Migrated {migrated} file-based voiceprint(s) into pgvector.")


if __name__ == "__main__":
    main()
