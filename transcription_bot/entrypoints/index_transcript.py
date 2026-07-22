"""Index an episode's diarized transcript into the searchable timestream (#13).

    uv run python -m transcription_bot.entrypoints.index_transcript 1096

Builds the diarized transcript (reusing cached ASR + diarization) and saves it to the
Postgres transcript_segments table — time-ordered and full-text searchable. Idempotent
per episode (replaces prior rows).
"""

import sys

from loguru import logger

from transcription_bot.handlers.transcription_handler import get_transcript
from transcription_bot.metadata import store
from transcription_bot.parsers.rss_feed import get_podcast_rss_entries
from transcription_bot.utils.global_http_client import http_client
from transcription_bot.utils.helpers import run_main_safely


def main(*, selected_episode: int) -> None:
    if not selected_episode:
        raise ValueError("Provide an episode number, e.g. `... index_transcript 1096`.")

    store.init_schema()

    rss_entry = next(
        (e for e in get_podcast_rss_entries(http_client) if e.episode_number == selected_episode), None
    )
    if rss_entry is None:
        raise ValueError(f"Episode {selected_episode} not found in the RSS feed.")

    transcript = get_transcript(rss_entry)
    if not transcript:
        logger.error("No transcript produced (transcription or diarization returned nothing).")
        return

    count = store.save_transcript(selected_episode, transcript)
    logger.success(f"Indexed {count} segments for episode {selected_episode} into the timestream.")


if __name__ == "__main__":
    _, *_args = sys.argv
    _episode = int(_args[0]) if _args else 0
    run_main_safely(main, selected_episode=_episode)
