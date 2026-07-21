"""Create local review artifacts (WebVTT + Markdown) for an episode WITHOUT publishing.

Transcript-only path:
    RSS -> local Whisper -> local pyannote -> merge -> write output/{vtt,md}

No segmentation, no LLM, no wiki — so it runs fully free/local and is the right way to
eyeball ASR + diarization quality on a first run. Usage:

    uv run python -m transcription_bot.entrypoints.create_local_transcript 1096
"""

import sys

from loguru import logger

from transcription_bot.handlers.transcription_handler import get_transcript
from transcription_bot.parsers.rss_feed import get_podcast_rss_entries
from transcription_bot.serializers.local_outputs import write_transcript_outputs
from transcription_bot.utils.config import config
from transcription_bot.utils.global_http_client import http_client
from transcription_bot.utils.helpers import run_main_safely


def main(*, selected_episode: int) -> None:
    """Build a diarized transcript for one episode and write local review artifacts."""
    if not selected_episode:
        raise ValueError("Provide an episode number, e.g. `... create_local_transcript 1096`.")

    config.validators.validate_all()

    logger.info("Getting episodes from RSS feed...")
    rss_entries = get_podcast_rss_entries(http_client)
    rss_entry = next((entry for entry in rss_entries if entry.episode_number == selected_episode), None)
    if rss_entry is None:
        raise ValueError(f"Episode {selected_episode} not found in the RSS feed.")

    logger.info(f"Building local transcript for episode #{rss_entry.episode_number}...")
    transcript = get_transcript(rss_entry)
    if not transcript:
        logger.error("No transcript produced (transcription or diarization returned nothing).")
        return

    paths = write_transcript_outputs(transcript, rss_entry)
    for path in paths:
        logger.success(f"Wrote {path}")


if __name__ == "__main__":
    _, *_args = sys.argv
    _episode = int(_args[0]) if _args else 0
    run_main_safely(main, selected_episode=_episode)
