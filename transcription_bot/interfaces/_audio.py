"""Shared episode-audio download used by the local ASR + diarization backends.

Downloads the episode MP3 once and caches it on disk so both faster-whisper (ASR)
and pyannote (diarization) read the same local file instead of fetching twice.
"""

from pathlib import Path

from loguru import logger

from transcription_bot.models.data_models import PodcastRssEntry
from transcription_bot.utils.global_http_client import http_client

# Sits alongside the cache dir (data/ is gitignored).
_AUDIO_DIR = Path("data/audio").resolve()

# Episodes are ~1-2h MP3s; override the client's short default timeout.
_DOWNLOAD_TIMEOUT = 600
_CHUNK_SIZE = 1 << 20  # 1 MiB


def get_episode_audio_path(rss_entry: PodcastRssEntry) -> Path:
    """Return a local path to the episode's MP3, downloading it once if needed."""
    _AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    dest = _AUDIO_DIR / f"{rss_entry.episode_number}.mp3"

    if dest.exists() and dest.stat().st_size > 0:
        logger.info(f"Using cached audio for episode {rss_entry.episode_number}: {dest}")
        return dest

    logger.info(f"Downloading audio for episode {rss_entry.episode_number}...")
    response = http_client.get(rss_entry.download_url, timeout=_DOWNLOAD_TIMEOUT, stream=True)

    # Write to a temp file, then rename, so an interrupted download never looks complete.
    partial = dest.with_name(dest.name + ".part")
    with partial.open("wb") as file:
        for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
            if chunk:
                file.write(chunk)
    partial.rename(dest)

    logger.success(f"Downloaded audio to {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest
