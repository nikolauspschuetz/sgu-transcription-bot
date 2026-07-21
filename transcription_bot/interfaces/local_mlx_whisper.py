"""Local ASR backend using MLX Whisper — runs on the Apple-Silicon GPU (Metal).

faster-whisper/CTranslate2 has no Metal backend, so on an M-series Mac it is CPU-only.
MLX Whisper uses the Apple GPU and is the fast path on the M4 mini. Same contract as
the other ASR backend:

    get_transcription(rss_entry) -> RawTranscript | None
    RawTranscript = list[{"start": float, "end": float, "text": str}]

Selected when ``config.transcription_backend`` is ``"local_mlx_whisper"``. The model is
an MLX-converted Whisper repo on HuggingFace (``config.mlx_whisper_repo``).

NOTE: Apple-Silicon only (the ``local-mlx`` dep group). Untested end-to-end pending an
online M-series box. See SETUP.md and issue #2.
"""

from loguru import logger

from transcription_bot.interfaces._audio import get_episode_audio_path
from transcription_bot.interfaces.local_whisper import _perform_low_level_text_corrections
from transcription_bot.models.data_models import PodcastRssEntry
from transcription_bot.models.simple_models import RawTranscript
from transcription_bot.utils.caching import cache_for_episode
from transcription_bot.utils.config import config


@cache_for_episode(should_cache=lambda x: x is not None)
def get_transcription(rss_entry: PodcastRssEntry) -> RawTranscript | None:
    """Transcribe an episode locally with MLX Whisper (Apple GPU)."""
    import mlx_whisper  # noqa: PLC0415

    audio_path = get_episode_audio_path(rss_entry)
    repo = config.mlx_whisper_repo

    logger.info(f"Transcribing episode {rss_entry.episode_number} with MLX Whisper ({repo})...")
    result = mlx_whisper.transcribe(str(audio_path), path_or_hf_repo=repo, word_timestamps=False)

    transcription: RawTranscript = []
    for seg in result.get("segments", []):
        text = _perform_low_level_text_corrections(seg["text"].strip())
        if not text:
            continue
        transcription.append({"start": float(seg["start"]), "end": float(seg["end"]), "text": text})

    if not transcription:
        logger.warning("MLX transcription produced no segments.")
        return None

    logger.success(f"MLX transcription complete: {len(transcription)} segments.")
    return transcription
