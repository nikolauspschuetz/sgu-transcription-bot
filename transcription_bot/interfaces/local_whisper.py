"""Local ASR backend using faster-whisper (free, offline).

Drop-in replacement for ``interfaces.azure.get_transcription``. It matches the exact
contract the pipeline expects:

    get_transcription(rss_entry) -> RawTranscript | None
    RawTranscript = list[{"start": float, "end": float, "text": str}]

so nothing downstream changes. Selected when ``config.transcription_backend`` is
``"local_whisper"`` (see ``handlers/transcription_handler/_diarized_transcript.py``).

Sizing (model / device / compute_type / beam_size) comes from config, so the same code
runs a small int8 model on a CPU dev box and large-v3 float16 on a CUDA box.

NOTE: not yet exercised end-to-end — needs the ``local`` dependency group + ffmpeg on a
machine that can run the model. See SETUP.md and issue #2.
"""

import functools

from loguru import logger

from transcription_bot.interfaces._audio import get_episode_audio_path
from transcription_bot.models.data_models import PodcastRssEntry
from transcription_bot.models.simple_models import RawTranscript
from transcription_bot.utils.caching import cache_for_episode
from transcription_bot.utils.config import config


@functools.cache
def _get_model(model_name: str, device: str, compute_type: str):  # noqa: ANN202
    """Load (and memoize) the Whisper model. Heavy import is deferred to call time."""
    from faster_whisper import WhisperModel  # noqa: PLC0415

    logger.info(f"Loading faster-whisper (model={model_name}, device={device}, compute_type={compute_type})...")
    return WhisperModel(model_name, device=device, compute_type=compute_type)


@cache_for_episode(should_cache=lambda x: x is not None)
def get_transcription(rss_entry: PodcastRssEntry) -> RawTranscript | None:
    """Transcribe an episode locally with faster-whisper."""
    audio_path = get_episode_audio_path(rss_entry)
    model = _get_model(config.whisper_model, config.whisper_device, config.whisper_compute_type)

    logger.info(f"Transcribing episode {rss_entry.episode_number} with local Whisper...")
    # `segments` is a generator; transcription actually runs as we consume it.
    segments, info = model.transcribe(
        str(audio_path),
        beam_size=int(config.whisper_beam_size),
        vad_filter=True,
        word_timestamps=False,
    )
    logger.info(f"Detected language {info.language!r} (probability {info.language_probability:.2f})")

    transcription: RawTranscript = []
    for seg in segments:
        text = _perform_low_level_text_corrections(seg.text.strip())
        if not text:
            continue
        transcription.append({"start": float(seg.start), "end": float(seg.end), "text": text})

    if not transcription:
        logger.warning("Local transcription produced no segments.")
        return None

    logger.success(f"Local transcription complete: {len(transcription)} segments.")
    return transcription


def _perform_low_level_text_corrections(text: str) -> str:
    """Match the Azure backend's low-level fixups (e.g. the recurring Kara/Cara mishear)."""
    return text.replace("Kara", "Cara")
