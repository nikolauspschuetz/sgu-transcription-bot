"""Local diarization backend using pyannote.audio (free, offline).

Drop-in replacement for ``interfaces.pyannote.create_diarization``. It matches the
contract the merge step expects:

    create_diarization(rss_entry) -> pandas.DataFrame[start, end, speaker] | None

Selected when ``config.diarization_backend`` is ``"local_pyannote"`` (see
``handlers/transcription_handler/_diarized_transcript.py``).

Speakers come back as generic labels (``SPEAKER_00`` ...). Mapping those to the rogues'
names (Steve/Bob/Jay/Cara/...) is a separate, retrainable step — see issue #5. The
downstream merge already treats a leading ``SPEAKER_`` label as the voice-over, so
generic labels are safe to hand back.

NOTE: not yet exercised end-to-end — needs the ``local`` dependency group and a (free)
HuggingFace token that has accepted the ``pyannote/speaker-diarization-3.1`` terms.
See SETUP.md and issue #3.
"""

import functools

import pandas as pd
from loguru import logger

from transcription_bot.interfaces._audio import get_episode_audio_path
from transcription_bot.models.data_models import PodcastRssEntry
from transcription_bot.utils.caching import cache_for_episode
from transcription_bot.utils.config import config


@functools.cache
def _get_pipeline(pipeline_name: str, hf_token: str, device: str):  # noqa: ANN202
    """Load (and memoize) the pyannote pipeline. Heavy import is deferred to call time."""
    import torch  # noqa: PLC0415
    from pyannote.audio import Pipeline  # noqa: PLC0415

    logger.info(f"Loading pyannote pipeline (name={pipeline_name}, device={device})...")
    pipeline = Pipeline.from_pretrained(pipeline_name, use_auth_token=hf_token or None)
    if pipeline is None:
        raise RuntimeError(
            f"Failed to load pyannote pipeline '{pipeline_name}'. "
            "Check TB_HF_TOKEN and that you've accepted the model's terms on HuggingFace."
        )

    return pipeline.to(torch.device(device))


@cache_for_episode(should_cache=lambda x: x is not None)
def create_diarization(rss_entry: PodcastRssEntry) -> pd.DataFrame | None:
    """Create a diarization DataFrame for the episode using local pyannote.audio."""
    audio_path = get_episode_audio_path(rss_entry)
    pipeline = _get_pipeline(config.diarization_pipeline, config.hf_token, config.diarization_device)

    logger.info(f"Diarizing episode {rss_entry.episode_number} with local pyannote...")
    annotation = pipeline(str(audio_path))

    rows = [
        {"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]

    if not rows:
        logger.warning("Local diarization produced no speaker turns.")
        return None

    speaker_count = len({row["speaker"] for row in rows})
    logger.success(f"Local diarization complete: {len(rows)} turns across {speaker_count} speakers.")
    return pd.DataFrame(rows, columns=["start", "end", "speaker"])
