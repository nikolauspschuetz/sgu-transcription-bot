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
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from loguru import logger

from transcription_bot.interfaces._audio import get_episode_audio_path
from transcription_bot.models.data_models import PodcastRssEntry
from transcription_bot.utils.caching import cache_for_episode
from transcription_bot.utils.config import config


def _ensure_wav16k(audio_path: Path) -> Path:
    """Decode the episode audio to a clean 16 kHz mono WAV for pyannote.

    pyannote's ``Audio.crop`` does a strict sample-count check, and MP3 decoding drifts
    by a few hundred samples per chunk (the decoder returns a slightly off length),
    raising ``ValueError: ... resulted in N samples instead of the expected M``. A WAV
    has exact sample counts, so we transcode once (cached beside the source) and hand
    pyannote the WAV. 16 kHz mono is what the segmentation/embedding models use anyway.

    We shell out to ``ffmpeg`` (already a hard requirement of the ASR backend) rather
    than add a decode dependency. The write is atomic (temp file + rename) so an
    interrupted transcode never leaves a truncated WAV that a later run would treat as a
    valid cache, and ffmpeg's stderr is surfaced on failure.
    """
    audio_path = Path(audio_path)
    wav_path = audio_path.with_suffix(".16k.wav")
    if wav_path.exists() and wav_path.stat().st_size > 0:
        return wav_path

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found; it is required for diarization.")

    logger.info(f"Transcoding {audio_path.name} -> 16 kHz mono WAV for diarization...")
    fd, tmp = tempfile.mkstemp(suffix=".wav", dir=str(wav_path.parent))
    os.close(fd)
    try:
        result = subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-i", str(audio_path),
             "-ac", "1", "-ar", "16000", "-vn", "-f", "wav", tmp],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed to transcode {audio_path.name} (exit {result.returncode}):\n"
                f"{result.stderr.strip()[-1500:]}"
            )
        if os.path.getsize(tmp) == 0:
            raise RuntimeError(f"ffmpeg produced an empty WAV for {audio_path.name}.")
        os.replace(tmp, wav_path)  # atomic: no partial file is ever seen as cache
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    return wav_path


def _resolve_hf_token() -> str:
    """Resolve the HuggingFace token from TB_HF_TOKEN or a cached ``hf auth login``.

    Two valid sources: the ``TB_HF_TOKEN`` env var, or the token cached by
    ``hf auth login`` at ``~/.cache/huggingface/token`` (auto-refreshing). Raise a clear
    error if neither is present, rather than failing deep inside pyannote.
    """
    if config.hf_token:
        return config.hf_token

    from huggingface_hub import get_token  # noqa: PLC0415

    cached = get_token()
    if cached:
        logger.info("Using HuggingFace token from `hf auth login` cache (TB_HF_TOKEN unset).")
        return cached

    raise RuntimeError(
        "No HuggingFace token found. Set TB_HF_TOKEN in .env, or run `hf auth login`. "
        "The token is free; you must also accept the terms for "
        "'pyannote/speaker-diarization-3.1' on HuggingFace."
    )


@functools.cache
def _get_pipeline(pipeline_name: str, hf_token: str, device: str):  # noqa: ANN202
    """Load (and memoize) the pyannote pipeline. Heavy import is deferred to call time."""
    import torch  # noqa: PLC0415
    from pyannote.audio import Pipeline  # noqa: PLC0415

    logger.info(f"Loading pyannote pipeline (name={pipeline_name}, device={device})...")
    # pyannote.audio 4.x / huggingface_hub 1.x renamed this kwarg: use_auth_token -> token.
    pipeline = Pipeline.from_pretrained(pipeline_name, token=hf_token or None)
    if pipeline is None:
        raise RuntimeError(
            f"Failed to load pyannote pipeline '{pipeline_name}'. "
            "Check TB_HF_TOKEN and that you've accepted the model's terms on HuggingFace."
        )

    return pipeline.to(torch.device(device))


@cache_for_episode(should_cache=lambda x: x is not None)
def create_diarization(rss_entry: PodcastRssEntry) -> pd.DataFrame | None:
    """Create a diarization DataFrame for the episode using local pyannote.audio."""
    audio_path = _ensure_wav16k(get_episode_audio_path(rss_entry))
    pipeline = _get_pipeline(config.diarization_pipeline, _resolve_hf_token(), config.diarization_device)

    logger.info(f"Diarizing episode {rss_entry.episode_number} with local pyannote...")
    result = pipeline(str(audio_path))

    # pyannote.audio 4.x pipelines (e.g. community-1) return a ``DiarizeOutput`` wrapper;
    # 3.x pipelines return an ``Annotation`` directly. Support both. For merging with the
    # transcript we prefer the exclusive (non-overlapping) diarization, which pyannote
    # documents as "adapted to downstream transcription".
    if hasattr(result, "itertracks"):
        annotation = result
    else:
        annotation = getattr(result, "exclusive_speaker_diarization", None) or result.speaker_diarization

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
