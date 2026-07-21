import numpy as np
import pandas as pd
from loguru import logger

from transcription_bot.models.data_models import PodcastRssEntry
from transcription_bot.models.simple_models import DiarizedTranscript, DiarizedTranscriptChunk, RawTranscript
from transcription_bot.utils.config import config


def _get_transcription(rss_entry: PodcastRssEntry) -> RawTranscript | None:
    """Dispatch to the configured ASR backend. Heavy deps import lazily, per backend."""
    backend = config.transcription_backend
    if backend == "local_whisper":
        from transcription_bot.interfaces.local_whisper import get_transcription  # noqa: PLC0415
    elif backend == "local_mlx_whisper":
        from transcription_bot.interfaces.local_mlx_whisper import get_transcription  # noqa: PLC0415
    elif backend == "azure":
        from transcription_bot.interfaces.azure import get_transcription  # noqa: PLC0415
    else:
        raise ValueError(f"Unknown transcription_backend: {backend!r}")
    return get_transcription(rss_entry)


def _create_diarization(rss_entry: PodcastRssEntry) -> "pd.DataFrame | None":
    """Dispatch to the configured diarization backend. Heavy deps import lazily, per backend."""
    backend = config.diarization_backend
    if backend == "local_pyannote":
        from transcription_bot.interfaces.local_diarization import create_diarization  # noqa: PLC0415
    elif backend == "pyannote_ai":
        from transcription_bot.interfaces.pyannote import create_diarization  # noqa: PLC0415
    else:
        raise ValueError(f"Unknown diarization_backend: {backend!r}")
    return create_diarization(rss_entry)


def get_diarized_transcript(rss_entry: PodcastRssEntry) -> DiarizedTranscript | None:
    """Create a transcript with the audio and podcast information."""
    logger.info("Getting diarized transcript...")

    transcription = _get_transcription(rss_entry)
    diarization = _create_diarization(rss_entry)

    if transcription is None:
        return None

    if diarization is None:
        return None

    return merge_transcript_and_diarization(transcription, diarization)


def merge_transcript_and_diarization(transcription: RawTranscript, diarization: "pd.DataFrame") -> DiarizedTranscript:
    logger.info("Merging transcript and diarization...")
    diarized_transcript: DiarizedTranscript = []

    for seg in transcription:
        if not seg["text"]:
            continue

        # Find active speakers during the segment
        diarization["intersection"] = np.minimum(diarization["end"], seg["end"]) - np.maximum(
            diarization["start"], seg["start"]
        )
        segment_speakers = diarization[diarization["intersection"] > 0]

        if len(segment_speakers) > 0:
            # Select the most active speaker
            segment_speaker = (
                segment_speakers.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]  # pyright: ignore[reportCallIssue]
            )

            if not isinstance(segment_speaker, str):
                raise TypeError(f"Unexpected speaker type: {type(segment_speaker)}")
        else:
            segment_speaker = "UNKNOWN"

        diarized_transcript.append(
            DiarizedTranscriptChunk(start=seg["start"], end=seg["end"], text=seg["text"], speaker=segment_speaker)
        )

    adjust_transcript_for_voiceover(diarized_transcript)

    return diarized_transcript


def adjust_transcript_for_voiceover(complete_transcript: DiarizedTranscript) -> None:
    """Adjust the transcript for voiceover."""
    voiceover = complete_transcript[0]["speaker"]

    if "SPEAKER_" not in voiceover:
        return

    for chunk in complete_transcript:
        if chunk["speaker"] == voiceover:
            chunk["speaker"] = "Voice-over"
