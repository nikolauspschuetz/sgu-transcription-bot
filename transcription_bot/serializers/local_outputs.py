"""Local, publish-free output artifacts for review: WebVTT + Markdown.

These serialize the flat diarized transcript (``list[{start, end, text, speaker}]``),
so they work straight off ASR + diarization without needing segmentation or any LLM —
ideal for eyeballing quality before anything is published (issue #7, output-first).

- **WebVTT** is the canonical machine artifact (speaker labels via ``<v Name>``); it is
  also what a Podcasting 2.0 ``<podcast:transcript>`` feed tag would point at.
- **Markdown** is the human review copy: consecutive same-speaker chunks are merged into
  readable turns, with optional segment headings when segments are available.

Nothing here writes to the wiki. See ``entrypoints/create_local_transcript.py``.
"""

import re
from pathlib import Path

from transcription_bot.models.data_models import PodcastRssEntry
from transcription_bot.models.episode_segments.base import BaseSegment
from transcription_bot.models.simple_models import DiarizedTranscript

_DRAFT_NOTICE = (
    "_Auto-generated draft (local Whisper + pyannote). "
    "Speakers may be misidentified; timings approximate. Needs human review._"
)


# --------------------------------------------------------------------------- #
# Timestamp + text helpers
# --------------------------------------------------------------------------- #
def _format_vtt_timestamp(seconds: float) -> str:
    """Format seconds as WebVTT ``HH:MM:SS.mmm``."""
    total_ms = max(0, int(round(seconds * 1000)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _format_md_timestamp(seconds: float) -> str:
    """Format seconds as ``h:mm:ss`` (hour omitted when zero)."""
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _clean_cue_text(text: str) -> str:
    """Sanitize transcript text for a VTT cue body."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = text.replace("-->", "→")
    return " ".join(text.split())  # collapse whitespace / newlines


def _clean_speaker_for_vtt(speaker: str) -> str:
    """Sanitize a speaker label for use inside a ``<v ...>`` voice tag."""
    return speaker.replace(">", "").strip() or "Unknown"


# --------------------------------------------------------------------------- #
# WebVTT
# --------------------------------------------------------------------------- #
def serialize_transcript_to_vtt(transcript: DiarizedTranscript) -> str:
    """Render the diarized transcript as a WebVTT document with speaker voice tags."""
    lines = ["WEBVTT", ""]

    for index, chunk in enumerate(transcript, start=1):
        text = _clean_cue_text(chunk["text"])
        if not text:
            continue
        start = _format_vtt_timestamp(chunk["start"])
        end = _format_vtt_timestamp(chunk["end"])
        speaker = _clean_speaker_for_vtt(chunk["speaker"])
        lines.append(str(index))
        lines.append(f"{start} --> {end}")
        lines.append(f"<v {speaker}>{text}")
        lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def _segment_heading(segment: BaseSegment) -> str:
    """Best-effort human label for a segment (class name, prettified, + title if any)."""
    name = segment.__class__.__name__.removesuffix("Segment")
    label = re.sub(r"(?<!^)(?=[A-Z])", " ", name).strip() or name
    title = getattr(segment, "title", None)
    if title:
        label = f"{label}: {title}"
    return label


def _segment_boundaries(segments: "list[BaseSegment] | None") -> list[tuple[float, str]]:
    """Sorted ``(start_time, heading)`` pairs for segments that have a start time."""
    if not segments:
        return []
    boundaries = [(seg.start_time, _segment_heading(seg)) for seg in segments if seg.start_time is not None]
    return sorted(boundaries, key=lambda pair: pair[0])


def serialize_transcript_to_markdown(
    transcript: DiarizedTranscript,
    *,
    rss_entry: PodcastRssEntry | None = None,
    segments: "list[BaseSegment] | None" = None,
) -> str:
    """Render the diarized transcript as a readable Markdown review copy.

    Consecutive chunks from the same speaker are merged into one turn. If ``segments``
    are supplied, a heading is inserted whenever the transcript crosses a segment start.
    """
    lines: list[str] = []

    if rss_entry is not None:
        lines.append(f"# SGU Episode {rss_entry.episode_number} — {rss_entry.date.isoformat()}")
    else:
        lines.append("# SGU Episode transcript")
    lines.append("")
    lines.append(_DRAFT_NOTICE)
    lines.append("")

    boundaries = _segment_boundaries(segments)
    boundary_idx = 0

    for speaker, start, text in _group_into_turns(transcript):
        # Emit any segment headings we have now passed.
        while boundary_idx < len(boundaries) and boundaries[boundary_idx][0] <= start:
            lines.append(f"## {boundaries[boundary_idx][1]}")
            lines.append("")
            boundary_idx += 1

        lines.append(f"**{speaker}** ({_format_md_timestamp(start)}): {text}")
        lines.append("")

    return "\n".join(lines)


def _group_into_turns(transcript: DiarizedTranscript):
    """Yield ``(speaker, start_time, merged_text)`` merging consecutive same-speaker chunks."""
    speaker: str | None = None
    start = 0.0
    parts: list[str] = []

    for chunk in transcript:
        text = chunk["text"].strip()
        if not text:
            continue
        if chunk["speaker"] != speaker:
            if speaker is not None and parts:
                yield speaker, start, " ".join(parts)
            speaker = chunk["speaker"]
            start = chunk["start"]
            parts = [text]
        else:
            parts.append(text)

    if speaker is not None and parts:
        yield speaker, start, " ".join(parts)


# --------------------------------------------------------------------------- #
# Writer
# --------------------------------------------------------------------------- #
def write_transcript_outputs(
    transcript: DiarizedTranscript,
    rss_entry: PodcastRssEntry,
    *,
    segments: "list[BaseSegment] | None" = None,
    output_dir: Path | None = None,
) -> list[Path]:
    """Write ``<stem>.vtt`` and ``<stem>.md`` to ``output/`` (created if needed). No publish."""
    output_dir = output_dir or Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"SGU_Episode_{rss_entry.episode_number}"
    vtt_path = output_dir / f"{stem}.vtt"
    md_path = output_dir / f"{stem}.md"

    vtt_path.write_text(serialize_transcript_to_vtt(transcript), encoding="utf-8")
    md_path.write_text(
        serialize_transcript_to_markdown(transcript, rss_entry=rss_entry, segments=segments),
        encoding="utf-8",
    )

    return [vtt_path, md_path]
