"""Local LLM layer (Ollama) for segmentation — free/local, traced by Langfuse (#6/#12).

The host verbally announces most SGU transitions ("it's time for Science or Fiction",
"who's that noisy time"). This detects those announcements in the transcript timestream
and returns structured segments with the timestamp of the announcing line.

Ollama is called through its OpenAI-compatible endpoint (`/v1`). When Langfuse is
configured (env keys + a reachable host), the `langfuse.openai` drop-in auto-traces every
call; otherwise it degrades to the plain OpenAI client with no tracing. This is a v1
single-pass windowed segmenter — the agentic/ensemble refinement is #12.
"""

import json

from loguru import logger

from transcription_bot.utils.config import config

_SEGMENT_SYSTEM = (
    "You find where segments START in a transcript of The Skeptics' Guide to the Universe. "
    "Each input line is `[H:MM:SS] Speaker: text`. The host announces most transitions in "
    "natural language (\"let's move on to the news\", \"it's time for Science or Fiction\", "
    "\"who's that noisy time\", \"we have an interview with...\"). Scan every line and emit a "
    "segment for each announcing line you find.\n\n"
    "Rules:\n"
    "- Copy `start_timestamp` EXACTLY from the [bracket] of the announcing line. Never invent a time.\n"
    "- Only mark a line that actually signals a transition. If none, return an empty list.\n"
    "- Segment types: intro, news item, who's that noisy, science or fiction, interview, "
    "emails, quote, outro, other.\n\n"
    "Example input line:\n"
    "[1:49:53] Steve: All right, guys, it's science or fiction time.\n"
    "Example output:\n"
    "{\"segments\": [{\"start_timestamp\": \"1:49:53\", \"type\": \"science or fiction\", "
    "\"title\": \"Science or Fiction\", \"evidence\": \"it's science or fiction time\"}]}\n\n"
    "Return ONLY JSON of that shape."
)


def _client():  # noqa: ANN202
    """OpenAI-compatible client pointed at Ollama; Langfuse-traced when configured."""
    try:
        from langfuse.openai import OpenAI  # noqa: PLC0415 - drop-in tracer
    except Exception:  # noqa: BLE001 - Langfuse optional
        from openai import OpenAI  # noqa: PLC0415

    return OpenAI(base_url=config.ollama_base_url.rstrip("/") + "/v1", api_key="ollama")


def _fmt_ts(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}"


def segment_window(lines: list[str], model: str | None = None) -> list[dict]:
    """Ask the LLM for segment boundaries in one excerpt (list of '[ts] Speaker: text')."""
    model = model or config.ollama_model
    response = _client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SEGMENT_SYSTEM},
            {"role": "user", "content": "\n".join(lines)},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = response.choices[0].message.content or "{}"
    try:
        return json.loads(content).get("segments", [])
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON segmentation for a window; skipping it.")
        return []


def segment_episode(episode_number: int, model: str | None = None, window_minutes: int = 30) -> list[dict]:
    """Segment an episode from its stored timestream, windowing by wall-clock time.

    Reads `transcript_segments` from the metadata store, splits into `window_minutes`
    buckets, and asks the LLM for the announced boundaries in each. Returns the merged,
    time-ordered list of segments. (v1 single-pass; ensemble/coverage is #12.)
    """
    from transcription_bot.metadata import store  # noqa: PLC0415

    segments = store.get_transcript_segments(episode_number)
    if not segments:
        logger.warning(f"No stored transcript for episode {episode_number}; index it first.")
        return []

    model = model or config.ollama_model
    window_s = window_minutes * 60
    windows: dict[int, list[dict]] = {}
    for seg in segments:
        windows.setdefault(int(seg["start"] // window_s), []).append(seg)

    logger.info(f"Segmenting episode {episode_number}: {len(segments)} lines, {len(windows)} window(s), model={model}.")
    found: list[dict] = []
    for bucket in sorted(windows):
        lines = [f"[{_fmt_ts(s['start'])}] {s['speaker']}: {s['text']}" for s in windows[bucket]]
        found.extend(segment_window(lines, model))

    found.sort(key=lambda x: x.get("start_timestamp", ""))
    logger.success(f"Segmentation found {len(found)} boundary/segment(s) for episode {episode_number}.")
    return found
