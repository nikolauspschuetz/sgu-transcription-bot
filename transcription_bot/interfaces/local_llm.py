"""Local LLM layer (Ollama) for segmentation — free/local, traced by Langfuse (#6/#12).

The host verbally announces most SGU transitions ("it's time for Science or Fiction",
"who's that noisy time"). This detects those announcements in the transcript timestream
and returns structured segments with the timestamp of the announcing line.

Ollama is called through its OpenAI-compatible endpoint (`/v1`). When Langfuse is
configured (env keys + a reachable host), the `langfuse.openai` drop-in auto-traces every
call; otherwise it degrades to the plain OpenAI client with no tracing. This is a v1
single-pass windowed segmenter — the agentic/ensemble refinement is #12.
"""

import contextlib
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


def segment_episode(
    episode_number: int,
    model: str | None = None,
    window_minutes: int = 15,
    max_windows: int | None = None,
    use_dspy: bool = False,
) -> list[dict]:
    """Segment an episode into announced transitions, traced by Langfuse (#18/#19).

    Reads the stored timestream, windows by wall-clock time, detects host-announced
    transitions per window, and **resolves each to a REAL transcript line so the timestamp
    is deterministic** (the LLM never supplies the time). Wrapped in ONE Langfuse trace per
    episode (session = ``episode-<n>``) with a nested generation per window and a
    ``cue_recall`` quality score vs the announced-transition ground truth.

    Engine: default is the direct JSON-prompt path (fast, clean, auto-traced). ``use_dspy``
    switches to the compiled DSPy program — correct but impractical with local *reasoning*
    models (they bury the answer in a reasoning channel DSPy can't parse and run minutes per
    call); use it once a JSON-clean model is available.
    """
    from transcription_bot.interfaces import dspy_segmenter  # noqa: PLC0415
    from transcription_bot.metadata import store  # noqa: PLC0415

    model = model or config.ollama_model
    segments = store.get_transcript_segments(episode_number)
    if not segments:
        logger.warning(f"No stored transcript for episode {episode_number}; index it first.")
        return []

    windows = _windows(segments, window_minutes * 60)
    if max_windows:
        windows = windows[:max_windows]
    module = dspy_segmenter.load_segmenter(model) if use_dspy else None
    logger.info(
        f"Segmenting episode {episode_number}: {len(segments)} lines, {len(windows)} window(s), "
        f"model={model}, engine={'dspy' if use_dspy else 'json-prompt'}."
    )

    lf = _langfuse_client()
    found: list[dict] = []
    recall = 1.0
    with contextlib.ExitStack() as stack:
        if lf is not None:
            from langfuse import propagate_attributes  # noqa: PLC0415

            stack.enter_context(
                propagate_attributes(session_id=f"episode-{episode_number}", tags=["segmentation", model, "dspy"])
            )
            stack.enter_context(
                lf.start_as_current_observation(
                    name=f"segment_episode:{episode_number}",
                    as_type="span",
                    input={"episode": episode_number, "windows": len(windows), "model": model},
                )
            )

        for window in windows:
            found.extend(_detect_in_window(lf, window, module, model, use_dspy))

        ground_truth = _cue_ground_truth(segments)
        recovered = _recovered_count(ground_truth, found)
        recall = recovered / len(ground_truth) if ground_truth else 1.0
        if lf is not None:
            try:
                lf.update_current_span(output={"segment_count": len(found), "segments": found[:20]})
                lf.score_current_trace(
                    name="cue_recall",
                    value=round(recall, 3),
                    comment=f"{recovered}/{len(ground_truth)} announced transitions recovered",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Langfuse scoring failed: {exc!r}")

    if lf is not None:
        with contextlib.suppress(Exception):
            lf.flush()

    logger.success(f"Segmentation: {len(found)} segment(s), cue_recall={recall:.2f} for episode {episode_number}.")
    return found


def _windows(segments: list[dict], window_s: int) -> list[list[dict]]:
    buckets: dict[int, list[dict]] = {}
    for seg in segments:
        buckets.setdefault(int(seg["start"] // window_s), []).append(seg)
    return [buckets[k] for k in sorted(buckets)]


def _langfuse_client():  # noqa: ANN202
    """An authenticated Langfuse client, or None if tracing isn't configured/reachable."""
    try:
        from langfuse import get_client  # noqa: PLC0415

        client = get_client()
        return client if client.auth_check() else None
    except Exception:  # noqa: BLE001 - tracing is optional
        return None


def _detect_in_window(lf, window, module, model, use_dspy):  # noqa: ANN001
    """Detect + deterministically timestamp-resolve transitions in one window."""
    from transcription_bot.interfaces import dspy_segmenter as _d  # noqa: PLC0415

    if not use_dspy:
        # Direct JSON prompt — fast/clean, and auto-traced by langfuse.openai under the
        # active episode span. Resolve the model's hits to real lines (deterministic ts).
        lines = [f"[{_fmt_ts(s['start'])}] {s['speaker']}: {s['text']}" for s in window]
        return _resolve_transitions(segment_window(lines, model), window)

    # DSPy path (opt-in): litellm isn't auto-traced, so wrap a manual generation span.
    if lf is None:
        return _d.detect_transitions(window, module)
    lines_str = "\n".join(f"[{_fmt_ts(s['start'])}] {s['speaker']}: {s['text']}" for s in window)
    with lf.start_as_current_observation(name="segment_window", as_type="generation", model=model, input=lines_str) as gen:
        result = _d.detect_transitions(window, module)
        with contextlib.suppress(Exception):
            gen.update(output=result)
        return result


def _resolve_transitions(raw, window):  # noqa: ANN001
    """Map raw LLM transitions to real transcript lines (deterministic timestamps, #18)."""
    from transcription_bot.interfaces.dspy_segmenter import _resolve_to_segment  # noqa: PLC0415

    out, seen = [], set()
    for t in raw or []:
        if not isinstance(t, dict):
            continue
        seg = _resolve_to_segment(t, window)
        if seg is None:
            continue
        key = (t.get("type") or "other", round(seg["start"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "type": t.get("type") or "other",
                "evidence": (t.get("evidence") or t.get("title") or "")[:120],
                "start_s": float(seg["start"]),
                "start_ts": _fmt_ts(seg["start"]),
                "speaker": seg["speaker"],
            }
        )
    return out


def _cue_ground_truth(segments: list[dict]) -> list[dict]:
    """Announced transitions found by simple cue patterns — the score's ground truth."""
    from transcription_bot.interfaces.dspy_segmenter import _CUES  # noqa: PLC0415

    ground_truth = []
    for seg in segments:
        low = seg["text"].lower()
        for cue, seg_type in _CUES.items():
            if cue in low:
                ground_truth.append({"type": seg_type, "start_s": float(seg["start"])})
                break
    return ground_truth


def _recovered_count(ground_truth: list[dict], found: list[dict], tol_s: float = 90.0) -> int:
    return sum(
        any(f["type"] == g["type"] and abs(f["start_s"] - g["start_s"]) <= tol_s for f in found)
        for g in ground_truth
    )
