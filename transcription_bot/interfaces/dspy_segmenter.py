"""DSPy-compiled segmentation — program, not prompt (#18).

Replaces the hand-tuned _SEGMENT_SYSTEM string with a DSPy program the model is *compiled*
to satisfy. Runs on Ollama (local/free); Langfuse traces the underlying LM calls.

Design (per #18):
- Signature: a transcript window -> the announcing lines (line_ref, type, evidence).
- Timestamps stay DETERMINISTIC — the caller maps each `line_ref` back to a real
  transcript line; the LLM never copies timestamps (its weakest skill).
- compile(): BootstrapFewShot over a trainset auto-bootstrapped from cue patterns, so no
  hand labeling is needed for the clear cases.
"""

from pathlib import Path

import dspy
from loguru import logger

from transcription_bot.utils.config import config

_ARTIFACT = Path("data/dspy/segmenter.json")

# Cue phrases that reliably mark an announced transition — used ONLY to auto-label a
# bootstrap trainset. DSPy's job is to generalize beyond these to the fuzzy cases.
_CUES = {
    "science or fiction": "science or fiction",
    "who's that noisy": "who's that noisy",
    "what's the word": "what's the word",
    "our next news item": "news item",
    "interview with": "interview",
    "quote of the week": "quote",
}


class DetectTransitions(dspy.Signature):
    """Find where segments START in a Skeptics' Guide transcript excerpt.

    Each input line is `[H:MM:SS] Speaker: text`. The host announces most transitions in
    natural language. Return one entry per announcing line; do NOT invent transitions.
    """

    transcript_window: str = dspy.InputField(desc="lines, each '[H:MM:SS] Speaker: text'")
    transitions: list[dict] = dspy.OutputField(
        desc=(
            "list of {line_ref, type, evidence}; line_ref = the exact [H:MM:SS] bracket of the "
            "announcing line; type in {intro,news item,who's that noisy,science or fiction,"
            "interview,emails,quote,outro,other}; evidence = the announcing phrase"
        )
    )


def _fmt_ts(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}"


def configure_lm(model: str | None = None) -> None:
    """Point DSPy at Ollama's OpenAI-compatible endpoint (Langfuse traces underneath)."""
    model = model or config.ollama_model
    dspy.configure(
        lm=dspy.LM(
            f"openai/{model}",
            api_base=config.ollama_base_url.rstrip("/") + "/v1",
            api_key="ollama",
            temperature=0.0,
            max_tokens=2000,  # bound generation — unbounded reasoning ran 13 min/call
        )
    )


def _window_lines(segments: list[dict]) -> str:
    return "\n".join(f"[{_fmt_ts(s['start'])}] {s['speaker']}: {s['text']}" for s in segments)


def bootstrap_trainset(episode_numbers: list[int], window_minutes: int = 6) -> list[dspy.Example]:
    """Auto-label windows via cue phrases -> DSPy examples (window -> known transitions)."""
    from transcription_bot.metadata import store  # noqa: PLC0415

    examples: list[dspy.Example] = []
    window_s = window_minutes * 60
    for episode in episode_numbers:
        buckets: dict[int, list[dict]] = {}
        for seg in store.get_transcript_segments(episode):
            buckets.setdefault(int(seg["start"] // window_s), []).append(seg)
        for bucket_segs in buckets.values():
            labels = []
            for seg in bucket_segs:
                low = seg["text"].lower()
                for cue, seg_type in _CUES.items():
                    if cue in low:
                        labels.append({"line_ref": _fmt_ts(seg["start"]), "type": seg_type, "evidence": cue})
                        break
            if labels:  # keep only windows that contain at least one known transition
                examples.append(
                    dspy.Example(transcript_window=_window_lines(bucket_segs), transitions=labels).with_inputs(
                        "transcript_window"
                    )
                )
    return examples


def _metric(example: dspy.Example, pred, trace=None) -> float:  # noqa: ANN001
    """Fraction of the known (cue-labeled) transition *types* the prediction recovered."""
    gold = {t["type"] for t in example.transitions}
    got = {t.get("type") for t in (getattr(pred, "transitions", None) or []) if isinstance(t, dict)}
    return len(gold & got) / len(gold) if gold else 1.0


def compile_segmenter(episode_numbers: list[int], model: str | None = None) -> dict:
    """Compile the DSPy segmenter from a cue-bootstrapped trainset; save the artifact."""
    configure_lm(model)
    trainset = bootstrap_trainset(episode_numbers)
    if not trainset:
        raise RuntimeError("No bootstrap examples — index episodes into the timestream first.")

    logger.info(f"Compiling DSPy segmenter on {len(trainset)} bootstrapped window(s), model={model or config.ollama_model}.")
    # Keep demos few and small — they're prepended to every inference prompt (a local model
    # chokes on giant few-shot context), and short cue windows are enough to teach the task.
    optimizer = dspy.BootstrapFewShot(metric=_metric, max_bootstrapped_demos=2, max_labeled_demos=2)
    compiled = optimizer.compile(dspy.Predict(DetectTransitions), trainset=trainset)

    _ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    compiled.save(str(_ARTIFACT))
    logger.success(f"Saved compiled segmenter -> {_ARTIFACT}")
    return {"trainset_size": len(trainset), "artifact": str(_ARTIFACT)}


def load_segmenter(model: str | None = None) -> dspy.Predict:
    """Load the compiled segmenter (or a fresh one if not yet compiled)."""
    configure_lm(model)
    module = dspy.Predict(DetectTransitions)
    if _ARTIFACT.exists():
        module.load(str(_ARTIFACT))
    return module


def _parse_ts(ref) -> float | None:  # noqa: ANN001
    """Parse an 'H:MM:SS' / 'MM:SS' reference to seconds, or None."""
    import re  # noqa: PLC0415

    if not ref:
        return None
    m = re.search(r"(\d+):(\d{1,2}):(\d{1,2})", str(ref))
    if m:
        h, mi, s = map(int, m.groups())
        return h * 3600 + mi * 60 + s
    m = re.search(r"(\d{1,2}):(\d{1,2})", str(ref))
    if m:
        mi, s = map(int, m.groups())
        return mi * 60 + s
    return None


def _resolve_to_segment(transition: dict, window_segments: list[dict]) -> dict | None:
    """Map an LLM transition to a REAL transcript line (deterministic timestamps, #18).

    Never trust the model's timestamp: match its `evidence` phrase against the actual line
    text and use that line's start time; fall back to snapping its `line_ref` to the
    nearest real line. Return None (drop) if neither resolves.
    """
    evidence = (transition.get("evidence") or "").lower().strip()
    if evidence:
        for seg in window_segments:
            if evidence in seg["text"].lower():
                return seg
    secs = _parse_ts(transition.get("line_ref") or transition.get("start_timestamp"))
    if secs is not None and window_segments:
        return min(window_segments, key=lambda s: abs(s["start"] - secs))
    return None


def detect_transitions(window_segments: list[dict], module: dspy.Predict) -> list[dict]:
    """Run the compiled DSPy segmenter on one window; resolve each hit to a real line."""
    import json  # noqa: PLC0415

    pred = module(transcript_window=_window_lines(window_segments))
    raw = getattr(pred, "transitions", None)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = []
    if isinstance(raw, dict):
        raw = raw.get("transitions", [])

    out, seen = [], set()
    for t in raw or []:
        if not isinstance(t, dict):
            continue
        seg = _resolve_to_segment(t, window_segments)
        if seg is None:
            continue
        key = (t.get("type") or "other", round(seg["start"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "type": t.get("type") or "other",
                "evidence": (t.get("evidence") or "")[:120],
                "start_s": float(seg["start"]),
                "start_ts": _fmt_ts(seg["start"]),
                "speaker": seg["speaker"],
            }
        )
    return out
