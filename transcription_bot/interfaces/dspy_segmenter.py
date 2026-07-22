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
        )
    )


def _window_lines(segments: list[dict]) -> str:
    return "\n".join(f"[{_fmt_ts(s['start'])}] {s['speaker']}: {s['text']}" for s in segments)


def bootstrap_trainset(episode_numbers: list[int], window_minutes: int = 15) -> list[dspy.Example]:
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
    optimizer = dspy.BootstrapFewShot(metric=_metric, max_bootstrapped_demos=4, max_labeled_demos=4)
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
