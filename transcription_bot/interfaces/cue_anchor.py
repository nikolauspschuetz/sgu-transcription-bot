"""Cue-anchor segmentation — the high-precision, instant, regex primary pass (#12).

SGU is self-documenting: the host announces most transitions with a fixed phrasing
("it's science or fiction time", "who's that noisy time", "let's move on to the news").
These announcement patterns are unambiguous and don't appear in ordinary discussion, so a
regex over the transcript timestream finds the real boundaries with high precision and
zero LLM cost. The LLM is only needed for the *un-announced* / fuzzy transitions the
anchor misses (the hybrid's second pass).

Benchmark motivation: a local LLM confused discussion mentions of "science or fiction" for
the segment transition; the cue regex used as ground truth caught all 3 real announcements
perfectly. So the regex is the primary detector, not the fallback.
"""

import re

# Each entry: (compiled pattern, fixed type or None). None types are classified from the
# captured group. Patterns target the ANNOUNCEMENT phrasing, not bare segment names.
_ANCHORS: list[tuple[re.Pattern, str | None]] = [
    (re.compile(r"who'?s that noisy\s+time", re.I), "who's that noisy"),
    (re.compile(r"science or fiction\s+time", re.I), "science or fiction"),
    (re.compile(r"time for\s+(?:some\s+)?(science or fiction|who'?s that noisy|what'?s the word|the news|news items?)", re.I), None),
    (re.compile(r"it'?s time for\s+(.{3,30}?)[.,!?]", re.I), None),
    (re.compile(r"what'?s the word", re.I), "what's the word"),
    (re.compile(r"(?:let'?s|we'?ll|we're going to|why don'?t we)\s+(?:move on|go on|jump|move)\s+(?:on\s+)?to\s+(?:the\s+|our\s+)?(.{3,25}?)[.,!?]", re.I), None),
    (re.compile(r"(?:our|the)\s+(?:next|first|last)\s+news item", re.I), "news item"),
    (re.compile(r"skeptical quote of the week|quote of the week", re.I), "quote"),
    (re.compile(r"dumbest thing of the week", re.I), "dumbest thing of the week"),
    (re.compile(r"name that logical fallacy", re.I), "logical fallacy"),
]

# Map free-text captures to canonical segment types.
_TYPE_HINTS = {
    "science or fiction": "science or fiction",
    "noisy": "who's that noisy",
    "what's the word": "what's the word",
    "news": "news item",
    "interview": "interview",
    "email": "emails",
    "quote": "quote",
}


def _classify(text: str) -> str:
    low = text.lower()
    for key, seg_type in _TYPE_HINTS.items():
        if key in low:
            return seg_type
    return "news item"  # "let's move on to X" that isn't a named segment is usually the next news item


def _fmt_ts(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}"


def anchor_transitions(segments: list[dict]) -> list[dict]:
    """Regex-detect announced transitions in the timestream (instant, high precision).

    Returns one transition per matching line with its REAL timestamp — no LLM, no guessing.
    """
    out, seen = [], set()
    for seg in segments:
        text = seg["text"]
        for pattern, fixed_type in _ANCHORS:
            m = pattern.search(text)
            if not m:
                continue
            seg_type = fixed_type or _classify(m.group(1) if m.groups() and m.group(1) else text)
            key = (seg_type, round(float(seg["start"]) / 30))  # dedup within ~30s
            if key in seen:
                break
            seen.add(key)
            out.append(
                {
                    "type": seg_type,
                    "evidence": text.strip()[:120],
                    "start_s": float(seg["start"]),
                    "start_ts": _fmt_ts(seg["start"]),
                    "speaker": seg["speaker"],
                    "source": "anchor",
                }
            )
            break
    out.sort(key=lambda x: x["start_s"])
    return out
