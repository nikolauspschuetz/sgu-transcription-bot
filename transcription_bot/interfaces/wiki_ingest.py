"""Ingest existing sgutranscripts.org transcripts into the timestream (test corpus, #14).

The wiki holds ~600 human-proofread transcripts. Pulling them gives MANY episodes for
segmentation testing without running ASR — and since the cue-anchor is pure regex, it runs
across dozens of episodes in seconds.

Caveat: wiki transcripts use single-letter speaker labels and carry no per-line timestamps
(only section-header times). We map speakers via a small table and **synthesize sequential
timestamps**. That's fine for text-based segmentation testing (does the anchor find
"it's science or fiction time"?), but it is NOT the real ASR timestream — treat these rows
as a text corpus, not ground-truth timing.
"""

import re

import httpx
from loguru import logger

_RAW_URL = "https://www.sgutranscripts.org/w/index.php?title=SGU_Episode_{ep}&action=raw"

# Wiki single-letter speaker codes -> display names (guests keep their initials).
_SPEAKER = {
    "S": "Steve", "B": "Bob", "J": "Jay", "C": "Cara", "E": "Evan",
    "R": "Rebecca", "P": "Perry", "VO": "Voice-over", "US": "Unknown",
}
_LINE = re.compile(r"^'''\s*([^:'\n]{1,20}?):\s*'''\s*(.*)$")


def _strip_markup(text: str) -> str:
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)                    # {{templates}}
    text = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", text)      # [[link|label]] -> label
    text = re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)               # [[link]] -> link
    text = re.sub(r"\[https?://\S+\s+([^\]]*)\]", r"\1", text)    # [url label] -> label
    text = re.sub(r"<[^>]+>", "", text)                           # html/ref tags
    text = text.replace("'''", "").replace("''", "")             # bold/italic
    return text.strip()


def fetch_transcript(episode: int, client: httpx.Client | None = None) -> list[dict] | None:
    """Fetch + parse a wiki episode into timestream rows (synthetic timestamps), or None."""
    own = client is None
    client = client or httpx.Client(timeout=30, follow_redirects=True)
    try:
        resp = client.get(_RAW_URL.format(ep=episode))
    finally:
        if own:
            client.close()
    if resp.status_code != 200 or len(resp.text) < 500:
        return None

    rows: list[tuple[str, str]] = []
    for line in resp.text.splitlines():
        m = _LINE.match(line.strip())
        if not m:
            continue
        text = _strip_markup(m.group(2))
        if text:
            rows.append((_SPEAKER.get(m.group(1), m.group(1)), text))

    if len(rows) < 20:  # not a real transcript (stub / editing-required page)
        return None

    return [
        {"seq": i, "start": i * 4.0, "end": i * 4.0 + 4.0, "speaker": sp, "text": tx}
        for i, (sp, tx) in enumerate(rows)
    ]


def ingest(episodes: list[int]) -> list[int]:
    """Fetch + store each episode's transcript. Returns the episodes actually ingested."""
    from transcription_bot.metadata import store  # noqa: PLC0415

    store.init_schema()
    done = []
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        for ep in episodes:
            segs = fetch_transcript(ep, client)
            if not segs:
                logger.warning(f"ep {ep}: no usable transcript (missing/stub)")
                continue
            store.save_transcript(ep, segs)
            done.append(ep)
            logger.success(f"ep {ep}: ingested {len(segs)} lines")
    return done
