"""Human approval gate before publishing a wiki page (issue #8).

Nothing reaches the wiki without an explicit go. The flow:

1. Always write the rendered wiki draft to ``output/`` (never lost).
2. Fetch the current wiki page (if any) and show a unified diff of what would change.
3. Gate:
   - default (no ``publish``): stop — this is a dry run.
   - ``publish`` + interactive TTY: ask for y/N confirmation.
   - ``publish`` + ``assume_yes``: auto-approve (for supervised automation).
   - ``publish`` but non-interactive and not ``assume_yes``: refuse (fail safe).
4. Only on approval call the real publish path.

This is engine-agnostic: it gates whatever wiki markup it's handed. (The full wiki
render still depends on segmentation/LLM — see issue #6 — but the gate does not.)
"""

import difflib
import sys
from pathlib import Path

from loguru import logger

from transcription_bot.interfaces.wiki import (
    EPISODE_PAGE_PREFIX,
    episode_has_wiki_page,
    get_wiki_page,
    save_wiki_page,
)


def review_and_maybe_publish(
    *,
    episode_number: int,
    page_text: str,
    output_dir: Path | None = None,
    publish: bool = False,
    assume_yes: bool = False,
) -> bool:
    """Draft + diff a wiki page and publish only with explicit approval.

    Returns True iff the page was published.
    """
    page_title = f"{EPISODE_PAGE_PREFIX}{episode_number}"
    output_dir = output_dir or Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Always persist the draft.
    draft_path = output_dir / f"{page_title}.draft.wiki"
    draft_path.write_text(page_text, encoding="utf-8")
    logger.success(f"Wrote wiki draft: {draft_path}")

    # 2. Diff against the live page (if it exists).
    page_exists = episode_has_wiki_page(episode_number)
    current_text = _fetch_current_text(page_title) if page_exists else None

    diff_text = _render_diff(current_text, page_text, page_title)
    (output_dir / f"{page_title}.diff").write_text(diff_text, encoding="utf-8")
    print("\n" + diff_text + "\n")

    # 3. Gate.
    if not publish:
        logger.info(
            f"Dry run: no publish requested. Draft + diff written to {output_dir}/. "
            f"Re-run with publish enabled to push {page_title}."
        )
        return False

    if not _approved(page_title, page_exists=page_exists, assume_yes=assume_yes):
        logger.info("Publish declined — nothing sent to the wiki.")
        return False

    # 4. Publish (create-only unless the page already exists).
    save_wiki_page(page_title, page_text, allow_page_editing=page_exists)
    logger.success(f"Published {page_title} to the wiki.")
    return True


def _fetch_current_text(page_title: str) -> str | None:
    """Best-effort fetch of the current page markup for diffing."""
    try:
        return str(get_wiki_page(page_title))
    except Exception as exc:  # noqa: BLE0001 — diff is best-effort; never block on it
        logger.warning(f"Couldn't fetch current wiki page for diff ({page_title}): {exc}")
        return None


def _render_diff(current_text: str | None, draft_text: str, page_title: str) -> str:
    """Unified diff of current vs draft, or a summary when the page is new."""
    if current_text is None:
        line_count = len(draft_text.splitlines())
        return f"# NEW PAGE: {page_title} ({line_count} lines) — no existing page to diff against."

    diff = difflib.unified_diff(
        current_text.splitlines(),
        draft_text.splitlines(),
        fromfile=f"{page_title} (current wiki)",
        tofile=f"{page_title} (draft)",
        lineterm="",
    )
    diff_body = "\n".join(diff)
    return diff_body or f"# NO CHANGES: draft is identical to the current {page_title}."


def _approved(page_title: str, *, page_exists: bool, assume_yes: bool) -> bool:
    """Decide whether to proceed with publishing."""
    action = "OVERWRITE existing" if page_exists else "CREATE"

    if assume_yes:
        logger.warning(f"assume_yes set — auto-approving publish ({action} {page_title}).")
        return True

    if not sys.stdin.isatty():
        logger.error("Refusing to publish: non-interactive session and approval not pre-granted (assume_yes).")
        return False

    reply = input(f"Publish? This will {action} '{page_title}' on the wiki. [y/N] ").strip().lower()
    return reply in {"y", "yes"}
