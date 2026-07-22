"""Dagster assets for the SGU pipeline (issue #15).

Per-episode **partitioned** assets over the existing cached pipeline:

    diarized_transcript ──┬─> indexed_transcript   (searchable timestream, Postgres)
                          └─> episode_outputs      (WebVTT + Markdown in output/)

Open the visual DAG + kick off runs:

    uv run dagster dev -m transcription_bot.orchestration.definitions   # UI at :3000

Each episode is a partition, so "process the whole archive" = a **backfill** across
partitions, and re-materializing one asset re-runs one stage (cheap: cache_for_episode
on disk + the Postgres store hold the data). Heavy ASR/diarization still runs on the
native `uv` path — Dagster orchestrates the calls, it doesn't move compute into Docker.
"""

import contextlib
import os
import sys

import dagster as dg
from dagster import AssetExecutionContext
from loguru import logger as _loguru

_LOGURU_TO_DAGSTER = {"WARNING": "warning", "ERROR": "error", "CRITICAL": "error", "DEBUG": "debug"}


@contextlib.contextmanager
def _dagster_logs(context: AssetExecutionContext):
    """Route loguru into Dagster's structured logger and silence tqdm progress bars.

    Inside an asset, our `logger.info/success/...` calls (and those in the interface
    modules) become structured Dagster log events instead of raw captured stderr, and
    DSPy's tqdm bar is disabled. Restores loguru's default stderr sink on exit.
    """

    def _sink(message):  # noqa: ANN001
        record = message.record
        getattr(context.log, _LOGURU_TO_DAGSTER.get(record["level"].name, "info"))(record["message"])

    _loguru.remove()
    handler_id = _loguru.add(_sink, level="INFO")
    prev_tqdm = os.environ.get("TQDM_DISABLE")
    os.environ["TQDM_DISABLE"] = "1"
    try:
        yield
    finally:
        _loguru.remove(handler_id)
        _loguru.add(sys.stderr, level="INFO")
        if prev_tqdm is None:
            os.environ.pop("TQDM_DISABLE", None)
        else:
            os.environ["TQDM_DISABLE"] = prev_tqdm

# Episodes 1..1097 (bump the upper bound as the show continues, or swap for a
# DynamicPartitionsDefinition once episodes are discovered from the RSS feed).
episode_partitions = dg.StaticPartitionsDefinition([str(n) for n in range(1, 1098)])


def _rss_entry(episode_number: int):
    from transcription_bot.parsers.rss_feed import get_podcast_rss_entries  # noqa: PLC0415
    from transcription_bot.utils.global_http_client import http_client  # noqa: PLC0415

    entry = next(
        (e for e in get_podcast_rss_entries(http_client) if e.episode_number == episode_number), None
    )
    if entry is None:
        raise dg.Failure(description=f"Episode {episode_number} not found in the RSS feed.")
    return entry


@dg.asset(
    partitions_def=episode_partitions,
    group_name="transcription",
    description="RSS -> ASR -> diarization -> merge -> speaker naming (reuses on-disk caches).",
)
def diarized_transcript(context: AssetExecutionContext) -> list[dict]:
    from transcription_bot.handlers.transcription_handler import get_transcript  # noqa: PLC0415

    with _dagster_logs(context):
        episode = int(context.partition_key)
        transcript = get_transcript(_rss_entry(episode))
        if not transcript:
            raise dg.Failure(description=f"No transcript produced for episode {episode}.")
        context.add_output_metadata(
            {"segments": len(transcript), "speakers": len({c["speaker"] for c in transcript})}
        )
        return list(transcript)


@dg.asset(
    partitions_def=episode_partitions,
    group_name="corpus",
    description="Persist the diarized transcript into the searchable timestream (Postgres + pgvector).",
)
def indexed_transcript(context: AssetExecutionContext, diarized_transcript: list[dict]) -> None:
    from transcription_bot.metadata import store  # noqa: PLC0415

    with _dagster_logs(context):
        episode = int(context.partition_key)
        store.init_schema()
        count = store.save_transcript(episode, diarized_transcript)
        context.add_output_metadata({"segments_indexed": count})


@dg.asset(
    partitions_def=episode_partitions,
    group_name="output",
    description="Write publish-free WebVTT + Markdown to output/.",
)
def episode_outputs(context: AssetExecutionContext, diarized_transcript: list[dict]) -> None:
    from transcription_bot.serializers.local_outputs import write_transcript_outputs  # noqa: PLC0415

    with _dagster_logs(context):
        episode = int(context.partition_key)
        paths = write_transcript_outputs(diarized_transcript, _rss_entry(episode))
        context.add_output_metadata({"outputs": [str(p) for p in paths]})


@dg.asset(
    partitions_def=episode_partitions,
    deps=[indexed_transcript],
    group_name="segmentation",
    description="Detect host-announced segment transitions for an episode (Ollama, Langfuse-traced).",
)
def episode_segments(context: AssetExecutionContext) -> None:
    from transcription_bot.interfaces.local_llm import segment_episode  # noqa: PLC0415

    with _dagster_logs(context):
        episode = int(context.partition_key)
        segments = segment_episode(episode, model="gemma4:12b")
        context.add_output_metadata(
            {"segments_found": len(segments), "sample": dg.MetadataValue.json(segments[:5])}
        )


@dg.asset(
    group_name="segmentation",
    description="Compile the DSPy segmenter from cue-bootstrapped examples and save the artifact (#18).",
)
def compiled_segmenter(context: AssetExecutionContext) -> None:
    from transcription_bot.interfaces.dspy_segmenter import compile_segmenter  # noqa: PLC0415
    from transcription_bot.metadata import store  # noqa: PLC0415

    with _dagster_logs(context):
        # Bootstrap the trainset from whatever episodes are indexed in the timestream.
        with store.connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT DISTINCT episode_number FROM transcript_segments ORDER BY 1")
            episodes = [r[0] for r in cur.fetchall()]
        if not episodes:
            raise dg.Failure(description="No indexed episodes; materialize indexed_transcript first.")

        meta = compile_segmenter(episodes, model="gemma4:12b")
        context.add_output_metadata({**meta, "bootstrap_episodes": episodes})


defs = dg.Definitions(
    assets=[
        diarized_transcript,
        indexed_transcript,
        episode_outputs,
        episode_segments,
        compiled_segmenter,
    ]
)
