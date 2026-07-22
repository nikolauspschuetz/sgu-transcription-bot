"""Local diarization backend using pyannote.audio (free, offline).

Drop-in replacement for ``interfaces.pyannote.create_diarization``. It matches the
contract the merge step expects:

    create_diarization(rss_entry) -> pandas.DataFrame[start, end, speaker] | None

Selected when ``config.diarization_backend`` is ``"local_pyannote"`` (see
``handlers/transcription_handler/_diarized_transcript.py``).

Speakers come back as generic labels (``SPEAKER_00`` ...). Mapping those to the rogues'
names (Steve/Bob/Jay/Cara/...) is a separate, retrainable step — see issue #5. The
downstream merge already treats a leading ``SPEAKER_`` label as the voice-over, so
generic labels are safe to hand back.

NOTE: not yet exercised end-to-end — needs the ``local`` dependency group and a (free)
HuggingFace token that has accepted the ``pyannote/speaker-diarization-3.1`` terms.
See SETUP.md and issue #3.
"""

import functools
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from transcription_bot.interfaces._audio import get_episode_audio_path
from transcription_bot.models.data_models import PodcastRssEntry
from transcription_bot.utils.caching import cache_for_episode
from transcription_bot.utils.config import config
from transcription_bot.utils import roster as roster_lib

# Per-episode cluster embeddings live under the gitignored /data/ tree (not package
# data), one .npz per episode mapping SPEAKER_NN -> embedding vector. Captured from
# pyannote 4.x's DiarizeOutput.speaker_embeddings so local speaker naming (#5) and
# enrollment can run without re-diarizing.
_EPISODE_EMBED_DIR = Path("data/embeddings")

# Cosine-similarity floor for accepting a cluster<->member match. wespeaker/pyannote
# embeddings put same-speaker pairs well above this and different speakers below it;
# tune against real episodes. Unmatched clusters stay SPEAKER_NN for human review.
_NAMING_THRESHOLD = 0.5


def _ensure_wav16k(audio_path: Path) -> Path:
    """Decode the episode audio to a clean 16 kHz mono WAV for pyannote.

    pyannote's ``Audio.crop`` does a strict sample-count check, and MP3 decoding drifts
    by a few hundred samples per chunk (the decoder returns a slightly off length),
    raising ``ValueError: ... resulted in N samples instead of the expected M``. A WAV
    has exact sample counts, so we transcode once (cached beside the source) and hand
    pyannote the WAV. 16 kHz mono is what the segmentation/embedding models use anyway.

    We shell out to ``ffmpeg`` (already a hard requirement of the ASR backend) rather
    than add a decode dependency. The write is atomic (temp file + rename) so an
    interrupted transcode never leaves a truncated WAV that a later run would treat as a
    valid cache, and ffmpeg's stderr is surfaced on failure.
    """
    audio_path = Path(audio_path)
    wav_path = audio_path.with_suffix(".16k.wav")
    if wav_path.exists() and wav_path.stat().st_size > 0:
        return wav_path

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found; it is required for diarization.")

    logger.info(f"Transcoding {audio_path.name} -> 16 kHz mono WAV for diarization...")
    fd, tmp = tempfile.mkstemp(suffix=".wav", dir=str(wav_path.parent))
    os.close(fd)
    try:
        result = subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-i", str(audio_path),
             "-ac", "1", "-ar", "16000", "-vn", "-f", "wav", tmp],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed to transcode {audio_path.name} (exit {result.returncode}):\n"
                f"{result.stderr.strip()[-1500:]}"
            )
        if os.path.getsize(tmp) == 0:
            raise RuntimeError(f"ffmpeg produced an empty WAV for {audio_path.name}.")
        os.replace(tmp, wav_path)  # atomic: no partial file is ever seen as cache
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    return wav_path


def _resolve_hf_token() -> str:
    """Resolve the HuggingFace token from TB_HF_TOKEN or a cached ``hf auth login``.

    Two valid sources: the ``TB_HF_TOKEN`` env var, or the token cached by
    ``hf auth login`` at ``~/.cache/huggingface/token`` (auto-refreshing). Raise a clear
    error if neither is present, rather than failing deep inside pyannote.
    """
    if config.hf_token:
        return config.hf_token

    from huggingface_hub import get_token  # noqa: PLC0415

    cached = get_token()
    if cached:
        logger.info("Using HuggingFace token from `hf auth login` cache (TB_HF_TOKEN unset).")
        return cached

    raise RuntimeError(
        "No HuggingFace token found. Set TB_HF_TOKEN in .env, or run `hf auth login`. "
        "The token is free; you must also accept the terms for "
        "'pyannote/speaker-diarization-3.1' on HuggingFace."
    )


@functools.cache
def _get_pipeline(pipeline_name: str, hf_token: str, device: str):  # noqa: ANN202
    """Load (and memoize) the pyannote pipeline. Heavy import is deferred to call time."""
    import torch  # noqa: PLC0415
    from pyannote.audio import Pipeline  # noqa: PLC0415

    logger.info(f"Loading pyannote pipeline (name={pipeline_name}, device={device})...")
    # pyannote.audio 4.x / huggingface_hub 1.x renamed this kwarg: use_auth_token -> token.
    pipeline = Pipeline.from_pretrained(pipeline_name, token=hf_token or None)
    if pipeline is None:
        raise RuntimeError(
            f"Failed to load pyannote pipeline '{pipeline_name}'. "
            "Check TB_HF_TOKEN and that you've accepted the model's terms on HuggingFace."
        )

    return pipeline.to(torch.device(device))


@cache_for_episode(should_cache=lambda x: x is not None)
def create_diarization(rss_entry: PodcastRssEntry) -> pd.DataFrame | None:
    """Create a diarization DataFrame for the episode using local pyannote.audio."""
    audio_path = _ensure_wav16k(get_episode_audio_path(rss_entry))
    pipeline = _get_pipeline(config.diarization_pipeline, _resolve_hf_token(), config.diarization_device)

    logger.info(f"Diarizing episode {rss_entry.episode_number} with local pyannote...")
    result = pipeline(str(audio_path))

    # pyannote.audio 4.x pipelines (e.g. community-1) return a ``DiarizeOutput`` wrapper;
    # 3.x pipelines return an ``Annotation`` directly. Support both. For merging with the
    # transcript we prefer the exclusive (non-overlapping) diarization, which pyannote
    # documents as "adapted to downstream transcription".
    if hasattr(result, "itertracks"):
        annotation = result
    else:
        annotation = getattr(result, "exclusive_speaker_diarization", None) or result.speaker_diarization

    rows = [
        {"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]

    if not rows:
        logger.warning("Local diarization produced no speaker turns.")
        return None

    _capture_cluster_embeddings(rss_entry, result)

    speaker_count = len({row["speaker"] for row in rows})
    logger.success(f"Local diarization complete: {len(rows)} turns across {speaker_count} speakers.")
    return pd.DataFrame(rows, columns=["start", "end", "speaker"])


def _capture_cluster_embeddings(rss_entry: PodcastRssEntry, result) -> None:  # noqa: ANN001
    """Persist SPEAKER_NN -> embedding vectors from a pyannote 4.x DiarizeOutput.

    No-op for a plain Annotation (3.x) or when embeddings are absent. Stored as one
    .npz per episode so naming (#5) and enrollment run without re-diarizing.
    """
    embeddings = getattr(result, "speaker_embeddings", None)
    diar = getattr(result, "speaker_diarization", None)
    if embeddings is None or diar is None:
        return
    labels = list(diar.labels())  # embeddings are sorted in this label order
    if len(labels) != len(embeddings):
        logger.warning(f"Embedding/label mismatch ({len(embeddings)} vs {len(labels)}); skipping capture.")
        return
    _EPISODE_EMBED_DIR.mkdir(parents=True, exist_ok=True)
    path = _EPISODE_EMBED_DIR / f"{rss_entry.episode_number}.npz"
    np.savez(path, **{label: np.asarray(vec, dtype=np.float32) for label, vec in zip(labels, embeddings)})
    logger.info(f"Captured {len(labels)} speaker embeddings -> {path}")


def load_episode_embeddings(episode_number: int) -> dict[str, np.ndarray] | None:
    """Load captured SPEAKER_NN -> embedding vectors for an episode, if present."""
    path = _EPISODE_EMBED_DIR / f"{episode_number}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return {key: data[key] for key in data.files}


def name_speakers(episode_number: int, diarization: pd.DataFrame, threshold: float = _NAMING_THRESHOLD) -> pd.DataFrame:
    """Relabel SPEAKER_NN clusters with roster names via cosine-matched voiceprints.

    Returns ``diarization`` unchanged when no per-episode embeddings were captured or
    no roster members are enrolled — so it is a safe no-op until voices are enrolled.
    Multiple clusters may map to the same member (this merges over-split clusters);
    clusters below ``threshold`` keep their SPEAKER_NN label for human review (#5).
    """
    embeddings = load_episode_embeddings(episode_number)
    if not embeddings:
        return diarization

    match_fn = _resolve_matcher(threshold)
    if match_fn is None:
        return diarization

    mapping: dict[str, str] = {label: name for label, vec in embeddings.items() if (name := match_fn(vec))}
    if not mapping:
        return diarization

    named = diarization.copy()
    named["speaker"] = named["speaker"].map(lambda s: mapping.get(s, s))
    logger.success(f"Named {len(mapping)} of {len(embeddings)} clusters: {mapping}")
    return named


def _resolve_matcher(threshold: float):  # noqa: ANN202
    """A ``cluster_embedding -> member_name | None`` matcher: metadata store first, else files.

    Prefers the pgvector store (the source of truth, #13); if it's unreachable or has no
    enrolled voiceprints, falls back to the file-based roster voiceprints so the pipeline
    still runs without the DB.
    """
    try:
        from transcription_bot.metadata import store  # noqa: PLC0415

        with store.connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM voiceprints")
            enrolled_count = cur.fetchone()[0]
        if enrolled_count:
            logger.info(f"Speaker naming via metadata store (pgvector): {enrolled_count} voiceprint(s).")

            def _db_match(vec: np.ndarray) -> str | None:
                hit = store.match_speaker(vec, threshold=threshold)
                return hit["display_name"] if hit else None

            return _db_match
        logger.info("Metadata store reachable but empty; trying file voiceprints.")
    except Exception as exc:  # noqa: BLE001 - DB optional; degrade to files
        logger.warning(f"Metadata store unavailable ({type(exc).__name__}); using file voiceprints.")

    enrolled = roster_lib.load_roster().enrolled()
    if not enrolled:
        return None
    refs = [(m.display_name, _l2_normalize(m.load_embedding())) for m in enrolled]  # pyright: ignore[reportArgumentType]
    logger.info(f"Speaker naming via file voiceprints: {len(refs)} enrolled.")

    def _file_match(vec: np.ndarray) -> str | None:
        cluster = _l2_normalize(vec)
        best_name, best_sim = None, threshold
        for name, ref in refs:
            sim = float(np.dot(cluster, ref))
            if sim > best_sim:
                best_name, best_sim = name, sim
        return best_name

    return _file_match


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    return vec if norm == 0 else vec / norm
