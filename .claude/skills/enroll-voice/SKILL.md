---
name: enroll-voice
description: Enroll a rogue's or guest's voice into the roster so diarization clusters get named instead of SPEAKER_NN (issue #5). Use when asked to name speakers, teach the bot a voice, fix SPEAKER_NN labels, or add a reference voiceprint.
---

# Enroll a voice (local speaker naming, #5)

Turns anonymous diarization clusters (`SPEAKER_NN`) into roster names by building
reference voiceprints from labelled clusters.

## Prereqs
- The episode's cluster embeddings must exist at `data/embeddings/<EP>.npz`. If missing,
  delete the diarization cache (`data/cache/*local_diarization*/*<EP>*`) and re-run the
  `transcribe-episode` skill — `create_diarization` captures embeddings on a real run.
- The person must be in `transcription_bot/data/roster.toml` with a stable `id` (e.g. `steve`).

## Steps
1. **Identify the cluster.** Read `output/SGU_Episode_<EP>.vtt` and find a line that
   identifies the person (Steve self-IDs in the intro: "this is your host, Steven
   Novella"). **pyannote cluster numbers are NOT stable across runs — always identify
   from the current run's output, never assume a fixed SPEAKER_NN.**
2. **Enroll:**
   ```python
   from transcription_bot.interfaces.local_diarization import load_episode_embeddings
   from transcription_bot.utils.roster import enroll_embedding
   emb = load_episode_embeddings(<EP>)
   enroll_embedding("<member_id>", emb["<SPEAKER_NN>"])   # -> data/voiceprints/<id>.npy
   ```
   Retrainable: repeat across episodes to average multiple samples into the member's centroid.
3. **Re-run** the entrypoint (ASR + diarization are cached; `name_speakers()` relabels
   post-cache). The VTT should now show the real name.
4. **Tune** `_NAMING_THRESHOLD` in `local_diarization.py` if matches are missed (too high)
   or wrong (too low).

## Notes
- Multiple clusters mapping to one member is expected and good (merges over-split clusters).
- Human "this cluster is X" corrections during review are the primary enrollment source over time.
