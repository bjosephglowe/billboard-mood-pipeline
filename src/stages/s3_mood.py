import time
from typing import Optional

import pandas as pd

from src.core.cache import get_inference_cache
from src.core.checkpoint import checkpoint_exists, read_checkpoint, write_checkpoint
from src.core.config import PipelineConfig
from src.core.logger import get_logger, get_low_confidence_logger
from src.core.models import get_device, load_mood_model, unload_model

log = get_logger("s3_mood")

# ── Constants (locked P2 Section 3) ──────────────────────────────────────────

# Canonical mood taxonomy — locked P2 Section 3.
# Labels match j-hartmann/emotion-english-distilroberta-base output exactly.
_MOOD_LABELS = frozenset({
    "joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"
})

# Minimum probability for a secondary mood label to be retained (locked P2).
_SECONDARY_THRESHOLD = 0.20

# Primary label is flagged low_confidence below this threshold (locked P2).
_LOW_CONFIDENCE_THRESHOLD = 0.35

# Tiebreak order when two labels have probability difference < 0.01 (locked P2).
# Earlier in list = higher priority.
_TIEBREAK_ORDER = [
    "joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"
]

# Schema 4 columns (locked P5)
_SCHEMA_4_COLUMNS = [
    "song_id",
    "mood_primary",
    "mood_primary_confidence",
    "mood_secondary",
    "mood_secondary_confidence",
    "mood_flag",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_tie(probs: dict) -> str:
    """
    Select the primary mood label when two labels are within 0.01 of each other.

    Applies the locked tiebreak order from P2 Section 3:
    joy > sadness > anger > fear > disgust > surprise > neutral

    Finds the top probability value, collects all labels within 0.01 of it,
    then returns the one earliest in the tiebreak order.

    Args:
        probs: dict mapping mood label -> probability

    Returns:
        Tiebreak-resolved primary label string.
    """
    if not probs:
        return "neutral"

    max_prob = max(probs.values())
    candidates = [
        label for label, prob in probs.items()
        if max_prob - prob < 0.01
    ]

    for label in _TIEBREAK_ORDER:
        if label in candidates:
            return label

    # Fallback: return the label with the highest raw probability
    return max(probs, key=lambda k: probs[k])


def select_secondary(
    probs: dict,
    primary: str,
    threshold: float = _SECONDARY_THRESHOLD,
) -> tuple[Optional[str], Optional[float]]:
    """
    Select the secondary mood label if it clears the confidence threshold.

    Excludes the primary label from consideration. Returns (None, None)
    if no remaining label meets the threshold.

    Args:
        probs: dict mapping mood label -> probability
        primary: already-selected primary label (excluded from candidates)
        threshold: minimum probability for secondary retention (default 0.20)

    Returns:
        Tuple of (secondary_label, secondary_confidence) or (None, None).
    """
    candidates = {
        label: prob for label, prob in probs.items()
        if label != primary and prob >= threshold
    }
    if not candidates:
        return None, None

    secondary = max(candidates, key=lambda k: candidates[k])
    return secondary, round(candidates[secondary], 6)


def aggregate_mood(chunk_results: list[dict]) -> dict:
    """
    Aggregate per-chunk mood classifications to a song-level result.

    Primary mood is the modal label across all chunks (most frequent).
    Primary confidence is the mean probability of the modal label across
    all chunks where it appeared.

    If the modal label has a tie in frequency, resolve_tie is applied
    to the mean probabilities of the tied labels.

    Args:
        chunk_results: list of prob dicts, one per chunk.
                       Each maps mood_label -> probability.

    Returns:
        Dict with all Schema 4 mood fields (excluding song_id).
    """
    if not chunk_results:
        return {
            "mood_primary": None,
            "mood_primary_confidence": None,
            "mood_secondary": None,
            "mood_secondary_confidence": None,
            "mood_flag": None,
        }

    # Count how many chunks each label is the top label
    label_counts: dict[str, int] = {}
    for prob_map in chunk_results:
        top_label = max(prob_map, key=lambda k: prob_map[k])
        label_counts[top_label] = label_counts.get(top_label, 0) + 1

    max_count = max(label_counts.values())
    modal_candidates = [
        label for label, count in label_counts.items()
        if count == max_count
    ]

    # Compute mean probabilities across all chunks for each label
    mean_probs: dict[str, float] = {}
    for label in _MOOD_LABELS:
        mean_probs[label] = sum(
            chunk.get(label, 0.0) for chunk in chunk_results
        ) / len(chunk_results)

    # Resolve ties among modal candidates using mean probabilities
    if len(modal_candidates) == 1:
        primary = modal_candidates[0]
    else:
        tie_probs = {label: mean_probs[label] for label in modal_candidates}
        primary = resolve_tie(tie_probs)

    primary_confidence = round(mean_probs[primary], 6)

    secondary, secondary_confidence = select_secondary(
        probs=mean_probs,
        primary=primary,
        threshold=_SECONDARY_THRESHOLD,
    )

    flag = "low_confidence" if primary_confidence < _LOW_CONFIDENCE_THRESHOLD else None

    return {
        "mood_primary": primary,
        "mood_primary_confidence": primary_confidence,
        "mood_secondary": secondary,
        "mood_secondary_confidence": secondary_confidence,
        "mood_flag": flag,
    }


def classify_chunks(chunks: list[str], pipe, batch_size: int) -> list[dict]:
    """
    Run mood classification inference on a list of lyric chunks.

    Args:
        chunks: list of lyric chunk strings
        pipe: loaded HuggingFace text-classification pipeline
        batch_size: number of chunks per inference batch

    Returns:
        List of dicts, one per chunk. Each maps mood_label -> probability.
    """
    results = []

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        raw_outputs = pipe(batch, truncation=True, max_length=512)

        for output in raw_outputs:
            prob_map = {
                item["label"].lower(): item["score"]
                for item in output
            }
            results.append(prob_map)

    return results


def _build_null_record(song_id: str) -> dict:
    """Return an all-null Schema 4 record for songs with missing lyrics."""
    return {
        "song_id": song_id,
        "mood_primary": None,
        "mood_primary_confidence": None,
        "mood_secondary": None,
        "mood_secondary_confidence": None,
        "mood_flag": None,
    }


# ── Stage entry point ─────────────────────────────────────────────────────────

def run(
    config: PipelineConfig,
    lyrics_df: Optional[pd.DataFrame] = None,
    run_id: str = "unknown",
) -> pd.DataFrame:
    """
    Execute Stage 3b: mood inference on all songs.

    Reads S2 checkpoint if lyrics_df not provided. Checks inference cache
    per song before running model. Unloads model after full pass.
    Writes 03_mood.parquet.

    Args:
        config: PipelineConfig instance
        lyrics_df: optional pre-loaded S2 DataFrame
        run_id: pipeline_run_id for log records

    Returns:
        pd.DataFrame matching Schema 4.
    """
    stage = "s3_mood"

    if checkpoint_exists(stage, config) and not config.checkpoints.force_rerun.s3_mood:
        log.info("S3b: checkpoint found — skipping mood inference.")
        return read_checkpoint(stage, config)

    log.info("S3b: starting mood inference.")

    if lyrics_df is None:
        lyrics_df = read_checkpoint("s2_lyrics", config)

    device = get_device(config)
    low_conf_log = get_low_confidence_logger(config.logging.low_confidence_log)
    inference_cache = get_inference_cache(config)

    pipe = None
    records = []
    total = len(lyrics_df)
    low_conf_count = 0

    try:
        pipe, device = load_mood_model(config, device)

        for i, row in enumerate(lyrics_df.to_dict("records"), start=1):
            song_id = row["song_id"]
            lyrics = row.get("lyrics")
            lyrics_status = row.get("lyrics_status", "missing")

            log.debug(
                f"S3b: [{i}/{total}] '{row['title']}' by '{row['artist']}'"
            )

            # ── Null path: missing lyrics ──
            if lyrics_status == "missing" or not lyrics:
                records.append(_build_null_record(song_id))
                continue

            # ── Inference cache check ──
            if config.inference.cache_enabled:
                cached = inference_cache.get_inference(song_id, "mood")
                if cached is not None:
                    log.debug(f"S3b: inference cache hit for {song_id}")
                    records.append(cached)
                    continue

            # ── Chunk and infer ──
            # Reuse chunk_lyrics from s3_sentiment for consistent chunking
            from src.stages.s3_sentiment import chunk_lyrics
            chunks = chunk_lyrics(lyrics, max_tokens=512)

            chunk_results = classify_chunks(
                chunks=chunks,
                pipe=pipe,
                batch_size=config.inference.batch_size,
            )

            result = aggregate_mood(chunk_results)
            result["song_id"] = song_id

            # ── Low-confidence logging ──
            if result["mood_flag"] == "low_confidence":
                low_conf_count += 1
                log.debug(
                    f"S3b: low_confidence for {song_id} "
                    f"(confidence={result['mood_primary_confidence']:.3f})"
                )
                low_conf_log.log(
                    song_id=song_id,
                    year=row["year"],
                    title=row["title"],
                    artist=row["artist"],
                    dimension="mood",
                    flag_value="low_confidence",
                    pipeline_run_id=run_id,
                    score=result["mood_primary_confidence"],
                    threshold=_LOW_CONFIDENCE_THRESHOLD,
                )

            # ── Write to inference cache ──
            if config.inference.cache_enabled:
                inference_cache.set_inference(song_id, "mood", result)

            records.append(result)

            # ── Thermal management ──
            if i % config.inference.batch_size == 0:
                time.sleep(config.inference.sleep_between_batches)

    finally:
        if pipe is not None:
            unload_model(pipe)
        inference_cache.close()

    log.info(
        f"S3b: inference complete. "
        f"total={total} low_confidence={low_conf_count}"
    )

    df = pd.DataFrame(records, columns=_SCHEMA_4_COLUMNS)
    write_checkpoint(stage, df, config)
    log.info("S3b: checkpoint written.")

    return df
