import time
from typing import Optional

import pandas as pd

from src.core.cache import get_inference_cache
from src.core.checkpoint import checkpoint_exists, read_checkpoint, write_checkpoint
from src.core.config import PipelineConfig
from src.core.logger import get_logger, get_low_confidence_logger
from src.core.models import get_device, load_sentiment_model, unload_model

log = get_logger("s3_sentiment")

# ── Constants (locked P2 Section 2) ──────────────────────────────────────────

_LOW_CONFIDENCE_THRESHOLD = 0.45
_MAX_TOKENS = 512

# Sentiment bin boundaries (locked P2 Section 2).
# Ranges are inclusive on the left, exclusive on the right.
# P2 spec:
#   strongly_positive:  0.60 to  1.00  (score >= 0.60)
#   positive:           0.20 to  0.59  (0.20 <= score < 0.60)
#   neutral:           -0.20 to  0.19  (-0.20 <= score < 0.20)  ← upper is 0.20 exclusive
#   negative:          -0.60 to -0.21  (-0.60 <= score < -0.20) ← lower is -0.60 inclusive
#   strongly_negative: -1.00 to -0.60  (score < -0.60)
#
# Implemented as (label, low_inclusive, high_exclusive) checked in priority order.
_BIN_BOUNDARIES = [
    ("strongly_positive",  0.60,  2.0),    # score >= 0.60
    ("positive",           0.20,  0.60),   # 0.20 <= score < 0.60
    ("neutral",           -0.20,  0.20),   # -0.20 <= score < 0.20
    ("negative",          -0.60, -0.20),   # -0.60 <= score < -0.20
    ("strongly_negative", -2.0,  -0.60),   # score < -0.60
]

# Map from model label strings to score multipliers (locked P2 Section 2)
# score = (p_positive * 1.0) + (p_neutral * 0.0) + (p_negative * -1.0)
_LABEL_WEIGHTS = {
    "positive": 1.0,
    "neutral":  0.0,
    "negative": -1.0,
}

# Schema 3 columns (locked P5)
_SCHEMA_3_COLUMNS = [
    "song_id",
    "sentiment_score",
    "sentiment_bin",
    "sentiment_confidence",
    "sentiment_flag",
    "sentiment_chunk_count",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def assign_bin(score: float) -> str:
    """
    Assign a sentiment bin label from a continuous score.

    Bin boundaries (locked P2 Section 2):
      strongly_positive:  score >= 0.60
      positive:           0.20 <= score < 0.60
      neutral:           -0.20 <= score < 0.20
      negative:          -0.60 <= score < -0.20
      strongly_negative:  score < -0.60

    Args:
        score: float, nominally in range -1.0 to 1.0

    Returns:
        Bin label string.
    """
    for label, low, high in _BIN_BOUNDARIES:
        if low <= score < high:
            return label
    # Should not be reached with the wide sentinel bounds above,
    # but guard against floating-point edge cases.
    return "strongly_positive" if score >= 0 else "strongly_negative"


def chunk_lyrics(lyrics: str, max_tokens: int = _MAX_TOKENS) -> list[str]:
    """
    Split lyrics into chunks that fit within the model's token limit.

    Uses a word-based approximation (1 token ≈ 0.75 words) to avoid
    loading a tokenizer solely for chunking. Chunks on line boundaries
    where possible to preserve lyric structure.

    Args:
        lyrics: full lyrics string
        max_tokens: maximum tokens per chunk (default 512)

    Returns:
        List of lyric chunk strings. Empty list if lyrics is empty.
    """
    if not lyrics or not lyrics.strip():
        return []

    # Approximate word limit: 512 tokens * 0.75 words/token ≈ 384 words
    max_words = int(max_tokens * 0.75)

    lines = [line for line in lyrics.split("\n") if line.strip()]
    chunks = []
    current_lines: list[str] = []
    current_word_count = 0

    for line in lines:
        line_words = len(line.split())
        if current_word_count + line_words > max_words and current_lines:
            chunks.append("\n".join(current_lines))
            current_lines = [line]
            current_word_count = line_words
        else:
            current_lines.append(line)
            current_word_count += line_words

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks if chunks else [lyrics]


def score_chunks(
    chunks: list[str],
    pipe,
    batch_size: int,
) -> list[dict]:
    """
    Run sentiment inference on a list of lyric chunks.

    Processes chunks in batches. Each result dict contains the raw
    class probabilities from the model.

    Args:
        chunks: list of lyric chunk strings
        pipe: loaded HuggingFace text-classification pipeline
        batch_size: number of chunks per inference batch

    Returns:
        List of dicts, one per chunk. Each dict maps label -> probability.
        e.g. {"positive": 0.82, "neutral": 0.11, "negative": 0.07}
    """
    results = []

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        raw_outputs = pipe(batch, truncation=True, max_length=_MAX_TOKENS)

        for output in raw_outputs:
            # output is a list of {"label": str, "score": float} dicts
            # Normalise label strings to lowercase for consistent mapping
            prob_map = {
                item["label"].lower(): item["score"]
                for item in output
            }
            results.append(prob_map)

    return results


def aggregate_scores(
    chunk_scores: list[dict],
    chunk_lengths: list[int],
) -> dict:
    """
    Aggregate per-chunk sentiment scores to a single song-level score.

    Method: length-weighted mean of per-chunk continuous scores.
    Locked in P2 Section 2.

    Args:
        chunk_scores: list of probability dicts, one per chunk
        chunk_lengths: list of word counts per chunk (same length)

    Returns:
        Dict with keys: sentiment_score, sentiment_confidence,
        sentiment_bin, sentiment_flag, sentiment_chunk_count.
    """
    if not chunk_scores:
        return {
            "sentiment_score": None,
            "sentiment_bin": None,
            "sentiment_confidence": None,
            "sentiment_flag": None,
            "sentiment_chunk_count": 0,
        }

    total_weight = sum(chunk_lengths)
    if total_weight == 0:
        weights = [1.0 / len(chunk_scores)] * len(chunk_scores)
    else:
        weights = [length / total_weight for length in chunk_lengths]

    weighted_score = 0.0
    weighted_confidence = 0.0

    for prob_map, weight in zip(chunk_scores, weights):
        chunk_score = sum(
            prob_map.get(label, 0.0) * multiplier
            for label, multiplier in _LABEL_WEIGHTS.items()
        )
        chunk_confidence = max(prob_map.values()) if prob_map else 0.0

        weighted_score += chunk_score * weight
        weighted_confidence += chunk_confidence * weight

    # Clamp to valid range
    final_score = max(-1.0, min(1.0, weighted_score))
    final_confidence = max(0.0, min(1.0, weighted_confidence))

    flag = "low_confidence" if final_confidence < _LOW_CONFIDENCE_THRESHOLD else None

    return {
        "sentiment_score": round(final_score, 6),
        "sentiment_bin": assign_bin(final_score),
        "sentiment_confidence": round(final_confidence, 6),
        "sentiment_flag": flag,
        "sentiment_chunk_count": len(chunk_scores),
    }


def _build_null_record(song_id: str) -> dict:
    """Return an all-null Schema 3 record for songs with missing lyrics."""
    return {
        "song_id": song_id,
        "sentiment_score": None,
        "sentiment_bin": None,
        "sentiment_confidence": None,
        "sentiment_flag": None,
        "sentiment_chunk_count": None,
    }


# ── Stage entry point ─────────────────────────────────────────────────────────

def run(
    config: PipelineConfig,
    lyrics_df: Optional[pd.DataFrame] = None,
    run_id: str = "unknown",
) -> pd.DataFrame:
    """
    Execute Stage 3a: sentiment inference on all songs.

    Reads S2 checkpoint if lyrics_df not provided. Checks inference cache
    per song before running model. Unloads model after full pass.
    Writes 03_sentiment.parquet.

    Args:
        config: PipelineConfig instance
        lyrics_df: optional pre-loaded S2 DataFrame
        run_id: pipeline_run_id for log records

    Returns:
        pd.DataFrame matching Schema 3.
    """
    stage = "s3_sentiment"

    if checkpoint_exists(stage, config) and not config.checkpoints.force_rerun.s3_sentiment:
        log.info("S3a: checkpoint found — skipping sentiment inference.")
        return read_checkpoint(stage, config)

    log.info("S3a: starting sentiment inference.")

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
        pipe, device = load_sentiment_model(config, device)

        for i, row in enumerate(lyrics_df.to_dict("records"), start=1):
            song_id = row["song_id"]
            lyrics = row.get("lyrics")
            lyrics_status = row.get("lyrics_status", "missing")

            log.debug(
                f"S3a: [{i}/{total}] '{row['title']}' by '{row['artist']}'"
            )

            # ── Null path: missing lyrics ──
            if lyrics_status == "missing" or not lyrics:
                records.append(_build_null_record(song_id))
                continue

            # ── Inference cache check ──
            if config.inference.cache_enabled:
                cached = inference_cache.get_inference(song_id, "sentiment")
                if cached is not None:
                    log.debug(f"S3a: inference cache hit for {song_id}")
                    records.append(cached)
                    continue

            # ── Chunk and infer ──
            chunks = chunk_lyrics(lyrics, max_tokens=_MAX_TOKENS)
            chunk_lengths = [len(c.split()) for c in chunks]

            chunk_scores = score_chunks(
                chunks=chunks,
                pipe=pipe,
                batch_size=config.inference.batch_size,
            )

            result = aggregate_scores(chunk_scores, chunk_lengths)
            result["song_id"] = song_id

            # ── Low-confidence logging ──
            if result["sentiment_flag"] == "low_confidence":
                low_conf_count += 1
                log.debug(
                    f"S3a: low_confidence for {song_id} "
                    f"(confidence={result['sentiment_confidence']:.3f})"
                )
                low_conf_log.log(
                    song_id=song_id,
                    year=row["year"],
                    title=row["title"],
                    artist=row["artist"],
                    dimension="sentiment",
                    flag_value="low_confidence",
                    pipeline_run_id=run_id,
                    score=result["sentiment_confidence"],
                    threshold=_LOW_CONFIDENCE_THRESHOLD,
                )

            # ── Write to inference cache ──
            if config.inference.cache_enabled:
                inference_cache.set_inference(song_id, "sentiment", result)

            records.append(result)

            # ── Thermal management ──
            if i % config.inference.batch_size == 0:
                time.sleep(config.inference.sleep_between_batches)

    finally:
        if pipe is not None:
            unload_model(pipe)
        inference_cache.close()

    log.info(
        f"S3a: inference complete. "
        f"total={total} low_confidence={low_conf_count}"
    )

    df = pd.DataFrame(records, columns=_SCHEMA_3_COLUMNS)
    write_checkpoint(stage, df, config)
    log.info("S3a: checkpoint written.")

    return df
