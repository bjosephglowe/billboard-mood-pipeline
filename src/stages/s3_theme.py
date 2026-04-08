import time
from typing import Optional

import pandas as pd

from src.core.cache import get_inference_cache
from src.core.checkpoint import checkpoint_exists, read_checkpoint, write_checkpoint
from src.core.config import PipelineConfig
from src.core.logger import get_logger, get_low_confidence_logger
from src.core.models import get_device, load_theme_model, unload_model
from src.core.text_utils import chunk_lyrics
from src.prompts.jungian_theme import get_permitted_themes

log = get_logger("s3_theme")

# ── Constants (locked P2 Section 4) ──────────────────────────────────────────

# Theme taxonomy labels — sourced from prompts module to stay in sync.
_THEME_LABELS = get_permitted_themes()

# Top-k themes retained per song (locked P2).
_TOP_K = 2

# Schema 5 columns (locked P5)
_SCHEMA_5_COLUMNS = [
    "song_id",
    "theme_primary",
    "theme_primary_confidence",
    "theme_secondary",
    "theme_secondary_confidence",
    "theme_source",
    "theme_flag",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def classify_song(
    lyrics: str,
    labels: list[str],
    pipe,
    threshold: float,
) -> dict:
    """
    Run zero-shot theme classification on a single song's lyrics.

    Chunks lyrics, runs inference on each chunk, then averages scores
    across chunks for each label to produce song-level confidence scores.

    Args:
        lyrics: full lyrics string
        labels: list of theme label strings to classify against
        pipe: loaded HuggingFace zero-shot-classification pipeline
        threshold: minimum confidence for a label to be retained

    Returns:
        Dict mapping theme_label -> mean_confidence across chunks.
        All label scores included regardless of threshold — threshold
        filtering is applied downstream in select_top_k.
    """
    chunks = chunk_lyrics(lyrics, max_tokens=512)
    if not chunks:
        return {label: 0.0 for label in labels}

    # Accumulate scores across chunks
    label_scores: dict[str, list[float]] = {label: [] for label in labels}

    for chunk in chunks:
        result = pipe(chunk, candidate_labels=labels, truncation=True)
        for label, score in zip(result["labels"], result["scores"]):
            if label in label_scores:
                label_scores[label].append(score)

    # Mean score per label across all chunks
    mean_scores = {
        label: (sum(scores) / len(scores) if scores else 0.0)
        for label, scores in label_scores.items()
    }

    return mean_scores


def select_top_k(
    scores: dict,
    k: int,
    threshold: float,
) -> tuple[Optional[str], Optional[float], Optional[str], Optional[float]]:
    """
    Select primary and secondary theme labels from a scores dict.

    Only labels clearing the confidence threshold are eligible.
    If no label clears the threshold, returns ("uncertain", None, None, None).

    Args:
        scores: dict mapping theme_label -> confidence float
        k: maximum labels to retain (locked at 2)
        threshold: minimum confidence for label retention

    Returns:
        Tuple of (primary, primary_conf, secondary, secondary_conf).
        secondary and secondary_conf are None if fewer than 2 labels
        clear the threshold.
    """
    eligible = [
        (label, conf) for label, conf in scores.items()
        if conf >= threshold
    ]
    eligible.sort(key=lambda x: x[1], reverse=True)

    if not eligible:
        return "uncertain", None, None, None

    primary, primary_conf = eligible[0]
    primary_conf = round(primary_conf, 6)

    secondary: Optional[str] = None
    secondary_conf: Optional[float] = None

    if len(eligible) >= 2:
        secondary, secondary_conf = eligible[1]
        secondary_conf = round(secondary_conf, 6)

    return primary, primary_conf, secondary, secondary_conf


def build_theme_record(
    song_id: str,
    primary: Optional[str],
    primary_conf: Optional[float],
    secondary: Optional[str],
    secondary_conf: Optional[float],
    source: str,
    flag: Optional[str],
) -> dict:
    """
    Construct a Schema 5 theme record dict.

    Args:
        song_id: 16-char hex song identifier
        primary: primary theme label or "uncertain"
        primary_conf: primary confidence float or None
        secondary: secondary theme label or None
        secondary_conf: secondary confidence float or None
        source: "minilm" | "haiku" | "uncertain"
        flag: "low_confidence" | "haiku_fallback" | None

    Returns:
        Dict matching Schema 5.
    """
    return {
        "song_id": song_id,
        "theme_primary": primary,
        "theme_primary_confidence": primary_conf,
        "theme_secondary": secondary,
        "theme_secondary_confidence": secondary_conf,
        "theme_source": source,
        "theme_flag": flag,
    }


def _build_null_record(song_id: str) -> dict:
    """Return an all-null Schema 5 record for songs with missing lyrics."""
    return {
        "song_id": song_id,
        "theme_primary": None,
        "theme_primary_confidence": None,
        "theme_secondary": None,
        "theme_secondary_confidence": None,
        "theme_source": None,
        "theme_flag": None,
    }


def _apply_haiku_fallback(
    uncertain_song_ids: list[str],
    lyrics_lookup: dict[str, dict],
    theme_records: dict[str, dict],
    config: PipelineConfig,
    inference_cache,
    run_id: str,
) -> None:
    """
    Attempt Haiku API theme fallback for songs where MiniLM returned uncertain.

    Calls call_haiku_for_theme_fallback from s3_jungian for each uncertain
    song. That function returns a partial Schema 5 dict directly — fields
    are at the top level (theme_primary, theme_primary_confidence, etc.),
    not nested under a theme_fallback key.

    Updates theme_records in-place with Haiku results where available.

    This function is an intentional cross-stage call to the public
    call_haiku_for_theme_fallback interface in s3_jungian. S3d will
    subsequently run a full Jungian pass on all songs independently.

    Args:
        uncertain_song_ids: list of song_ids needing Haiku fallback
        lyrics_lookup: dict mapping song_id -> full row dict from lyrics_df
        theme_records: dict mapping song_id -> Schema 5 record (mutated)
        config: PipelineConfig instance
        inference_cache: InferenceCache instance
        run_id: pipeline_run_id for log records
    """
    if not uncertain_song_ids or not config.theme.haiku_fallback_enabled:
        log.info(
            f"S3c: Haiku fallback {'disabled in config' if not config.theme.haiku_fallback_enabled else 'skipped — no uncertain songs'}."
        )
        return

    # Deferred import to avoid circular dependency at module level.
    # s3_jungian depends on prompts/jungian_theme, not on s3_theme.
    from src.stages.s3_jungian import call_haiku_for_theme_fallback

    log.info(
        f"S3c: requesting Haiku theme fallback for "
        f"{len(uncertain_song_ids)} uncertain songs."
    )

    fallback_success = 0
    low_conf_log = get_low_confidence_logger(config.logging.low_confidence_log)

    for song_id in uncertain_song_ids:
        row = lyrics_lookup.get(song_id)
        if not row:
            log.debug(f"S3c: no row found for {song_id} — skipping fallback.")
            continue

        lyrics = row.get("lyrics", "")
        if not lyrics:
            log.debug(f"S3c: no lyrics for {song_id} — skipping fallback.")
            continue

        # Check inference cache for a prior Haiku theme result
        if config.inference.cache_enabled:
            cached = inference_cache.get_inference(song_id, "theme")
            if cached is not None and cached.get("theme_source") == "haiku":
                log.debug(f"S3c: Haiku theme cache hit for {song_id}")
                theme_records[song_id] = cached
                fallback_success += 1
                continue

        # call_haiku_for_theme_fallback returns a partial Schema 5 dict
        # with keys: song_id, theme_primary, theme_primary_confidence,
        # theme_secondary, theme_secondary_confidence, theme_source,
        # theme_flag — or None if Haiku returned no valid theme.
        result = call_haiku_for_theme_fallback(
            song_id=song_id,
            lyrics=lyrics,
            title=row.get("title", ""),
            artist=row.get("artist", ""),
            config=config,
            run_id=run_id,
        )

        if result is not None and result.get("theme_primary") is not None:
            theme_records[song_id] = result
            fallback_success += 1

            if config.inference.cache_enabled:
                inference_cache.set_inference(song_id, "theme", result)

            log.debug(
                f"S3c: Haiku fallback succeeded for {song_id} "
                f"— theme={result['theme_primary']}"
            )
        else:
            log.debug(
                f"S3c: Haiku fallback returned null theme for {song_id} "
                f"— retaining uncertain."
            )
            low_conf_log.log(
                song_id=song_id,
                year=row.get("year", 0),
                title=row.get("title", ""),
                artist=row.get("artist", ""),
                dimension="theme",
                flag_value="low_confidence",
                pipeline_run_id=run_id,
            )

    log.info(
        f"S3c: Haiku fallback complete. "
        f"resolved={fallback_success} "
        f"still_uncertain={len(uncertain_song_ids) - fallback_success}"
    )


# ── Stage entry point ─────────────────────────────────────────────────────────

def run(
    config: PipelineConfig,
    lyrics_df: Optional[pd.DataFrame] = None,
    run_id: str = "unknown",
) -> pd.DataFrame:
    """
    Execute Stage 3c: theme classification on all songs.

    Pass 1 — MiniLM zero-shot inference on all songs with lyrics.
    Pass 2 — Haiku API fallback for songs where MiniLM confidence
              was below config.theme.min_confidence.

    Reads S2 checkpoint if lyrics_df not provided. Checks inference cache
    per song before running model. Unloads MiniLM model before Haiku
    fallback to free memory. Writes 03_theme.parquet.

    Args:
        config: PipelineConfig instance
        lyrics_df: optional pre-loaded S2 DataFrame
        run_id: pipeline_run_id for log records

    Returns:
        pd.DataFrame matching Schema 5.
    """
    stage = "s3_theme"

    if checkpoint_exists(stage, config) and not config.checkpoints.force_rerun.s3_theme:
        log.info("S3c: checkpoint found — skipping theme inference.")
        return read_checkpoint(stage, config)

    log.info("S3c: starting theme classification.")

    if lyrics_df is None:
        lyrics_df = read_checkpoint("s2_lyrics", config)

    device = get_device(config)
    inference_cache = get_inference_cache(config)
    threshold = config.theme.min_confidence

    pipe = None
    theme_records: dict[str, dict] = {}
    uncertain_song_ids: list[str] = []
    lyrics_lookup: dict[str, dict] = {}
    total = len(lyrics_df)
    low_conf_log = get_low_confidence_logger(config.logging.low_confidence_log)

    rows = lyrics_df.to_dict("records")

    # Build full-row lyrics lookup for Haiku fallback pass
    for row in rows:
        lyrics_lookup[row["song_id"]] = row

    try:
        pipe, device = load_theme_model(config, device)

        for i, row in enumerate(rows, start=1):
            song_id = row["song_id"]
            lyrics = row.get("lyrics")
            lyrics_status = row.get("lyrics_status", "missing")

            log.debug(
                f"S3c: [{i}/{total}] '{row['title']}' by '{row['artist']}'"
            )

            # ── Null path: missing lyrics ──
            if lyrics_status == "missing" or not lyrics:
                theme_records[song_id] = _build_null_record(song_id)
                continue

            # ── Inference cache check ──
            if config.inference.cache_enabled:
                cached = inference_cache.get_inference(song_id, "theme")
                if cached is not None:
                    log.debug(f"S3c: inference cache hit for {song_id}")
                    theme_records[song_id] = cached
                    continue

            # ── MiniLM zero-shot classification ──
            scores = classify_song(
                lyrics=lyrics,
                labels=_THEME_LABELS,
                pipe=pipe,
                threshold=threshold,
            )

            primary, primary_conf, secondary, secondary_conf = select_top_k(
                scores=scores,
                k=_TOP_K,
                threshold=threshold,
            )

            if primary == "uncertain":
                # Queue for Haiku fallback — low_confidence logged in
                # _apply_haiku_fallback if fallback also fails.
                record = build_theme_record(
                    song_id=song_id,
                    primary="uncertain",
                    primary_conf=None,
                    secondary=None,
                    secondary_conf=None,
                    source="uncertain",
                    flag="low_confidence",
                )
                theme_records[song_id] = record
                uncertain_song_ids.append(song_id)
                log.debug(
                    f"S3c: uncertain for {song_id} — queued for Haiku fallback."
                )
            else:
                record = build_theme_record(
                    song_id=song_id,
                    primary=primary,
                    primary_conf=primary_conf,
                    secondary=secondary,
                    secondary_conf=secondary_conf,
                    source="minilm",
                    flag=None,
                )
                theme_records[song_id] = record

                if config.inference.cache_enabled:
                    inference_cache.set_inference(song_id, "theme", record)

            # ── Thermal management ──
            if i % config.inference.batch_size == 0:
                time.sleep(config.inference.sleep_between_batches)

    finally:
        if pipe is not None:
            unload_model(pipe)

    log.info(
        f"S3c: MiniLM pass complete. "
        f"total={total} uncertain={len(uncertain_song_ids)}"
    )

    # ── Pass 2: Haiku fallback ────────────────────────────────────────────────
    _apply_haiku_fallback(
        uncertain_song_ids=uncertain_song_ids,
        lyrics_lookup=lyrics_lookup,
        theme_records=theme_records,
        config=config,
        inference_cache=inference_cache,
        run_id=run_id,
    )

    inference_cache.close()

    uncertain_final = sum(
        1 for r in theme_records.values()
        if r.get("theme_primary") == "uncertain"
    )
    log.info(
        f"S3c: theme classification complete. "
        f"total={total} uncertain_final={uncertain_final}"
    )

    # Preserve original row order from lyrics_df
    records_ordered = [
        theme_records.get(row["song_id"], _build_null_record(row["song_id"]))
        for row in rows
    ]

    df = pd.DataFrame(records_ordered, columns=_SCHEMA_5_COLUMNS)
    write_checkpoint(stage, df, config)
    log.info("S3c: checkpoint written.")

    return df
