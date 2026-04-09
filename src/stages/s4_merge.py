from typing import Optional

import pandas as pd

from src.core.checkpoint import checkpoint_exists, read_checkpoint, write_checkpoint
from src.core.config import PipelineConfig
from src.core.logger import get_logger

log = get_logger("s4_merge")

# ── Schema 8 columns (locked P5) ─────────────────────────────────────────────
# All Schema 1 + Schema 2 + Schema 3 + Schema 4 + Schema 5 + Schema 6 +
# Schema 7 fields, plus record_complete, skip_reason, pipeline_run_id.
_SCHEMA_8_COLUMNS = [
    # Schema 1 — song metadata
    "song_id",
    "year",
    "rank",
    "title",
    "artist",
    "decade",
    "title_normalized",
    "artist_normalized",
    "collision_flag",
    # Schema 2 — lyrics
    "lyrics",
    "lyrics_status",
    "lyrics_source",
    "lyrics_truncated",
    "lyrics_word_count",
    "lyrics_fetched_at",
    "lyrics_cache_hit",
    # Schema 3 — sentiment
    "sentiment_score",
    "sentiment_bin",
    "sentiment_confidence",
    "sentiment_flag",
    "sentiment_chunk_count",
    # Schema 4 — mood
    "mood_primary",
    "mood_primary_confidence",
    "mood_secondary",
    "mood_secondary_confidence",
    "mood_flag",
    # Schema 5 — theme
    "theme_primary",
    "theme_primary_confidence",
    "theme_secondary",
    "theme_secondary_confidence",
    "theme_source",
    "theme_flag",
    # Schema 6 — jungian
    "jungian_primary",
    "jungian_secondary",
    "jungian_confidence",
    "jungian_evidence",
    "jungian_flag",
    "jungian_source",
    # Schema 7 — semantic
    "mtld_score",
    "imagery_density",
    "avg_line_length",
    "tfidf_keywords",
    "subject_focus",
    "semantic_vector",
    # Schema 8 — merge metadata
    "record_complete",
    "skip_reason",
    "pipeline_run_id",
]

# S3 stage keys mapped to their parquet checkpoint names.
# Used to load each analysis checkpoint independently.
_S3_STAGES = [
    "s3_sentiment",
    "s3_mood",
    "s3_theme",
    "s3_jungian",
    "s3_semantic",
]

# Analysis fields that come from S3 checkpoints rather than S2.
# These must not be duplicated from the S2 DataFrame.
_S3_FIELD_SETS = {
    "s3_sentiment": [
        "sentiment_score", "sentiment_bin", "sentiment_confidence",
        "sentiment_flag", "sentiment_chunk_count",
    ],
    "s3_mood": [
        "mood_primary", "mood_primary_confidence",
        "mood_secondary", "mood_secondary_confidence",
        "mood_flag",
    ],
    "s3_theme": [
        "theme_primary", "theme_primary_confidence",
        "theme_secondary", "theme_secondary_confidence",
        "theme_source", "theme_flag",
    ],
    "s3_jungian": [
        "jungian_primary", "jungian_secondary", "jungian_confidence",
        "jungian_evidence", "jungian_flag", "jungian_source",
    ],
    "s3_semantic": [
        "mtld_score", "imagery_density", "avg_line_length",
        "tfidf_keywords", "subject_focus", "semantic_vector",
    ],
}


# ── record_complete logic (locked P5 Schema 8) ────────────────────────────────

def compute_record_complete(df: pd.DataFrame) -> pd.Series:
    """
    Compute the record_complete boolean for each row.

    Locked definition from P5 Schema 8:
      True when ALL of:
        - lyrics_status in ["found", "truncated"]
        - sentiment_score is not null
        - mood_primary is not null
        - theme_primary is not null
        - theme_primary != "uncertain"

    Jungian and semantic fields are explicitly excluded from this
    definition per P5 — null Jungian or semantic does not make a
    record incomplete.

    Args:
        df: merged DataFrame containing all required columns

    Returns:
        Boolean pd.Series aligned with df index.
    """
    lyrics_ok = df["lyrics_status"].isin(["found", "truncated"])
    sentiment_ok = df["sentiment_score"].notna()
    mood_ok = df["mood_primary"].notna()
    theme_ok = df["theme_primary"].notna() & (df["theme_primary"] != "uncertain")
    return lyrics_ok & sentiment_ok & mood_ok & theme_ok


def validate_schema(df: pd.DataFrame) -> list[str]:
    """
    Validate that all Schema 8 columns are present in the merged DataFrame.

    Does not validate data types or value constraints — only column
    presence. Type and value validation belongs in the validation report.

    Args:
        df: merged DataFrame

    Returns:
        List of column names that are missing. Empty list if all present.
    """
    missing = [col for col in _SCHEMA_8_COLUMNS if col not in df.columns]
    return missing


def join_checkpoints(
    s2_df: pd.DataFrame,
    s3_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Left-join all S3 analysis DataFrames onto the S2 lyrics DataFrame
    using song_id as the join key.

    Each S3 DataFrame contributes only its analysis columns — song_id
    is used as the join key and dropped from S3 frames before merging
    to avoid duplicate columns.

    The join is a left join on S2: all S2 rows are retained, and S3
    fields are null for any song not present in an S3 checkpoint
    (e.g. songs skipped due to errors).

    Args:
        s2_df: Schema 2 DataFrame (base for the join)
        s3_dfs: dict mapping stage key to Schema 3–7 DataFrame

    Returns:
        Merged DataFrame containing all S2 fields plus all S3 fields.
    """
    merged = s2_df.copy()

    for stage, s3_df in s3_dfs.items():
        analysis_cols = _S3_FIELD_SETS.get(stage, [])
        cols_to_join = ["song_id"] + [
            c for c in analysis_cols if c in s3_df.columns
        ]
        s3_subset = s3_df[cols_to_join].copy()

        merged = merged.merge(
            s3_subset,
            on="song_id",
            how="left",
        )
        log.debug(f"S4: joined {stage} ({len(s3_subset)} records)")

    return merged


# ── Stage entry point ─────────────────────────────────────────────────────────

def run(
    config: PipelineConfig,
    s2_df: Optional[pd.DataFrame] = None,
    s3_dfs: Optional[dict[str, pd.DataFrame]] = None,
    run_id: str = "unknown",
) -> pd.DataFrame:
    """
    Execute Stage 4: join all S3 outputs onto the S2 lyrics DataFrame.

    Reads all required checkpoints if DataFrames not provided. Computes
    record_complete per P5 locked definition. Assigns pipeline_run_id
    to every record. Writes 04_merged.parquet.

    Null propagation: all S3 analysis fields are nullable. Songs where
    an S3 stage produced no output (e.g. missing lyrics, API failure)
    carry null analysis fields — they are retained in the merged output
    with record_complete=False.

    Args:
        config: PipelineConfig instance
        s2_df: optional pre-loaded S2 DataFrame; loaded from checkpoint
               if not provided
        s3_dfs: optional dict mapping stage key to S3 DataFrame; each
                missing key is loaded from its checkpoint
        run_id: pipeline_run_id assigned to every output record

    Returns:
        pd.DataFrame matching Schema 8.
    """
    stage = "s4_merge"

    if checkpoint_exists(stage, config) and not config.checkpoints.force_rerun.s4_merge:
        log.info("S4: checkpoint found — skipping merge.")
        return read_checkpoint(stage, config)

    log.info("S4: starting merge.")

    # ── Load S2 ──
    if s2_df is None:
        s2_df = read_checkpoint("s2_lyrics", config)

    # ── Load S3 checkpoints ──
    loaded_s3: dict[str, pd.DataFrame] = {}
    for s3_stage in _S3_STAGES:
        if s3_dfs and s3_stage in s3_dfs:
            loaded_s3[s3_stage] = s3_dfs[s3_stage]
        else:
            try:
                loaded_s3[s3_stage] = read_checkpoint(s3_stage, config)
                log.debug(f"S4: loaded {s3_stage} checkpoint "
                          f"({len(loaded_s3[s3_stage])} records)")
            except FileNotFoundError:
                log.warning(
                    f"S4: checkpoint for {s3_stage} not found — "
                    f"analysis fields will be null for all records."
                )
                loaded_s3[s3_stage] = pd.DataFrame(
                    {"song_id": s2_df["song_id"]}
                )

    # ── Join all checkpoints ──
    merged = join_checkpoints(s2_df=s2_df, s3_dfs=loaded_s3)

    # ── Ensure all Schema 8 analysis columns exist (null if absent) ──
    for col in _SCHEMA_8_COLUMNS:
        if col not in merged.columns:
            merged[col] = None

    # ── Compute record_complete ──
    merged["record_complete"] = compute_record_complete(merged)

    # ── skip_reason: null unless populated externally ──
    if "skip_reason" not in merged.columns:
        merged["skip_reason"] = None

    # ── Assign pipeline_run_id ──
    merged["pipeline_run_id"] = run_id

    # ── Enforce Schema 8 column order ──
    merged = merged[_SCHEMA_8_COLUMNS]

    # ── Validate schema ──
    missing_cols = validate_schema(merged)
    if missing_cols:
        log.error(
            f"S4: schema validation failed — missing columns: {missing_cols}"
        )
    else:
        log.info("S4: schema validation passed — all Schema 8 columns present.")

    complete_count = int(merged["record_complete"].sum())
    total = len(merged)
    log.info(
        f"S4: merge complete. "
        f"total={total} "
        f"record_complete={complete_count} "
        f"({100 * complete_count // total if total else 0}%)"
    )

    write_checkpoint(stage, merged, config)
    log.info("S4: checkpoint written.")

    return merged
