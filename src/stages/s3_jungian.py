import time
from typing import Optional

import pandas as pd

from src.core.cache import get_inference_cache
from src.core.checkpoint import checkpoint_exists, read_checkpoint, write_checkpoint
from src.core.config import PipelineConfig
from src.core.logger import get_logger, get_low_confidence_logger
from src.prompts.jungian_theme import build_prompt, parse_response

log = get_logger("s3_jungian")

# Import the parse exception — name may be PromptParseError or ValueError
# depending on which version of jungian_theme.py is on disk. We catch both
# to be safe across versions.
try:
    from src.prompts.jungian_theme import PromptParseError as _ParseError
except ImportError:
    _ParseError = ValueError

# ── Schema 6 columns (locked P5) ─────────────────────────────────────────────
_SCHEMA_6_COLUMNS = [
    "song_id",
    "jungian_primary",
    "jungian_secondary",
    "jungian_confidence",
    "jungian_evidence",
    "jungian_flag",
    "jungian_source",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_null_record(song_id: str, flag: str = "api_unavailable") -> dict:
    """
    Return an all-null Schema 6 record.

    Args:
        song_id: 16-char hex song identifier
        flag: jungian_flag value — "api_unavailable" or
              "insufficient_evidence"
    """
    return {
        "song_id": song_id,
        "jungian_primary": None,
        "jungian_secondary": None,
        "jungian_confidence": None,
        "jungian_evidence": None,
        "jungian_flag": flag,
        "jungian_source": None,
    }


def _build_record_from_parsed(song_id: str, parsed: dict) -> dict:
    """
    Build a Schema 6 record from a validated parse_response output.

    Args:
        song_id: 16-char hex song identifier
        parsed: validated dict from parse_response()

    Returns:
        Dict matching Schema 6.
    """
    j = parsed["jungian"]
    return {
        "song_id": song_id,
        "jungian_primary": j["primary"],
        "jungian_secondary": j["secondary"],
        "jungian_confidence": j["confidence"],
        "jungian_evidence": j["evidence"],
        "jungian_flag": j["flag"],
        "jungian_source": "haiku" if j["primary"] is not None or j["flag"] is not None else None,
    }


def call_haiku(
    prompt: str,
    config: PipelineConfig,
) -> Optional[str]:
    """
    Send a prompt to the Claude Haiku API and return the raw response text.

    Implements retry logic per config.jungian.max_retries with
    config.jungian.retry_sleep seconds between attempts.

    Args:
        prompt: fully constructed prompt string
        config: PipelineConfig instance

    Returns:
        Raw response string from the model, or None if all retries fail.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    last_error: Optional[Exception] = None

    for attempt in range(1, config.jungian.max_retries + 1):
        try:
            message = client.messages.create(
                model=config.jungian.haiku_model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            for block in message.content:
                if hasattr(block, "text"):
                    return block.text
            log.warning(
                f"Haiku response contained no text block on attempt {attempt}."
            )
            return None

        except Exception as exc:
            last_error = exc
            if attempt < config.jungian.max_retries:
                log.warning(
                    f"Haiku API call failed (attempt {attempt}/{config.jungian.max_retries}): "
                    f"{exc}. Retrying in {config.jungian.retry_sleep}s."
                )
                time.sleep(config.jungian.retry_sleep)
            else:
                log.error(
                    f"Haiku API call failed after {config.jungian.max_retries} attempts: {exc}"
                )

    return None


def _process_song(
    song_id: str,
    lyrics: str,
    title: str,
    artist: str,
    request_theme: bool,
    config: PipelineConfig,
    run_id: str,
) -> tuple[dict, Optional[dict]]:
    """
    Run the full Haiku call for a single song.

    Builds the prompt, calls the API, parses the response. Returns a
    Schema 6 Jungian record and optionally a theme fallback result dict.

    Args:
        song_id: 16-char hex song identifier
        lyrics: full lyric text
        title: song title for prompt context
        artist: artist name for prompt context
        request_theme: whether to activate theme fallback in the prompt
        config: PipelineConfig instance
        run_id: pipeline_run_id for logging

    Returns:
        Tuple of (jungian_record, theme_result_or_None).
        theme_result is a partial Schema 5 dict if request_theme=True
        and a valid theme was returned, otherwise None.
    """
    prompt = build_prompt(
        lyrics=lyrics,
        song_title=title,
        artist=artist,
        request_theme_fallback=request_theme,
    )

    raw_response = call_haiku(prompt=prompt, config=config)

    if raw_response is None:
        log.error(f"Haiku returned no response for song_id={song_id}")
        return _build_null_record(song_id, flag="api_unavailable"), None

    try:
        parsed = parse_response(raw_response)
    except (ValueError, Exception) as exc:
        # Catches both ValueError and any custom PromptParseError subclass
        log.error(
            f"Haiku parse failure for song_id={song_id}: {exc}\n"
            f"Raw response (first 500 chars): {raw_response[:500]}"
        )
        return _build_null_record(song_id, flag="api_unavailable"), None

    jungian_record = _build_record_from_parsed(song_id, parsed)

    # Always set jungian_source to "haiku" — this stage always uses Haiku
    jungian_record["jungian_source"] = "haiku"

    theme_result: Optional[dict] = None
    if request_theme:
        tf = parsed.get("theme_fallback", {})
        t_primary = tf.get("primary")
        if t_primary is not None:
            theme_result = {
                "song_id": song_id,
                "theme_primary": t_primary,
                "theme_primary_confidence": tf.get("primary_confidence", 0.0),
                "theme_secondary": tf.get("secondary"),
                "theme_secondary_confidence": tf.get("secondary_confidence", 0.0),
                "theme_source": "haiku",
                "theme_flag": "haiku_fallback",
            }

    return jungian_record, theme_result


def handle_api_failure(song_id: str) -> dict:
    """
    Return an all-null Schema 6 record for a complete API failure.

    Public so s3_theme.py can call it when Haiku is unavailable.

    Args:
        song_id: 16-char hex song identifier

    Returns:
        Schema 6 dict with jungian_flag: api_unavailable.
    """
    return _build_null_record(song_id, flag="api_unavailable")


# ── Theme fallback entry point (called by s3_theme) ───────────────────────────

def call_haiku_for_theme_fallback(
    song_id: str,
    lyrics: str,
    title: str,
    artist: str,
    config: PipelineConfig,
    run_id: str,
) -> Optional[dict]:
    """
    Call Haiku for theme fallback only — no Jungian result is returned.

    Called by s3_theme._apply_haiku_fallback() for songs where MiniLM
    confidence was below threshold. Returns a partial Schema 5 theme
    dict if a valid theme is returned, otherwise None.

    Args:
        song_id: 16-char hex song identifier
        lyrics: full lyric text
        title: song title
        artist: artist name
        config: PipelineConfig instance
        run_id: pipeline_run_id for logging

    Returns:
        Partial Schema 5 dict with theme fields, or None if Haiku
        returns no valid theme label.
    """
    _, theme_result = _process_song(
        song_id=song_id,
        lyrics=lyrics,
        title=title,
        artist=artist,
        request_theme=True,
        config=config,
        run_id=run_id,
    )
    return theme_result


# ── Stage entry point ─────────────────────────────────────────────────────────

def run(
    config: PipelineConfig,
    lyrics_df: Optional[pd.DataFrame] = None,
    theme_fallback_song_ids: Optional[list[str]] = None,
    run_id: str = "unknown",
) -> pd.DataFrame:
    """
    Execute Stage 3d: Jungian archetype analysis for all songs.

    Reads S2 checkpoint if lyrics_df not provided. Checks inference cache
    per song. Calls Haiku for each song with lyrics. If theme_fallback_song_ids
    is provided, activates theme fallback in the prompt for those songs.
    Writes 03_jungian.parquet.

    Args:
        config: PipelineConfig instance
        lyrics_df: optional pre-loaded S2 DataFrame
        theme_fallback_song_ids: optional list of song_ids needing theme
                                  fallback in addition to Jungian analysis
        run_id: pipeline_run_id for log records

    Returns:
        pd.DataFrame matching Schema 6.
    """
    stage = "s3_jungian"

    if checkpoint_exists(stage, config) and not config.checkpoints.force_rerun.s3_jungian:
        log.info("S3d: checkpoint found — skipping Jungian analysis.")
        return read_checkpoint(stage, config)

    log.info("S3d: starting Jungian analysis.")

    if lyrics_df is None:
        lyrics_df = read_checkpoint("s2_lyrics", config)

    inference_cache = get_inference_cache(config)
    low_conf_log = get_low_confidence_logger(config.logging.low_confidence_log)
    fallback_ids = set(theme_fallback_song_ids or [])

    records: list[dict] = []
    total = len(lyrics_df)
    insufficient_count = 0
    api_fail_count = 0
    speculative_count = 0

    for i, row in enumerate(lyrics_df.to_dict("records"), start=1):
        song_id = row["song_id"]
        lyrics = row.get("lyrics")
        lyrics_status = row.get("lyrics_status", "missing")

        log.debug(
            f"S3d: [{i}/{total}] '{row['title']}' by '{row['artist']}'"
        )

        # ── Null path: missing lyrics ──
        if lyrics_status == "missing" or not lyrics:
            records.append(_build_null_record(song_id, flag="insufficient_evidence"))
            continue

        # ── Inference cache check ──
        if config.inference.cache_enabled:
            cached = inference_cache.get_inference(song_id, "jungian")
            if cached is not None:
                # Only trust cache hits that represent completed Haiku analysis.
                # api_unavailable records from failed runs must not block a fresh
                # attempt — discard them and fall through to a new API call.
                if cached.get("jungian_flag") != "api_unavailable":
                    log.debug(f"S3d: inference cache hit for {song_id}")
                    records.append(cached)
                    continue
                else:
                    log.debug(
                        f"S3d: discarding stale api_unavailable cache entry "
                        f"for {song_id} — retrying."
                    )

        # ── Haiku call ──
        request_theme = song_id in fallback_ids
        jungian_record, _ = _process_song(
            song_id=song_id,
            lyrics=lyrics,
            title=row.get("title", ""),
            artist=row.get("artist", ""),
            request_theme=request_theme,
            config=config,
            run_id=run_id,
        )

        # ── Flag accounting ──
        flag = jungian_record.get("jungian_flag")
        if flag == "api_unavailable":
            api_fail_count += 1
        elif flag == "insufficient_evidence":
            insufficient_count += 1
        elif flag == "speculative":
            speculative_count += 1
            low_conf_log.log(
                song_id=song_id,
                year=row["year"],
                title=row["title"],
                artist=row["artist"],
                dimension="jungian",
                flag_value="speculative",
                pipeline_run_id=run_id,
            )

        # ── Write to inference cache ──
        if config.inference.cache_enabled:
            inference_cache.set_inference(song_id, "jungian", jungian_record)

        records.append(jungian_record)

        if i % config.inference.batch_size == 0:
            time.sleep(config.inference.sleep_between_batches)

    inference_cache.close()

    log.info(
        f"S3d: Jungian analysis complete. "
        f"total={total} "
        f"insufficient_evidence={insufficient_count} "
        f"speculative={speculative_count} "
        f"api_unavailable={api_fail_count}"
    )

    df = pd.DataFrame(records, columns=_SCHEMA_6_COLUMNS)
    write_checkpoint(stage, df, config)
    log.info("S3d: checkpoint written.")

    return df
