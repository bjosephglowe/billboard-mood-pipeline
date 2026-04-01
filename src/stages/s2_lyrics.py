import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from src.core.cache import LyricsCache, get_lyrics_cache
from src.core.checkpoint import checkpoint_exists, read_checkpoint, write_checkpoint
from src.core.config import PipelineConfig
from src.core.logger import get_logger, get_missing_lyrics_logger

log = get_logger("s2_lyrics")

# ── Schema 2 columns (locked P5) ─────────────────────────────────────────────
_SCHEMA_2_NEW_COLUMNS = [
    "lyrics",
    "lyrics_status",
    "lyrics_source",
    "lyrics_truncated",
    "lyrics_word_count",
    "lyrics_fetched_at",
    "lyrics_cache_hit",
]


# ── Genius fetch ──────────────────────────────────────────────────────────────

def fetch_genius(
    title: str,
    artist: str,
    token: str,
    sleep_time: float,
) -> tuple[Optional[str], Optional[str]]:
    """
    Fetch lyrics from Genius API via the lyricsgenius library.

    Strips section annotations (e.g. [Chorus]) and excess whitespace from
    the returned lyrics text. Genius is the primary lyrics source.

    Args:
        title: song title
        artist: artist name
        token: Genius client access token
        sleep_time: seconds to sleep after each request (rate limiting)

    Returns:
        Tuple of (lyrics_text, error_message).
        lyrics_text is None on failure; error_message is None on success.
    """
    try:
        import lyricsgenius

        genius = lyricsgenius.Genius(
            token,
            verbose=False,
            remove_section_headers=True,
            skip_non_songs=True,
            timeout=15,
        )

        song = genius.search_song(title, artist)
        time.sleep(sleep_time)

        if song is None or not song.lyrics:
            return None, "No results returned"

        lyrics = song.lyrics.strip()
        if not lyrics:
            return None, "Empty lyrics returned"

        return lyrics, None

    except Exception as exc:
        return None, str(exc)


# ── Musixmatch fetch ──────────────────────────────────────────────────────────

def fetch_musixmatch(
    title: str,
    artist: str,
    key: str,
) -> tuple[Optional[str], bool, Optional[str]]:
    """
    Fetch lyrics from the Musixmatch API.

    Musixmatch free tier returns only 30% of lyrics. Any successful
    response is flagged as truncated per the locked source strategy.

    Args:
        title: song title
        artist: artist name
        key: Musixmatch API key

    Returns:
        Tuple of (lyrics_text, truncated_flag, error_message).
        lyrics_text is None on failure.
        truncated_flag is always True on success (free tier limitation).
        error_message is None on success.
    """
    try:
        import requests

        params = {
            "q_track": title,
            "q_artist": artist,
            "apikey": key,
            "f_has_lyrics": 1,
        }

        # Search for the track first
        search_resp = requests.get(
            "https://api.musixmatch.com/ws/1.1/track.search",
            params=params,
            timeout=15,
        )
        search_resp.raise_for_status()
        search_data = search_resp.json()

        track_list = (
            search_data
            .get("message", {})
            .get("body", {})
            .get("track_list", [])
        )

        if not track_list:
            return None, False, "No track found"

        track_id = track_list[0]["track"]["track_id"]

        # Fetch lyrics for the track
        lyrics_resp = requests.get(
            "https://api.musixmatch.com/ws/1.1/track.lyrics.get",
            params={"track_id": track_id, "apikey": key},
            timeout=15,
        )
        lyrics_resp.raise_for_status()
        lyrics_data = lyrics_resp.json()

        lyrics_body = (
            lyrics_data
            .get("message", {})
            .get("body", {})
            .get("lyrics", {})
            .get("lyrics_body", "")
        )

        if not lyrics_body or not lyrics_body.strip():
            return None, False, "Empty lyrics body returned"

        # Strip Musixmatch attribution footer if present
        lyrics_text = lyrics_body.split("******* This Lyrics is NOT for Commercial use")[0]
        lyrics_text = lyrics_text.strip()

        if not lyrics_text:
            return None, False, "Lyrics empty after stripping attribution"

        # Free tier always returns truncated lyrics
        return lyrics_text, True, None

    except Exception as exc:
        return None, False, str(exc)


# ── Record builder ────────────────────────────────────────────────────────────

def build_lyrics_record(
    song_id: str,
    lyrics: Optional[str],
    source: Optional[str],
    truncated: bool,
    cache_hit: bool,
) -> dict:
    """
    Construct a Schema 2 lyrics result dict for a single song.

    Args:
        song_id: 16-char hex song identifier
        lyrics: lyrics text or None
        source: "genius" | "musixmatch" | None
        truncated: True if Musixmatch 30% truncation applies
        cache_hit: True if result served from local cache

    Returns:
        Dict with all Schema 2 lyrics fields.
    """
    word_count: Optional[int] = None
    if lyrics:
        word_count = len(lyrics.split())

    status: str
    if lyrics and truncated:
        status = "truncated"
    elif lyrics:
        status = "found"
    else:
        status = "missing"

    return {
        "song_id": song_id,
        "lyrics": lyrics,
        "lyrics_status": status,
        "lyrics_source": source,
        "lyrics_truncated": truncated,
        "lyrics_word_count": word_count,
        "lyrics_fetched_at": datetime.now(timezone.utc).isoformat(),
        "lyrics_cache_hit": cache_hit,
    }


# ── Per-song fetch orchestration ──────────────────────────────────────────────

def _fetch_song_lyrics(
    song: dict,
    config: PipelineConfig,
    cache: LyricsCache,
    missing_log,
    run_id: str,
) -> dict:
    """
    Fetch lyrics for a single song. Checks cache first. Falls back from
    Genius to Musixmatch. Logs misses.

    Args:
        song: Schema 1 song record dict
        config: PipelineConfig instance
        cache: LyricsCache instance
        missing_log: MissingLyricsWriter instance
        run_id: pipeline_run_id string for log records

    Returns:
        Dict with all Schema 2 lyrics fields for this song.
    """
    song_id = song["song_id"]
    title = song["title"]
    artist = song["artist"]

    # ── Cache check ──
    if config.lyrics.cache_enabled:
        cached = cache.get(song_id)
        if cached is not None:
            log.debug(f"Cache hit: {song_id} '{title}'")
            cached["lyrics_cache_hit"] = True
            return cached

    # ── Genius primary ──
    genius_error: Optional[str] = None
    musixmatch_error: Optional[str] = None
    genius_tried = False
    musixmatch_tried = False

    if config.genius_api_token:
        genius_tried = True
        log.debug(f"Genius fetch: '{title}' by '{artist}'")
        lyrics, genius_error = fetch_genius(
            title=title,
            artist=artist,
            token=config.genius_api_token,
            sleep_time=config.lyrics.genius_sleep_time,
        )

        if lyrics:
            record = build_lyrics_record(
                song_id=song_id,
                lyrics=lyrics,
                source="genius",
                truncated=False,
                cache_hit=False,
            )
            if config.lyrics.cache_enabled:
                cache.set(song_id, record)
            return record

        log.debug(f"Genius miss: '{title}' — {genius_error}")
    else:
        log.warning("GENIUS_API_TOKEN not set — skipping Genius fetch.")

    # ── Musixmatch fallback ──
    if config.musixmatch_api_key:
        musixmatch_tried = True
        log.debug(f"Musixmatch fallback: '{title}' by '{artist}'")
        lyrics, truncated, musixmatch_error = fetch_musixmatch(
            title=title,
            artist=artist,
            key=config.musixmatch_api_key,
        )

        if lyrics:
            log.info(
                f"Musixmatch: lyrics retrieved (truncated) for '{title}' by '{artist}'"
            )
            record = build_lyrics_record(
                song_id=song_id,
                lyrics=lyrics,
                source="musixmatch",
                truncated=truncated,
                cache_hit=False,
            )
            if config.lyrics.cache_enabled:
                cache.set(song_id, record)
            return record

        log.debug(f"Musixmatch miss: '{title}' — {musixmatch_error}")
    else:
        log.warning("MUSIXMATCH_API_KEY not set — skipping Musixmatch fallback.")

    # ── Both failed ──
    log.warning(
        f"Lyrics missing: '{title}' by '{artist}' ({song['year']}) — "
        f"Genius: {genius_error} | Musixmatch: {musixmatch_error}"
    )

    missing_log.log(
        song_id=song_id,
        year=song["year"],
        title=title,
        artist=artist,
        genius_tried=genius_tried,
        genius_error=genius_error,
        musixmatch_tried=musixmatch_tried,
        musixmatch_error=musixmatch_error,
        pipeline_run_id=run_id,
    )

    record = build_lyrics_record(
        song_id=song_id,
        lyrics=None,
        source=None,
        truncated=False,
        cache_hit=False,
    )
    return record


# ── Stage entry point ─────────────────────────────────────────────────────────

def run(
    config: PipelineConfig,
    songs_df: Optional[pd.DataFrame] = None,
    run_id: str = "unknown",
) -> pd.DataFrame:
    """
    Execute Stage 2: lyrics fetch for all songs in the S1 checkpoint.

    Reads S1 checkpoint if songs_df not provided. Checks lyrics cache
    before making any API call. Falls back from Genius to Musixmatch.
    Logs all misses to missing_lyrics.jsonl. Writes 02_lyrics.parquet.

    Args:
        config: PipelineConfig instance
        songs_df: optional pre-loaded S1 DataFrame; loaded from checkpoint
                  if not provided
        run_id: pipeline_run_id for log records

    Returns:
        pd.DataFrame matching Schema 2 (all Schema 1 fields plus lyrics fields).
    """
    stage = "s2_lyrics"

    if checkpoint_exists(stage, config) and not config.checkpoints.force_rerun.s2_lyrics:
        log.info("S2: checkpoint found — skipping lyrics fetch.")
        return read_checkpoint(stage, config)

    log.info("S2: starting lyrics fetch.")

    if songs_df is None:
        songs_df = read_checkpoint("s1_ingest", config)

    missing_log = get_missing_lyrics_logger(config.logging.missing_lyrics_log)
    cache = get_lyrics_cache(config)

    lyrics_records = []
    total = len(songs_df)

    for i, row in enumerate(songs_df.to_dict("records"), start=1):
        log.debug(
            f"S2: [{i}/{total}] '{row['title']}' by '{row['artist']}' ({row['year']})"
        )
        result = _fetch_song_lyrics(
            song=row,
            config=config,
            cache=cache,
            missing_log=missing_log,
            run_id=run_id,
        )
        lyrics_records.append(result)

    lyrics_df = pd.DataFrame(lyrics_records)

    # Merge Schema 1 fields with Schema 2 lyrics fields on song_id
    merged = songs_df.merge(
        lyrics_df.drop(columns=["song_id"], errors="ignore"),
        left_index=True,
        right_index=True,
    )

    # Ensure song_id from songs_df is the canonical one
    # (lyrics_df also has song_id but we keep songs_df version)
    for col in _SCHEMA_2_NEW_COLUMNS:
        if col not in merged.columns:
            merged[col] = None

    cache.close()

    found = (merged["lyrics_status"] == "found").sum()
    truncated = (merged["lyrics_status"] == "truncated").sum()
    missing = (merged["lyrics_status"] == "missing").sum()

    log.info(
        f"S2: lyrics fetch complete. "
        f"found={found} truncated={truncated} missing={missing} "
        f"total={total}"
    )

    write_checkpoint(stage, merged, config)
    log.info("S2: checkpoint written.")

    return merged
