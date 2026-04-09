import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

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

_GENIUS_HEADERS = {
    "User-Agent": "curl/7.88.1",
    "Accept": "application/json",
}


# ── Genius fetch (direct REST API — no lyricsgenius library) ──────────────────

def fetch_genius(
    title: str,
    artist: str,
    token: str,
    sleep_time: float,
) -> tuple[Optional[str], Optional[str]]:
    """
    Fetch lyrics from Genius using the authenticated search API
    then the embed.js endpoint for clean lyrics extraction.

    Uses embed.js rather than scraping the song page directly —
    the embed endpoint returns lyrics in a structured JS payload
    that can be decoded without interference from metadata containers.
    """
    import re as _re

    auth_headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "curl/7.88.1",
        "Accept": "application/json",
    }

    # Step 1: authenticated search to get song_id and url
    try:
        search_resp = requests.get(
            "https://api.genius.com/search",
            params={"q": f"{title} {artist}"},
            headers=auth_headers,
            timeout=15,
        )
        time.sleep(sleep_time)
        search_resp.raise_for_status()
        hits = search_resp.json().get("response", {}).get("hits", [])
    except Exception as exc:
        return None, f"Genius search failed: {exc}"

    if not hits:
        return None, "No search results returned"

    # Filter to song-type hits only
    song_hits = [h for h in hits if h.get("type") == "song"]
    if not song_hits:
        return None, "No song-type results returned"

    # Match artist name — strip special chars for loose comparison
    artist_clean = _re.sub(r"[^a-z0-9 ]", "", artist.lower()).strip()
    artist_words = [w for w in artist_clean.split() if len(w) > 2]

    selected = None
    for hit in song_hits[:5]:
        result = hit.get("result", {})
        result_artist = result.get("primary_artist", {}).get("name", "").lower()
        result_artist_clean = _re.sub(r"[^a-z0-9 ]", "", result_artist)
        if any(w in result_artist_clean for w in artist_words):
            selected = result
            break

    if not selected:
        selected = song_hits[0]["result"]

    song_id = selected.get("id")
    if not song_id:
        return None, "No song ID in search result"

    # Step 2: fetch embed.js — contains lyrics as escaped HTML in JSON.parse()
    try:
        embed_resp = requests.get(
            f"https://genius.com/songs/{song_id}/embed.js",
            headers={"User-Agent": "curl/7.88.1"},
            timeout=15,
        )
        embed_resp.raise_for_status()
    except Exception as exc:
        return None, f"Genius embed fetch failed: {exc}"

    lyrics = _extract_lyrics_from_embed(embed_resp.text)
    if not lyrics:
        return None, "Could not extract lyrics from embed"

    return lyrics, None


def _extract_lyrics_from_embed(js_text: str) -> Optional[str]:
    """
    Extract and clean lyrics from the Genius embed.js JavaScript payload.

    The embed JS wraps HTML in a JSON.parse('...') call with multiple
    levels of escaping. We decode with unicode_escape then parse the
    resulting HTML with BeautifulSoup to find the rg_embed_body div.
    """
    import re as _re

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return None

    # Extract the single-quoted string argument to JSON.parse()
    match = _re.search(r"JSON\.parse\('(.*?)'\)", js_text, _re.DOTALL)
    if not match:
        return None

    raw = match.group(1)

    # Decode all escape levels using unicode_escape
    try:
        decoded = raw.encode("utf-8").decode("unicode_escape")
    except Exception:
        return None

    # Parse the decoded HTML
    soup = BeautifulSoup(decoded, "html.parser")

    # Find the lyrics body div — try id first, then class
    body = soup.find("div", id=_re.compile("rg_embed_body"))
    if not body:
        body = soup.find("div", class_=_re.compile("rg_embed_body"))
    if not body:
        return None

    # Replace <br> tags with newlines before text extraction
    for br in body.find_all("br"):
        br.replace_with("\n")

    # Extract text
    lyrics = body.get_text(separator="\n")

    # Clean up: remove escaped forward slashes, excess blank lines
    lyrics = lyrics.replace("\\/", "/")
    lyrics = _re.sub(r"\n{3,}", "\n\n", lyrics)
    lyrics = lyrics.strip()

    return lyrics if len(lyrics) > 50 else None


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
    """
    try:
        search_resp = requests.get(
            "https://api.musixmatch.com/ws/1.1/track.search",
            params={
                "q_track": title,
                "q_artist": artist,
                "apikey": key,
                "f_has_lyrics": 1,
                "s_track_rating": "desc",
            },
            timeout=15,
        )
        search_resp.raise_for_status()
        search_data = search_resp.json()

        body = search_data.get("message", {}).get("body", {})

        # Handle both dict and list response shapes
        if isinstance(body, list):
            return None, False, "Unexpected list in response body"

        track_list = body.get("track_list", [])
        if not track_list:
            return None, False, "No track found"

        # Each item in track_list is {"track": {...}}
        first = track_list[0]
        if isinstance(first, dict) and "track" in first:
            track_id = first["track"]["track_id"]
        else:
            return None, False, "Unexpected track_list structure"

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

        lyrics_text = lyrics_body.split(
            "******* This Lyrics is NOT for Commercial use"
        )[0].strip()

        if not lyrics_text:
            return None, False, "Lyrics empty after stripping attribution"

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
    word_count: Optional[int] = None
    if lyrics:
        word_count = len(lyrics.split())

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
    song_id = song["song_id"]
    title = song["title"]
    artist = song["artist"]

    # ── Cache check ──
    if config.lyrics.cache_enabled:
        cached = cache.get(song_id)
        if cached is not None:
            log.debug(f"Cache hit: {song_id} '{title}'")
            result = cached.copy()
            result["lyrics_cache_hit"] = True
            return result

    genius_error: Optional[str] = None
    musixmatch_error: Optional[str] = None
    genius_tried = False
    musixmatch_tried = False

    # ── Genius primary ──
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
                f"Musixmatch: lyrics retrieved (truncated) for "
                f"'{title}' by '{artist}'"
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

    return build_lyrics_record(
        song_id=song_id,
        lyrics=None,
        source=None,
        truncated=False,
        cache_hit=False,
    )


# ── Stage entry point ─────────────────────────────────────────────────────────

def run(
    config: PipelineConfig,
    songs_df: Optional[pd.DataFrame] = None,
    run_id: str = "unknown",
) -> pd.DataFrame:
    """
    Execute Stage 2: lyrics fetch for all songs in the S1 checkpoint.
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
            f"S2: [{i}/{total}] '{row['title']}' by "
            f"'{row['artist']}' ({row['year']})"
        )
        result = _fetch_song_lyrics(
            song=row,
            config=config,
            cache=cache,
            missing_log=missing_log,
            run_id=run_id,
        )
        lyrics_records.append(result)

    cache.close()

    lyrics_df = pd.DataFrame(lyrics_records)

    merged = songs_df.merge(
        lyrics_df.drop(columns=["song_id"]),
        left_on="song_id",
        right_on=lyrics_df["song_id"],
        how="left",
    )

    if "key_0" in merged.columns:
        merged = merged.drop(columns=["key_0"])

    for col in _SCHEMA_2_NEW_COLUMNS:
        if col not in merged.columns:
            merged[col] = None

    found = (merged["lyrics_status"] == "found").sum()
    truncated_count = (merged["lyrics_status"] == "truncated").sum()
    missing = (merged["lyrics_status"] == "missing").sum()

    log.info(
        f"S2: lyrics fetch complete. "
        f"found={found} truncated={truncated_count} "
        f"missing={missing} total={total}"
    )

    write_checkpoint(stage, merged, config)
    log.info("S2: checkpoint written.")

    return merged
