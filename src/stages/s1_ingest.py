import json
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from src.core.checkpoint import checkpoint_exists, write_checkpoint
from src.core.config import PipelineConfig
from src.core.logger import get_logger
from src.core.song_id import generate_song_id, normalize_artist, normalize_title

log = get_logger("s1_ingest")

# ── Schema 1 column order (locked P5) ────────────────────────────────────────
_SCHEMA_1_COLUMNS = [
    "song_id",
    "year",
    "rank",
    "title",
    "artist",
    "decade",
    "title_normalized",
    "artist_normalized",
    "collision_flag",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _derive_decade(year: int) -> str:
    """
    Derive the decade string from a 4-digit year.

    Examples:
        1993 -> "1990s"
        1958 -> "1950s"
    """
    return f"{(year // 10) * 10}s"


def fetch_dataset(url: str, local_path: str) -> list[dict]:
    """
    Load the Billboard dataset from local_path if it exists, otherwise
    fetch from url and save to local_path.

    On fetch failure, raises with a clear message directing the user to
    download the file manually — no silent degradation on the data source
    per P3 fallback policy.

    Args:
        url: remote URL for the Billboard JSON dataset
        local_path: local file path to read from or write to

    Returns:
        Parsed list of raw chart entry dicts.

    Raises:
        FileNotFoundError: if local file absent and remote fetch fails,
                           with instructions for manual download.
        ValueError: if the fetched or loaded content is not valid JSON
                    or does not contain a list.
    """
    path = Path(local_path)

    if path.exists():
        log.info(f"Loading dataset from local cache: {local_path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        log.info(f"Local dataset not found. Fetching from: {url}")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise FileNotFoundError(
                f"Failed to fetch Billboard dataset from {url}.\n"
                f"Error: {exc}\n"
                f"Manual fallback: download the file and save it to {local_path}"
            ) from exc

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(response.text)
        log.info(f"Dataset saved to {local_path}")

        data = response.json()

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON array at root of dataset, got {type(data).__name__}. "
            f"Check the dataset format at {local_path}."
        )

    log.info(f"Dataset loaded: {len(data)} raw entries")
    return data


def normalize_records(
    raw: list[dict],
    year_range: list[int],
    top_n: int,
) -> pd.DataFrame:
    """
    Filter raw Billboard entries to the configured year range and top_n rank,
    normalize metadata fields, and generate song_id for each record.

    Expected raw entry shape (KorbenD dataset):
        {"rank": 1, "song": "Title", "artist": "Artist", "year": 1993, ...}

    Args:
        raw: list of raw dicts from the Billboard JSON dataset
        year_range: two-element list [start_year, end_year] inclusive
        top_n: maximum chart rank to include (1 through top_n)

    Returns:
        pd.DataFrame with Schema 1 columns, sorted by year then rank.
    """
    start_year, end_year = year_range
    records = []

    for entry in raw:
        try:
            year = int(entry.get("year", 0))
            rank = int(entry.get("rank", 0))
        except (TypeError, ValueError):
            log.debug(f"Skipping entry with unparseable year or rank: {entry}")
            continue

        if year < start_year or year > end_year:
            continue
        if rank < 1 or rank > top_n:
            continue

        title = str(entry.get("song", entry.get("title", ""))).strip()
        artist = str(entry.get("artist", "")).strip()

        if not title or not artist:
            log.warning(
                f"Skipping entry with missing title or artist: year={year} rank={rank}"
            )
            continue

        records.append({
            "song_id": generate_song_id(title, artist, year),
            "year": year,
            "rank": rank,
            "title": title,
            "artist": artist,
            "decade": _derive_decade(year),
            "title_normalized": normalize_title(title),
            "artist_normalized": normalize_artist(artist),
            "collision_flag": False,
        })

    if not records:
        log.warning(
            f"No records found for year_range={year_range}, top_n={top_n}. "
            f"Check dataset coverage and config."
        )
        return pd.DataFrame(columns=_SCHEMA_1_COLUMNS)

    df = pd.DataFrame(records, columns=_SCHEMA_1_COLUMNS)
    df = df.sort_values(["year", "rank"]).reset_index(drop=True)

    log.info(
        f"Normalized {len(df)} records for years {start_year}–{end_year}, "
        f"top_n={top_n}"
    )
    return df


def detect_collisions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect song_id collisions (two distinct songs with the same hash).

    On collision, retains the first record and sets collision_flag=True
    on the second record, appending "_2" to its song_id. Logs a WARNING
    for each collision.

    Args:
        df: DataFrame with a song_id column

    Returns:
        DataFrame with collision_flag updated and duplicate song_ids resolved.
    """
    seen: dict[str, int] = {}
    df = df.copy()

    for idx, row in df.iterrows():
        sid = row["song_id"]
        if sid in seen:
            original_idx = seen[sid]
            original = df.loc[original_idx]
            log.warning(
                f"song_id collision detected: '{row['title']}' by '{row['artist']}' "
                f"({row['year']}) collides with '{original['title']}' by "
                f"'{original['artist']}' ({original['year']}). "
                f"Appending '_2' to duplicate song_id."
            )
            df.at[idx, "song_id"] = sid + "_2"
            df.at[idx, "collision_flag"] = True
        else:
            seen[sid] = idx

    collision_count = int(df["collision_flag"].sum())
    if collision_count:
        log.warning(f"Total collisions resolved: {collision_count}")
    else:
        log.info("No song_id collisions detected.")

    return df


def _validate_year_coverage(df: pd.DataFrame, year_range: list[int]) -> None:
    """
    Warn if any year in the configured range has zero records.

    Args:
        df: normalized records DataFrame
        year_range: [start_year, end_year]
    """
    start_year, end_year = year_range
    years_present = set(df["year"].unique())
    for year in range(start_year, end_year + 1):
        if year not in years_present:
            log.warning(f"No records found for year {year} — check dataset coverage.")


# ── Stage entry point ─────────────────────────────────────────────────────────

def run(config: PipelineConfig) -> pd.DataFrame:
    """
    Execute Stage 1: dataset ingest and normalization.

    Reads from checkpoint if available and force_rerun.s1_ingest is False.
    Writes 01_songs.parquet on completion.

    Args:
        config: PipelineConfig instance

    Returns:
        pd.DataFrame matching Schema 1.
    """
    stage = "s1_ingest"

    if checkpoint_exists(stage, config) and not config.checkpoints.force_rerun.s1_ingest:
        from src.core.checkpoint import read_checkpoint
        log.info("S1: checkpoint found — skipping ingest.")
        return read_checkpoint(stage, config)

    log.info("S1: starting dataset ingest.")

    raw = fetch_dataset(
        url=config.dataset.github_url,
        local_path=config.dataset.local_path,
    )

    df = normalize_records(
        raw=raw,
        year_range=config.project.year_range,
        top_n=config.project.top_n,
    )

    df = detect_collisions(df)
    _validate_year_coverage(df, config.project.year_range)

    write_checkpoint(stage, df, config)
    log.info(f"S1: complete. {len(df)} song records written to checkpoint.")

    return df
