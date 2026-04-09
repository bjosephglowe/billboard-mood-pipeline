import hashlib
import re


# ── Normalization constants ───────────────────────────────────────────────────

# Characters removed from title and artist before hashing.
# Locked in G2-OI1.
_STRIP_CHARS_PATTERN = re.compile(r"[.,\'\"!?&()\[\]/]")

# Collapses any whitespace sequence (space, tab, newline) to a single space.
_WHITESPACE_PATTERN = re.compile(r"\s+")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _normalize_field(value: str) -> str:
    """
    Apply normalization rules to a single metadata field (title or artist).

    Rules applied in order (locked in G2-OI1):
      1. Lowercase
      2. Strip leading and trailing whitespace
      3. Collapse internal whitespace sequences to a single space
      4. Remove characters: . , ' " ! ? & ( ) [ ] /

    Args:
        value: raw string from source dataset

    Returns:
        Normalized string ready for hash input construction.
    """
    value = value.lower()
    value = value.strip()
    value = _WHITESPACE_PATTERN.sub(" ", value)
    value = _STRIP_CHARS_PATTERN.sub("", value)
    # Re-strip in case punctuation removal left leading/trailing spaces
    value = value.strip()
    return value


# ── Public interface ──────────────────────────────────────────────────────────

def generate_song_id(title: str, artist: str, year: int) -> str:
    """
    Generate a deterministic 16-character hex song identifier.

    Canonical implementation per G2-OI1. Every module that produces or
    consumes a song_id must import exclusively from this module. No module
    may construct a song_id independently.

    Algorithm:
      1. Normalize title and artist using _normalize_field()
      2. Construct input string: "{normalized_title}|{normalized_artist}|{year}"
      3. Compute SHA-256 digest of the UTF-8 encoded input string
      4. Return the first 16 hexadecimal characters of the digest (lowercase)

    Args:
        title: song title as it appears in the source dataset
        artist: artist name as it appears in the source dataset
        year: 4-digit release/chart year as integer

    Returns:
        16-character lowercase hex string e.g. "a3f8c1d24e7b9051"

    Examples:
        >>> generate_song_id("Can't Help Falling in Love", "UB40", 1993)
        # consistent output across all runs and machines
    """
    normalized_title = _normalize_field(title)
    normalized_artist = _normalize_field(artist)
    hash_input = f"{normalized_title}|{normalized_artist}|{year}"
    digest = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
    return digest[:16]


def normalize_title(title: str) -> str:
    """
    Return the normalized title string used as hash input.

    Exposed for storage in Schema 1 field title_normalized.
    """
    return _normalize_field(title)


def normalize_artist(artist: str) -> str:
    """
    Return the normalized artist string used as hash input.

    Exposed for storage in Schema 1 field artist_normalized.
    """
    return _normalize_field(artist)
