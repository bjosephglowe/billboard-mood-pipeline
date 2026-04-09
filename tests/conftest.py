import os
import shutil
from pathlib import Path
from typing import Generator

import pytest

from src.core.config import load_config, PipelineConfig
from src.core.song_id import generate_song_id

# ── Paths ─────────────────────────────────────────────────────────────────────

_FIXTURES_DIR = Path("tests/fixtures")
_TEST_CONFIG_PATH = str(_FIXTURES_DIR / "test_config.yaml")

# Directories created per test session and torn down afterwards.
_TEST_RUNTIME_DIRS = [
    _FIXTURES_DIR / "cache" / "lyrics",
    _FIXTURES_DIR / "cache" / "inference",
    _FIXTURES_DIR / "checkpoints",
    _FIXTURES_DIR / "logs",
    _FIXTURES_DIR / "outputs" / "viz",
]


# ── Session-scoped setup / teardown ───────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def create_test_directories() -> Generator[None, None, None]:
    """
    Create all runtime directories required by tests at session start.
    Remove generated runtime files (not fixture source files) at session end.
    """
    for d in _TEST_RUNTIME_DIRS:
        d.mkdir(parents=True, exist_ok=True)

    yield

    # Teardown: remove generated runtime directories only.
    # Preserve tests/fixtures/data/ and tests/fixtures/test_config.yaml.
    for d in _TEST_RUNTIME_DIRS:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)


# ── Config fixture ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def minimal_config() -> PipelineConfig:
    """
    Load and return the test PipelineConfig from tests/fixtures/test_config.yaml.

    API keys are not required — require_env_keys=False so tests that do not
    make live API calls pass without a populated .env file.

    Scope: session — loaded once, shared across all tests.
    """
    return load_config(_TEST_CONFIG_PATH, require_env_keys=False)


# ── Song record fixtures ──────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_song_record() -> dict:
    """
    One fully populated Schema 1 song record.

    song_id is generated via the canonical generate_song_id() function
    so tests that exercise song_id logic use a real, known value.
    """
    title = "Can't Help Falling in Love"
    artist = "UB40"
    year = 1993
    return {
        "song_id": generate_song_id(title, artist, year),
        "year": year,
        "rank": 3,
        "title": title,
        "artist": artist,
        "decade": "1990s",
        "title_normalized": "cant help falling in love",
        "artist_normalized": "ub40",
        "collision_flag": False,
    }


@pytest.fixture(scope="session")
def sample_song_record_2() -> dict:
    """Second distinct song record for multi-song test scenarios."""
    title = "Mmmbop"
    artist = "Hanson"
    year = 1997
    return {
        "song_id": generate_song_id(title, artist, year),
        "year": year,
        "rank": 1,
        "title": title,
        "artist": artist,
        "decade": "1990s",
        "title_normalized": "mmmbop",
        "artist_normalized": "hanson",
        "collision_flag": False,
    }


# ── Lyrics fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_lyrics_found() -> str:
    """
    ~250-word lyric string with:
      - Section headers ([Verse 1], [Chorus]) for preprocessing tests
      - Known pronoun distribution: dominant second-person / relational
        (you, your, we, us) for subject_focus tests
      - Sufficient tokens for MTLD (> 50)
      - Concrete noun chunks for imagery density tests
    """
    return (
        "[Verse 1]\n"
        "Wise men say only fools rush in\n"
        "But I can't help falling in love with you\n"
        "Shall I stay would it be a sin\n"
        "If I can't help falling in love with you\n"
        "\n"
        "[Chorus]\n"
        "Like a river flows surely to the sea\n"
        "Darling so it goes some things are meant to be\n"
        "Take my hand take my whole life too\n"
        "For I can't help falling in love with you\n"
        "\n"
        "[Verse 2]\n"
        "Like a river flows surely to the sea\n"
        "Darling so it goes some things are meant to be\n"
        "Take my hand take my whole life too\n"
        "For I can't help falling in love with you\n"
        "For I can't help falling in love with you\n"
    )


@pytest.fixture(scope="session")
def sample_lyrics_truncated() -> str:
    """
    ~60-word lyric string simulating a Musixmatch 30% truncation result.
    Sufficient for basic analysis but below MTLD token threshold edge cases.
    """
    return (
        "Wise men say only fools rush in\n"
        "But I can't help falling in love with you\n"
        "Shall I stay would it be a sin\n"
        "If I can't help falling in love with you\n"
    )


@pytest.fixture(scope="session")
def sample_lyrics_missing() -> None:
    """Null lyrics representing a failed fetch. Returns None."""
    return None


@pytest.fixture(scope="session")
def sample_lyrics_self_focused() -> str:
    """
    Lyric string with dominant first-person pronouns (I, me, my, myself)
    for subject_focus: self classification tests.
    """
    return (
        "I woke up this morning thinking about myself\n"
        "My dreams my fears my secrets on the shelf\n"
        "I built this world with my own two hands\n"
        "Me and myself we made our plans\n"
        "I am the only one I need\n"
        "My heart my soul my will to lead\n"
        "I walk alone I stand my ground\n"
        "Myself is all I've ever found\n"
    )


@pytest.fixture(scope="session")
def sample_lyrics_society_focused() -> str:
    """
    Lyric string with dominant third-person / societal pronouns
    for subject_focus: society classification tests.
    """
    return (
        "They walk the streets with nowhere to go\n"
        "People crying out but nobody knows\n"
        "The world keeps turning people keep on falling\n"
        "Everyone is lost and no one is calling\n"
        "They built the walls that keep us all inside\n"
        "The world outside has nowhere left to hide\n"
        "People rising up they won't be ignored\n"
        "Everyone deserves more than this world has stored\n"
    )


# ── TF-IDF corpus fixture ─────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def decade_corpus_lyrics() -> list:
    """
    List of 10 lyric strings representing a minimal decade corpus for TF-IDF
    fitting tests. Each string has distinct vocabulary to verify that TF-IDF
    weighting produces different top keywords per song.
    """
    return [
        "love romance heart together forever hold",
        "rain storm dark night alone shadow pain",
        "dance party celebrate good times feeling free",
        "money gold success ambition power rise",
        "remember yesterday nostalgia past home return",
        "fight struggle resist rise up rebel stand",
        "god prayer faith heaven spirit light soul",
        "desire longing waiting missing you far away",
        "anger rage fire burn battle conflict war",
        "joy sunshine smile bright happy world alive",
    ]


# ── Merged record fixtures ────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_merged_record_complete(sample_song_record: dict) -> dict:
    """
    One fully populated Schema 8 merged record with record_complete=True.
    All required analysis fields are non-null and non-uncertain.
    """
    base = sample_song_record.copy()
    base.update({
        "lyrics_status": "found",
        "lyrics_source": "genius",
        "lyrics_truncated": False,
        "lyrics_word_count": 312,
        "sentiment_score": 0.74,
        "sentiment_bin": "positive",
        "sentiment_confidence": 0.87,
        "sentiment_flag": None,
        "sentiment_chunk_count": 2,
        "mood_primary": "joy",
        "mood_primary_confidence": 0.71,
        "mood_secondary": "sadness",
        "mood_secondary_confidence": 0.22,
        "mood_flag": None,
        "theme_primary": "love_and_romance",
        "theme_primary_confidence": 0.81,
        "theme_secondary": "nostalgia_and_memory",
        "theme_secondary_confidence": 0.63,
        "theme_source": "minilm",
        "theme_flag": None,
        "jungian_primary": "anima_animus",
        "jungian_secondary": None,
        "jungian_confidence": "high",
        "jungian_evidence": [
            "wise men say only fools rush in",
            "take my hand take my whole life too",
        ],
        "jungian_flag": None,
        "jungian_source": "haiku",
        "mtld_score": 68.4,
        "imagery_density": 0.31,
        "avg_line_length": 7.2,
        "tfidf_keywords": [
            "wise", "fools", "rush", "rivers", "seas",
            "darling", "hand", "life", "still", "falling",
        ],
        "subject_focus": "relationship",
        "semantic_vector": None,
        "record_complete": True,
        "skip_reason": None,
        "pipeline_run_id": "1990s_19930101T000000",
    })
    return base


@pytest.fixture(scope="session")
def sample_merged_record_incomplete(sample_song_record_2: dict) -> dict:
    """
    One Schema 8 merged record with lyrics_status=missing and all analysis
    fields null. record_complete=False.
    """
    base = sample_song_record_2.copy()
    base.update({
        "lyrics_status": "missing",
        "lyrics_source": None,
        "lyrics_truncated": False,
        "lyrics_word_count": None,
        "sentiment_score": None,
        "sentiment_bin": None,
        "sentiment_confidence": None,
        "sentiment_flag": None,
        "sentiment_chunk_count": None,
        "mood_primary": None,
        "mood_primary_confidence": None,
        "mood_secondary": None,
        "mood_secondary_confidence": None,
        "mood_flag": None,
        "theme_primary": None,
        "theme_primary_confidence": None,
        "theme_secondary": None,
        "theme_secondary_confidence": None,
        "theme_source": None,
        "theme_flag": None,
        "jungian_primary": None,
        "jungian_secondary": None,
        "jungian_confidence": None,
        "jungian_evidence": None,
        "jungian_flag": None,
        "jungian_source": None,
        "mtld_score": None,
        "imagery_density": None,
        "avg_line_length": None,
        "tfidf_keywords": None,
        "subject_focus": None,
        "semantic_vector": None,
        "record_complete": False,
        "skip_reason": None,
        "pipeline_run_id": "1990s_19930101T000000",
    })
    return base


# ── Pytest markers ────────────────────────────────────────────────────────────

def pytest_configure(config):
    """
    Register custom pytest markers.

    slow: marks tests that require model weight downloads.
          Deselect with: pytest -m "not slow"
    """
    config.addinivalue_line(
        "markers",
        "slow: marks tests that require model weight downloads "
        "(deselect with -m 'not slow')",
    )
