"""
Tests for src/stages/s2_lyrics.py

Covers:
  - Schema 2 field presence and correctness
  - lyrics_status enum values (found, truncated, missing)
  - lyrics_word_count accuracy
  - lyrics_truncated flag behavior
  - cache hit path skips API call
  - Genius miss triggers Musixmatch attempt
  - both APIs failing produces missing record
  - missing lyrics record written to missing_lyrics.jsonl
  - Schema 9 fields present in missing lyrics log record
  - cache_enabled=False bypasses cache reads and writes
  - checkpoint read/write behavior
"""
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.stages.s2_lyrics import (
    build_lyrics_record,
    fetch_genius,
    fetch_musixmatch,
    run,
)

# ── Schema 2 new columns (locked P5) ─────────────────────────────────────────
_SCHEMA_2_NEW_COLUMNS = [
    "lyrics",
    "lyrics_status",
    "lyrics_source",
    "lyrics_truncated",
    "lyrics_word_count",
    "lyrics_fetched_at",
    "lyrics_cache_hit",
]

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


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_songs_df(sample_song_record, sample_song_record_2):
    """Two-row DataFrame matching Schema 1."""
    return pd.DataFrame([sample_song_record, sample_song_record_2])


@pytest.fixture()
def found_lyrics_text():
    return (
        "Wise men say only fools rush in\n"
        "But I can't help falling in love with you\n"
        "Take my hand take my whole life too\n"
        "For I can't help falling in love with you\n"
    )


@pytest.fixture()
def truncated_lyrics_text():
    return "Wise men say only fools rush in\nBut I can't help falling"


# ── build_lyrics_record ───────────────────────────────────────────────────────

class TestBuildLyricsRecord:
    def test_found_status_when_lyrics_present_not_truncated(self, found_lyrics_text):
        record = build_lyrics_record(
            song_id="abc123",
            lyrics=found_lyrics_text,
            source="genius",
            truncated=False,
            cache_hit=False,
        )
        assert record["lyrics_status"] == "found"
        assert record["lyrics_source"] == "genius"
        assert record["lyrics_truncated"] is False
        assert record["lyrics_cache_hit"] is False

    def test_truncated_status_when_truncated_flag_set(self, truncated_lyrics_text):
        record = build_lyrics_record(
            song_id="abc123",
            lyrics=truncated_lyrics_text,
            source="musixmatch",
            truncated=True,
            cache_hit=False,
        )
        assert record["lyrics_status"] == "truncated"
        assert record["lyrics_source"] == "musixmatch"
        assert record["lyrics_truncated"] is True

    def test_missing_status_when_lyrics_none(self):
        record = build_lyrics_record(
            song_id="abc123",
            lyrics=None,
            source=None,
            truncated=False,
            cache_hit=False,
        )
        assert record["lyrics_status"] == "missing"
        assert record["lyrics"] is None
        assert record["lyrics_source"] is None
        assert record["lyrics_word_count"] is None

    def test_word_count_matches_actual_words(self, found_lyrics_text):
        record = build_lyrics_record(
            song_id="abc123",
            lyrics=found_lyrics_text,
            source="genius",
            truncated=False,
            cache_hit=False,
        )
        expected = len(found_lyrics_text.split())
        assert record["lyrics_word_count"] == expected

    def test_word_count_none_when_lyrics_missing(self):
        record = build_lyrics_record(
            song_id="abc123",
            lyrics=None,
            source=None,
            truncated=False,
            cache_hit=False,
        )
        assert record["lyrics_word_count"] is None

    def test_cache_hit_flag_set_correctly(self, found_lyrics_text):
        record = build_lyrics_record(
            song_id="abc123",
            lyrics=found_lyrics_text,
            source="genius",
            truncated=False,
            cache_hit=True,
        )
        assert record["lyrics_cache_hit"] is True

    def test_fetched_at_is_iso_string(self, found_lyrics_text):
        record = build_lyrics_record(
            song_id="abc123",
            lyrics=found_lyrics_text,
            source="genius",
            truncated=False,
            cache_hit=False,
        )
        assert isinstance(record["lyrics_fetched_at"], str)
        assert "T" in record["lyrics_fetched_at"]

    def test_song_id_preserved(self):
        record = build_lyrics_record(
            song_id="a3f8c1d24e7b9051",
            lyrics=None,
            source=None,
            truncated=False,
            cache_hit=False,
        )
        assert record["song_id"] == "a3f8c1d24e7b9051"


# ── run — cache behavior ──────────────────────────────────────────────────────

class TestRunCacheBehavior:
    def test_cache_hit_skips_api_call(
        self, minimal_config, tmp_path, sample_songs_df, found_lyrics_text
    ):
        """When cache is enabled and has a hit, no API call should be made."""
        minimal_config.checkpoints.dir = str(tmp_path / "checkpoints")
        minimal_config.cache.lyrics_dir = str(tmp_path / "cache" / "lyrics")
        minimal_config.cache.inference_dir = str(tmp_path / "cache" / "inference")
        minimal_config.logging.missing_lyrics_log = str(
            tmp_path / "missing_lyrics.jsonl"
        )
        Path(minimal_config.checkpoints.dir).mkdir(parents=True, exist_ok=True)
        minimal_config.lyrics.cache_enabled = True
        minimal_config.checkpoints.force_rerun.s2_lyrics = True

        # Pre-populate cache for both songs
        from src.core.cache import get_lyrics_cache

        cache = get_lyrics_cache(minimal_config)
        for row in sample_songs_df.to_dict("records"):
            cached_record = build_lyrics_record(
                song_id=row["song_id"],
                lyrics=found_lyrics_text,
                source="genius",
                truncated=False,
                cache_hit=True,
            )
            cache.set(row["song_id"], cached_record)
        cache.close()

        with patch("src.stages.s2_lyrics.fetch_genius") as mock_genius:
            with patch("src.stages.s2_lyrics.fetch_musixmatch") as mock_mxm:
                df = run(minimal_config, songs_df=sample_songs_df, run_id="test")

        mock_genius.assert_not_called()
        mock_mxm.assert_not_called()
        assert (df["lyrics_status"] == "found").all()

    def test_cache_disabled_always_calls_api(
        self, minimal_config, tmp_path, sample_songs_df
    ):
        """When cache_enabled=False, API must be called regardless of cache state."""
        minimal_config.checkpoints.dir = str(tmp_path / "checkpoints")
        minimal_config.cache.lyrics_dir = str(tmp_path / "cache" / "lyrics")
        minimal_config.cache.inference_dir = str(tmp_path / "cache" / "inference")
        minimal_config.logging.missing_lyrics_log = str(
            tmp_path / "missing_lyrics.jsonl"
        )
        Path(minimal_config.checkpoints.dir).mkdir(parents=True, exist_ok=True)
        minimal_config.lyrics.cache_enabled = False
        minimal_config.checkpoints.force_rerun.s2_lyrics = True

        with patch("src.stages.s2_lyrics.fetch_genius", return_value=(None, "not found")):
            with patch(
                "src.stages.s2_lyrics.fetch_musixmatch",
                return_value=(None, False, "not found"),
            ):
                df = run(minimal_config, songs_df=sample_songs_df, run_id="test")

        assert (df["lyrics_status"] == "missing").all()


# ── run — fallback behavior ───────────────────────────────────────────────────

class TestRunFallbackBehavior:
    def _base_config(self, minimal_config, tmp_path):
        minimal_config.checkpoints.dir = str(tmp_path / "checkpoints")
        minimal_config.cache.lyrics_dir = str(tmp_path / "cache" / "lyrics")
        minimal_config.cache.inference_dir = str(tmp_path / "cache" / "inference")
        minimal_config.logging.missing_lyrics_log = str(
            tmp_path / "missing_lyrics.jsonl"
        )
        Path(minimal_config.checkpoints.dir).mkdir(parents=True, exist_ok=True)
        minimal_config.lyrics.cache_enabled = False
        minimal_config.checkpoints.force_rerun.s2_lyrics = True
        minimal_config.genius_api_token = "fake_token"
        minimal_config.musixmatch_api_key = "fake_key"
        return minimal_config

    def test_genius_miss_triggers_musixmatch(
        self, minimal_config, tmp_path, sample_songs_df, found_lyrics_text
    ):
        config = self._base_config(minimal_config, tmp_path)

        with patch(
            "src.stages.s2_lyrics.fetch_genius", return_value=(None, "not found")
        ):
            with patch(
                "src.stages.s2_lyrics.fetch_musixmatch",
                return_value=(found_lyrics_text, True, None),
            ) as mock_mxm:
                df = run(config, songs_df=sample_songs_df, run_id="test")

        assert mock_mxm.call_count == len(sample_songs_df)
        assert (df["lyrics_status"] == "truncated").all()
        assert (df["lyrics_source"] == "musixmatch").all()
        assert (df["lyrics_truncated"] == True).all()

    def test_both_apis_fail_produces_missing(
        self, minimal_config, tmp_path, sample_songs_df
    ):
        config = self._base_config(minimal_config, tmp_path)

        with patch("src.stages.s2_lyrics.fetch_genius", return_value=(None, "404")):
            with patch(
                "src.stages.s2_lyrics.fetch_musixmatch",
                return_value=(None, False, "no results"),
            ):
                df = run(config, songs_df=sample_songs_df, run_id="test")

        assert (df["lyrics_status"] == "missing").all()
        assert (df["lyrics"].isna()).all()
        assert (df["lyrics_source"].isna()).all()

    def test_genius_success_skips_musixmatch(
        self, minimal_config, tmp_path, sample_songs_df, found_lyrics_text
    ):
        config = self._base_config(minimal_config, tmp_path)

        with patch(
            "src.stages.s2_lyrics.fetch_genius",
            return_value=(found_lyrics_text, None),
        ):
            with patch("src.stages.s2_lyrics.fetch_musixmatch") as mock_mxm:
                df = run(config, songs_df=sample_songs_df, run_id="test")

        mock_mxm.assert_not_called()
        assert (df["lyrics_status"] == "found").all()
        assert (df["lyrics_source"] == "genius").all()


# ── run — missing lyrics log ──────────────────────────────────────────────────

class TestMissingLyricsLog:
    def test_missing_lyrics_record_written_to_jsonl(
        self, minimal_config, tmp_path, sample_songs_df
    ):
        """Both APIs failing must produce a Schema 9 record in missing_lyrics.jsonl."""
        minimal_config.checkpoints.dir = str(tmp_path / "checkpoints")
        minimal_config.cache.lyrics_dir = str(tmp_path / "cache" / "lyrics")
        minimal_config.cache.inference_dir = str(tmp_path / "cache" / "inference")
        log_path = tmp_path / "missing_lyrics.jsonl"
        minimal_config.logging.missing_lyrics_log = str(log_path)
        Path(minimal_config.checkpoints.dir).mkdir(parents=True, exist_ok=True)
        minimal_config.lyrics.cache_enabled = False
        minimal_config.checkpoints.force_rerun.s2_lyrics = True
        minimal_config.genius_api_token = "fake_token"
        minimal_config.musixmatch_api_key = "fake_key"

        with patch("src.stages.s2_lyrics.fetch_genius", return_value=(None, "404")):
            with patch(
                "src.stages.s2_lyrics.fetch_musixmatch",
                return_value=(None, False, "no results"),
            ):
                run(minimal_config, songs_df=sample_songs_df, run_id="test_run")

        assert log_path.exists(), "missing_lyrics.jsonl was not created"

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == len(sample_songs_df)

        # Validate Schema 9 fields on first record
        record = json.loads(lines[0])
        schema_9_fields = [
            "song_id", "year", "title", "artist",
            "genius_tried", "genius_error",
            "musixmatch_tried", "musixmatch_error",
            "logged_at", "pipeline_run_id",
        ]
        for field in schema_9_fields:
            assert field in record, f"Schema 9 field missing from log record: {field}"

        assert record["genius_tried"] is True
        assert record["musixmatch_tried"] is True
        assert record["pipeline_run_id"] == "test_run"


# ── run — schema and checkpoint ───────────────────────────────────────────────

class TestRunSchema:
    def _base_config(self, minimal_config, tmp_path):
        minimal_config.checkpoints.dir = str(tmp_path / "checkpoints")
        minimal_config.cache.lyrics_dir = str(tmp_path / "cache" / "lyrics")
        minimal_config.cache.inference_dir = str(tmp_path / "cache" / "inference")
        minimal_config.logging.missing_lyrics_log = str(
            tmp_path / "missing_lyrics.jsonl"
        )
        Path(minimal_config.checkpoints.dir).mkdir(parents=True, exist_ok=True)
        minimal_config.lyrics.cache_enabled = False
        minimal_config.checkpoints.force_rerun.s2_lyrics = True
        minimal_config.genius_api_token = "fake_token"
        minimal_config.musixmatch_api_key = "fake_key"
        return minimal_config

    def test_all_schema_2_columns_present(
        self, minimal_config, tmp_path, sample_songs_df, found_lyrics_text
    ):
        config = self._base_config(minimal_config, tmp_path)

        with patch(
            "src.stages.s2_lyrics.fetch_genius",
            return_value=(found_lyrics_text, None),
        ):
            df = run(config, songs_df=sample_songs_df, run_id="test")

        for col in _SCHEMA_1_COLUMNS + _SCHEMA_2_NEW_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_checkpoint_written(
        self, minimal_config, tmp_path, sample_songs_df, found_lyrics_text
    ):
        config = self._base_config(minimal_config, tmp_path)

        with patch(
            "src.stages.s2_lyrics.fetch_genius",
            return_value=(found_lyrics_text, None),
        ):
            run(config, songs_df=sample_songs_df, run_id="test")

        checkpoint = Path(minimal_config.checkpoints.dir) / "02_lyrics.parquet"
        assert checkpoint.exists()

    def test_checkpoint_read_when_exists(
        self, minimal_config, tmp_path, sample_songs_df, found_lyrics_text
    ):
        config = self._base_config(minimal_config, tmp_path)

        with patch(
            "src.stages.s2_lyrics.fetch_genius",
            return_value=(found_lyrics_text, None),
        ):
            run(config, songs_df=sample_songs_df, run_id="test")

        config.checkpoints.force_rerun.s2_lyrics = False

        with patch("src.stages.s2_lyrics.fetch_genius") as mock_genius:
            df = run(config, songs_df=sample_songs_df, run_id="test")

        mock_genius.assert_not_called()
        assert len(df) == len(sample_songs_df)

    def test_truncated_lyrics_word_count_positive(
        self, minimal_config, tmp_path, sample_songs_df, truncated_lyrics_text
    ):
        config = self._base_config(minimal_config, tmp_path)

        with patch("src.stages.s2_lyrics.fetch_genius", return_value=(None, "miss")):
            with patch(
                "src.stages.s2_lyrics.fetch_musixmatch",
                return_value=(truncated_lyrics_text, True, None),
            ):
                df = run(config, songs_df=sample_songs_df, run_id="test")

        assert (df["lyrics_word_count"] > 0).all()
        assert (df["lyrics_truncated"] == True).all()
